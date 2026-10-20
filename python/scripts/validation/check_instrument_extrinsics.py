#!/usr/bin/env python3
"""Live Galaxy instrument-tip projection into Vcamera RGB with XYZ dials."""
import sys
import argparse
import base64
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from zoneinfo import ZoneInfo

import cv2
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.validation.extrinsic_check import ROOT, InstrumentTracker, intrinsics, matrix_file, project, transform_tip, export_adjustment
from scripts.workflow.config import load_config
from scripts.capture import open_sources


class Dial(ttk.Frame):
    def __init__(self, parent, axis, variable, callback, limit, step):
        super().__init__(parent)
        self.variable, self.callback, self.limit, self.step = variable, callback, limit, step
        ttk.Label(self, text=f'Δ{axis} (mm)').pack()
        self.canvas = tk.Canvas(self, width=86, height=86, bg='#202a38', highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.start)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<MouseWheel>', lambda event: self.change(self.value() + self.step*(1 if event.delta > 0 else -1)))
        box = ttk.Spinbox(self, from_=-limit, to=limit, increment=step, textvariable=variable,
                          width=10, command=self.redraw)
        box.pack(pady=4)
        box.bind('<Return>', lambda event: self.change(self.value()))
        box.bind('<FocusOut>', lambda event: self.change(self.value()))
        self.redraw()

    def value(self):
        try:
            value = float(self.variable.get())
            return value if np.isfinite(value) else 0.
        except ValueError:
            return 0.

    def change(self, value):
        self.variable.set(f'{max(-self.limit, min(self.limit, value)):.3f}')
        self.redraw()

    def start(self, event):
        self.origin = (event.y, self.value())

    def drag(self, event):
        self.change(self.origin[1] + (self.origin[0] - event.y)*self.step)

    def redraw(self):
        c = self.canvas
        c.delete('all')
        c.create_oval(10, 10, 76, 76, outline='#91a5ba', width=2)
        angle = np.deg2rad(-90 + 135*self.value()/self.limit)
        c.create_line(43, 43, 43+28*np.cos(angle), 43+28*np.sin(angle), fill='#56d7be', width=4)
        self.callback()


class App:
    def __init__(self, root, config, tracker, calibration_dir, workflow, geometry):
        self.root, self.config, self.tracker = root, config, tracker
        point_name = config.get('point_name', '末端')
        self.workflow, self.geometry = workflow, geometry
        self.calibration_dir = calibration_dir
        self.kg, self.dg, self.sg = intrinsics(calibration_dir / 'sensor_galaxy_gray_intrinsics.yml')
        self.kv, self.dv, self.sv = intrinsics(calibration_dir / 'sensor_vcamera_color_intrinsics.yml')
        self.rotation, translation = matrix_file(calibration_dir / 'pair_galaxy_gray_to_vcamera_color_extrinsics.yml', 'R', 'T')
        self.translation = translation.reshape(3)
        if self.rotation.shape != (3, 3) or not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-5):
            raise ValueError('Invalid extrinsic rotation')
        self.queue, self.stop = queue.Queue(maxsize=1), threading.Event()
        self.packet, self.target, self.samples = None, None, []
        self.sequence = 0
        self.sample_sequence = None
        self.error = None
        self.delta_vars = [tk.StringVar(value='0.000') for _ in range(3)]
        root.title('Galaxy → Vcamera RGB | Instrument extrinsic check')
        root.geometry('1450x850')
        root.protocol('WM_DELETE_WINDOW', self.close)
        controls = ttk.Frame(root, padding=10)
        controls.pack(side='right', fill='y')
        ttk.Label(controls, text=f'大恒{point_name} → 外参转换 → 图漾 RGB', font=('', 11, 'bold')).pack(pady=8)
        ttk.Label(controls, text='P_RGB = R × P_Galaxy + T + ΔT\nXYZ 均为相机光学坐标，单位 mm').pack(pady=4)
        dials = ttk.Frame(controls)
        dials.pack()
        self.dials = []
        for axis, variable in zip('XYZ', self.delta_vars):
            dial = Dial(dials, axis, variable, self.changed, config['translation_range_mm'], config['translation_step_mm'])
            dial.pack(side='left', padx=3)
            self.dials.append(dial)
        ttk.Label(controls, text='Drag up/down or scroll; type exact mm.\nX: right   Y: down   Z: forward\nT_new = T_original + ΔT; R stays fixed').pack(pady=8)
        ttk.Button(controls, text='恢复原外参（ΔXYZ 清零）', command=self.reset).pack(fill='x', pady=4)
        ttk.Button(controls, text='清除 RGB 目标点', command=self.clear_target).pack(fill='x', pady=4)
        ttk.Button(controls, text='记录当前转换与目标偏差', command=self.sample).pack(fill='x', pady=4)
        ttk.Button(controls, text='另存调整外参与记录', command=self.save).pack(fill='x', pady=4)
        self.show_marker_projection = tk.BooleanVar(value=config.get('show_marker_projection', False))
        ttk.Checkbutton(controls, text='同时显示标记角点的跨相机投影', variable=self.show_marker_projection).pack(pady=4)
        self.transform_status = tk.StringVar()
        ttk.Label(controls, textvariable=self.transform_status, justify='left', font=('Consolas', 9)).pack(pady=4)
        ttk.Label(controls, text='Click the physical tip / target in RGB.\nKeep instrument stationary when comparing.\nMagenta: original projection\nGreen: adjusted projection\nYellow: manually selected target\n\nA 2D target does not determine 3D XYZ.\nCheck several positions, depths and poses.\nTip geometry/PnP errors also affect residuals.', justify='left').pack(pady=12)
        self.status = tk.StringVar(value='Opening cameras…')
        ttk.Label(controls, textvariable=self.status, wraplength=310, justify='left').pack(fill='x')
        self.panes = ttk.Frame(root)
        self.panes.pack(side='left', fill='both', expand=True)
        self.canvases, self.photos, self.scales = [], [None, None], [None, None]
        for name in (f'大恒：实时角点与计算出的{point_name}', '图漾 RGB：仅显示由大恒转换来的投影'):
            frame = ttk.Frame(self.panes)
            frame.pack(side='left', expand=True, fill='both')
            ttk.Label(frame, text=name).pack()
            canvas = tk.Canvas(frame, bg='#111820', highlightthickness=0)
            canvas.pack(expand=True, fill='both')
            self.canvases.append(canvas)
        self.canvases[1].bind('<Button-1>', self.click_target)
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()
        root.after(40, self.tick)

    def changed(self):
        # UI and acquisition communicate only through completed packets.
        pass

    def delta(self):
        values = np.array([float(variable.get()) for variable in self.delta_vars])
        if not np.isfinite(values).all() or np.max(np.abs(values)) > self.config['translation_range_mm']:
            raise ValueError('XYZ must be finite and within the configured range')
        return values

    def reset(self):
        for dial in self.dials:
            dial.change(0.)

    def clear_target(self):
        self.target = None

    def click_target(self, event):
        mapping = self.scales[1]
        if mapping and self.packet:
            scale, ox, oy, width, height = mapping
            point = np.array([(event.x-ox)/scale, (event.y-oy)/scale])
            if 0 <= point[0] < width and 0 <= point[1] < height:
                self.target = point

    def worker(self):
        adapters = []
        try:
            adapters = open_sources(self.workflow['sensors'])
            sources = {sensor['sensor_type']: adapter for adapter, sensors in adapters for sensor in sensors}
            while not self.stop.is_set():
                galaxy = sources['galaxy'].read(500)
                tg = time.monotonic()
                rgb = sources['vcamera'].read(500)
                tv = time.monotonic()
                if galaxy is None or rgb is None:
                    continue
                if galaxy.shape[1::-1] != self.sg or rgb.shape[1::-1] != self.sv:
                    raise ValueError('Live image resolution differs from calibration; use matching sensor modes')
                pose = self.tracker.detect(galaxy, self.kg, self.dg)
                packet = dict(galaxy=galaxy, rgb=rgb, pose=pose, time=tg,
                              receive_delta_ms=abs(tv-tg)*1000)
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                self.queue.put_nowait(packet)
        except Exception as exc:
            self.error = str(exc)
        finally:
            for adapter, _ in reversed(adapters):
                try:
                    adapter.close()
                except Exception as exc:
                    self.error = self.error or str(exc)

    def current(self):
        if not self.packet or time.monotonic()-self.packet['time'] > self.config['max_frame_age_seconds']:
            return None
        pose = self.packet['pose']
        if not pose:
            return None
        return (pose, *transform_tip(pose['tip'], self.rotation, self.translation,
                                    self.delta(), self.kv, self.dv))

    @staticmethod
    def cross(image, point, color, label):
        if point is None or not np.isfinite(point).all():
            return
        x, y = point
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            return
        location = (round(float(x)), round(float(y)))
        size = max(28, round(image.shape[1]/55))
        thickness = max(4, round(image.shape[1]/350))
        cv2.drawMarker(image, location, color, cv2.MARKER_CROSS, size, thickness, cv2.LINE_AA)
        cv2.putText(image, label, (location[0]+size, location[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    max(.5, image.shape[1]/2200), color, 2)

    def normal_arrow(self, image, tip, normal, k, d, color):
        length = float(self.config.get('normal_length_mm', 80.0))
        start, end = project(tip, k, d), project(tip+normal*length, k, d)
        if start is None or end is None or not np.isfinite([start, end]).all():
            return
        if np.max(np.abs([start, end])) > 1e7:
            return
        thickness = max(4, round(image.shape[1]/350))
        cv2.arrowedLine(image, tuple(np.int32(np.round(start))), tuple(np.int32(np.round(end))),
                        color, thickness, cv2.LINE_AA, tipLength=.18)

    def show(self, index, image):
        canvas = self.canvases[index]
        cw, ch = max(10, canvas.winfo_width()), max(10, canvas.winfo_height())
        scale = min(cw/image.shape[1], ch/image.shape[0])
        shown = cv2.resize(image, (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale))))
        height, width = shown.shape[:2]
        ok, encoded = cv2.imencode('.png', shown)
        if not ok:
            raise ValueError('Cannot encode camera preview')
        photo = tk.PhotoImage(data=base64.b64encode(encoded).decode('ascii'), format='png')
        ox, oy = (cw-width)/2, (ch-height)/2
        canvas.delete('all')
        canvas.create_image(ox, oy, image=photo, anchor='nw')
        self.photos[index] = photo
        self.scales[index] = (width/image.shape[1], ox, oy, image.shape[1], image.shape[0])

    def tick(self):
        if self.stop.is_set():
            return
        try:
            self.packet = self.queue.get_nowait()
            self.sequence += 1
        except queue.Empty:
            pass
        try:
            self.transform_status.set('R (Galaxy → RGB):\n'+np.array2string(self.rotation, precision=5)
                +'\nT 原始 mm: '+str(np.round(self.translation, 3))
                +'\nT 当前 mm: '+str(np.round(self.translation+self.delta(), 3)))
            current = self.current()
            if self.error:
                self.status.set('Camera/tracking error: '+self.error)
            elif self.packet:
                galaxy = cv2.cvtColor(self.packet['galaxy'], cv2.COLOR_GRAY2BGR)
                rgb = self.packet['rgb'].copy()
                if current:
                    pose, original, adjusted, p0, p1 = current
                    for marker_id, points in zip(pose['ids'], pose['pixels'].reshape(-1, 4, 2)):
                        cv2.polylines(galaxy, [np.int32(np.round(points))], True, (255, 180, 0), 2)
                        for corner_index, point in enumerate(points):
                            cv2.circle(galaxy, tuple(np.int32(np.round(point))), 7, (255, 180, 0), 2)
                            cv2.putText(galaxy, f'{marker_id}:{corner_index}', tuple(np.int32(np.round(point))),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 180, 0), 2)
                    if self.show_marker_projection.get():
                        points_galaxy = pose['objects'] @ pose['rotation'].T + pose['center']
                        points_rgb = points_galaxy @ self.rotation.T + self.translation + self.delta()
                        for point in points_rgb:
                            self.cross(rgb, project(point, self.kv, self.dv), (255, 180, 0), '')
                    self.cross(galaxy, project(pose['tip'], self.kg, self.dg), (0, 255, 0), self.config.get('point_label', 'TIP'))
                    self.cross(rgb, p0, (255, 0, 255), 'BASE')
                    self.cross(rgb, p1, (0, 255, 0), 'ADJUSTED')
                    # Positive instrument-local Z, including the same axis fix as the tip.
                    normal_galaxy = pose['rotation'] @ self.tracker.fix @ np.array([0., 0., 1.])
                    normal_rgb = self.rotation @ normal_galaxy
                    self.normal_arrow(galaxy, pose['tip'], normal_galaxy, self.kg, self.dg, (0, 255, 0))
                    self.normal_arrow(rgb, original, normal_rgb, self.kv, self.dv, (255, 0, 255))
                    self.normal_arrow(rgb, adjusted, normal_rgb, self.kv, self.dv, (0, 255, 0))
                    text = (f"大恒可见码: {pose['ids']}\n角点位姿 RMS: {pose['rms']:.3f} px"
                            f"\n1. 大恒{self.config.get('point_name', '末端')} XYZ mm:\n{np.round(pose['tip'],3)}"
                            f"\n2. 原外参转换 RGB XYZ mm:\n{np.round(original,3)}"
                            f"\n3. 调整后 RGB XYZ mm:\n{np.round(adjusted,3)}")
                    if p0 is not None and p1 is not None:
                        text += f'\n4. RGB 原始/调整像素:\n{np.round(p0,2)} → {np.round(p1,2)}'
                    if pose['ambiguity_rms_gap'] is not None and pose['ambiguity_rms_gap'] < .1:
                        text += '\nPlanar pose ambiguity: inspect tip stability.'
                    if p1 is None:
                        text += '\nTip is behind RGB camera.'
                    elif not (0 <= p1[0] < self.sv[0] and 0 <= p1[1] < self.sv[1]):
                        text += '\nProjection is outside RGB image.'
                    if self.target is not None and p1 is not None and p0 is not None:
                        text += f'\nTarget residual: {np.linalg.norm(p0-self.target):.2f} → {np.linalg.norm(p1-self.target):.2f} px'
                else:
                    text = f'No current instrument pose (need ≥{self.tracker.min_markers} markers / fresh frame / acceptable RMS).'
                self.cross(rgb, self.target, (0, 255, 255), 'TARGET')
                self.show(0, galaxy)
                self.show(1, rgb)
                if not self.error:
                    self.status.set(text+f"\nReceive gap: {self.packet['receive_delta_ms']:.1f} ms (no exposure sync)\nRecorded samples: {len(self.samples)}")
        except (ValueError, tk.TclError) as exc:
            self.status.set(str(exc))
        self.root.after(60, self.tick)

    def sample(self):
        try:
            current = self.current()
            if current is None or self.target is None:
                raise ValueError('Need a current pose and a clicked physical tip/target in RGB')
            if self.sample_sequence == self.sequence:
                raise ValueError('This frame is already recorded; wait for a new frame')
            pose, original, adjusted, p0, p1 = current
            if p0 is None or p1 is None:
                raise ValueError('Projection is behind the camera')
            self.samples.append(dict(marker_ids=pose['ids'], pose_rms_px=pose['rms'],
                galaxy_tip_mm=pose['tip'].tolist(), rgb_tip_mm=adjusted.tolist(),
                target_rgb_px=self.target.tolist(), original_projection_px=p0.tolist(),
                adjusted_projection_px=p1.tolist(), original_rgb_tip_mm=original.tolist(),
                R_galaxy_to_rgb=self.rotation.tolist(), T_original_mm=self.translation.tolist(),
                T_adjusted_mm=(self.translation+self.delta()).tolist(),
                delta_T_rgb_mm=self.delta().tolist(),
                original_residual_px=float(np.linalg.norm(p0-self.target)),
                adjusted_residual_px=float(np.linalg.norm(p1-self.target)),
                receive_delta_ms=self.packet['receive_delta_ms']))
            self.sample_sequence = self.sequence
        except ValueError as exc:
            messagebox.showinfo('Record comparison', str(exc))

    def save(self):
        try:
            stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
            path = export_adjustment(ROOT/'output'/('instrument_check_'+stamp), self.rotation,
                self.translation, self.delta(), dict(source_calibration_dir=str(self.calibration_dir),
                workflow_config=self.workflow['_config_path'], geometry_file=self.geometry,
                instrument_config=self.config, samples=self.samples,
                assessment='manual RGB projection comparison; not measured 3D accuracy'))
            messagebox.showinfo('Saved', str(path))
        except (ValueError, OSError) as exc:
            messagebox.showerror('Save failed', str(exc))

    def close(self):
        self.stop.set()
        self.root.withdraw()
        self.finish_close()

    def finish_close(self):
        if self.thread.is_alive():
            self.root.after(100, self.finish_close)
        else:
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/instruments/instrument_extrinsic_check_large_qr.json')
    parser.add_argument('--geometry', help='Navigation instrument geometry YAML (marker_positions + tip_offset)')
    parser.add_argument('--calibration-dir', help='Override calibration output directory')
    args = parser.parse_args()
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else ROOT/path
    root = None
    try:
        config = json.loads(resolve(args.config).read_text(encoding='utf-8-sig'))
        workflow = load_config(resolve(config['workflow_config']))
        if sorted(s['sensor_type'] for s in workflow['sensors']) != ['galaxy', 'vcamera']:
            raise ValueError('Workflow must select one Galaxy and one Vcamera RGB sensor')
        if config['translation_range_mm'] <= 0 or config['translation_step_mm'] <= 0 or config['max_frame_age_seconds'] <= 0:
            raise ValueError('UI ranges, step and maximum frame age must be positive')
        geometry = args.geometry or config.get('geometry_file')
        geometry = str(resolve(geometry)) if geometry else None
        tracker = InstrumentTracker(config, geometry)
        root = tk.Tk()
        App(root, config, tracker, resolve(args.calibration_dir or config['calibration_dir']), workflow, geometry)
        root.mainloop()
        return 0
    except (ValueError, KeyError, AttributeError, OSError, ImportError, cv2.error) as exc:
        print(f'Error: {exc}')
        if root:
            root.destroy()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

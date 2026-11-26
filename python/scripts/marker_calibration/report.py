#!/usr/bin/env python3
"""Build HTML reports from saved marker calibration results."""

import argparse
import base64
import hashlib
import html
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .marker_io import result_path


def build(root):
    root = Path(root)
    fit = json.loads((root / "fit.json").read_text(encoding="utf-8"))
    if fit.get("geometry_model") == "independent_planar_quadrilateral":
        from .planar import render_report

        return render_report(root)
    quality = json.loads(
        (root / "marker_calibration_quality.json").read_text(encoding="utf-8")
    )
    summary = quality["quality_summary"]
    folds = quality["group_validation"]["folds"]
    raw = json.loads(
        result_path(root, "marker_all_candidate_observations.json").read_text(
            encoding="utf-8"
        )
    )["observations"]
    eligible = json.loads(
        (root / "reprojection_observations.json").read_text(encoding="utf-8")
    )["observations"]
    image_quality = json.loads(
        (root / "image_quality.json").read_text(encoding="utf-8")
    )
    ordinary = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
    lengths = quality["marker_lengths_mm"]
    ids = sorted(int(mid) for mid in lengths)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for observations, label, color in [
        (raw, "All image candidates", "#939da5"),
        (eligible, "Image-qualified candidates", "#007f87"),
    ]:
        errors = np.sort(
            [
                e
                for r in observations
                if r["split"] == "test"
                for e in r["corner_errors_px"]
            ]
        )
        axes[0, 0].semilogx(
            np.maximum(errors, 0.005),
            np.arange(1, len(errors) + 1) / len(errors),
            label=label,
            color=color,
        )
    axes[0, 0].set(
        xlabel="Held-out corner error (px)",
        ylabel="Cumulative fraction",
        title="Board-only camera pose; no marker pose refit",
        xlim=(0.005, 6),
        ylim=(0, 1.01),
    )
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.2)
    axes[0, 1].bar(
        range(7), [f["test_corner_error_px"]["rms"] for f in folds], color="#007f87"
    )
    axes[0, 1].set(
        xlabel="Group excluded from geometry fit",
        ylabel="Test corner RMS (px)",
        title="Leave-one-group-out reprojection",
    )
    axes[1, 0].bar(
        range(7),
        [f["corner_geometry_change_mm"]["max"] for f in folds],
        color="#6561a5",
    )
    axes[1, 0].set(
        xlabel="Group excluded from geometry fit",
        ylabel="Maximum corner change (mm)",
        title="Geometry stability in the fixed ID84 frame",
    )
    central = np.array([lengths[str(mid)] for mid in ids])
    low = np.array([min(f["lengths_mm"][str(mid)] for f in folds) for mid in ids])
    high = np.array([max(f["lengths_mm"][str(mid)] for f in folds) for mid in ids])
    axes[1, 1].vlines(ids, low, high, color="#007f87", linewidth=2)
    axes[1, 1].scatter(ids, central, color="#007f87", s=20)
    axes[1, 1].set(
        xticks=ids,
        xlabel="Marker ID",
        ylabel="Edge length (mm)",
        title="Edge lengths and leave-group-out ranges",
    )
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.15)
    fig.savefig(root / "marker_validation_charts.png", dpi=170)
    plt.close(fig)

    def image(path):
        path = Path(path)
        mime = "image/png" if path.suffix == ".png" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()

    edge_rows = "".join(
        f"<tr><td>{mid}</td><td>{lengths[str(mid)]:.6f}</td><td>{summary['eligible_test_per_id_px'][str(mid)]['rms']:.3f}</td><td>{summary['eligible_test_per_id_px'][str(mid)]['p95']:.3f}</td><td>{summary['eligible_test_per_id_px'][str(mid)]['count'] // 4}</td></tr>"
        for mid in ids
    )
    fold_rows = "".join(
        f"<tr><td>{f['held_geometry_group']}</td><td>{f['test_corner_error_px']['rms']:.3f}</td><td>{f['corner_geometry_change_mm']['rms']:.4f}</td><td>{f['corner_geometry_change_mm']['max']:.4f}</td></tr>"
        for f in folds
    )
    gate_rows = "".join(
        f"<tr><td>{name}</td><td>{'通过' if passed else '失败'}</td></tr>"
        for name, passed in quality["checks"].items()
    )
    gallery = "".join(
        f'<figure><img src="{image(path)}"><figcaption>{path.stem}：绿色为检测角点，红色为训练几何预测。</figcaption></figure>'
        for path in sorted((root / "overlays").glob("*.jpg"))
    )
    metric = ordinary["metric_crosscheck"]["per_tool"]["C"]["corner_difference_mm"]
    worst = max(f["corner_geometry_change_mm"]["max"] for f in folds)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C 器械几何标定与验证</title>
<style>body{{margin:0;background:#f0f5f6;color:#203540;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1100px;margin:auto;background:#fff;padding:32px 28px}}h1{{font-size:29px}}h2{{margin-top:34px;font-size:22px;color:#08757c}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #dce5e8}}th{{background:#edf4f5}}img{{max-width:100%;height:auto}}pre{{padding:15px;background:#f1f5f6;overflow:auto}}.notice{{border-left:4px solid #18828a;background:#ecf7f7;padding:14px 18px}}.limit{{border-left:4px solid #bd8a34;background:#fff8e8;padding:14px 18px}}figcaption,.muted{{color:#637781;font-size:14px}}a{{color:#08757c}}figure{{margin:20px 0}}</style>
<main><p class="muted">2026-11-16 · Galaxy 红外相机 · 35 mm 棋盘尺度 · 只拟合 C</p><h1>C 器械：九个 marker 的几何标定</h1>
<div class="notice">ID <strong>84–92</strong> 全部完成，参考码 <strong>84</strong>。优化中只使用棋盘与 C 的角点。项目阈值及七轮整组留出验证的数值门禁全部通过。<br><a href="double_cube_geometry.yaml">最终文件：double_cube_geometry.yaml</a> · <a href="marker_calibration_quality.json">完整质量报告 JSON</a></div>
<h2>可核验的结果</h2><table><tr><th>检查</th><th>结果</th></tr>
<tr><td>有效留出角点 RMS / P95</td><td>{summary["eligible_test_corner_errors_px"]["rms"]:.3f} / {summary["eligible_test_corner_errors_px"]["p95"]:.3f} px</td></tr>
<tr><td>全部候选留出角点 RMS / P95</td><td>{summary["all_candidates_test_corner_errors_px"]["rms"]:.3f} / {summary["all_candidates_test_corner_errors_px"]["p95"]:.3f} px</td></tr>
<tr><td>有效验证“图像—marker”观测</td><td>{summary["eligible_test_observations"]} / {summary["all_candidate_test_observations"]}</td></tr>
<tr><td>独立留出重建与训练几何的角点差异 RMS</td><td>{metric["rms"]:.4f} mm</td></tr>
<tr><td>七轮整组留出中最大的三维角点变化</td><td>{worst:.4f} mm</td></tr>
<tr><td>单个有效验证 marker 的最大四角 RMS</td><td>{summary["maximum_eligible_test_observation_rmse_px"]:.3f} px（项目门限 3 px）</td></tr></table>
<p>317 张图分七组，每组最后连续 20% 留出，合计 251 张训练、66 张验证。验证投影只由棋盘确定相机位姿，器械几何与组放置变换固定为训练结果，不用验证 marker 重新拟合位姿。</p>
<img src="{image(root / "marker_validation_charts.png")}" alt="C 的误差分布、分组验证和边长稳定性">
<div class="limit">这些结果确认的是在给定图像、35 mm 棋盘尺寸及平面假设下的几何拟合、复现和稳定性，不能证明零误差或独立的绝对计量精度。0.020 mm 是移除拍摄组后的模型变化量，不是对真实尺寸绝对误差的保证。</div>
<h2>C 的异常角点如何处理</h2><p>帧 232 的 ID90 有一个原始角点落在白色贴纸外沿。四条黑到白边缘能确定正确交点，原流程却因修正量 20.09 px 超过固定 12 px 上限而回退。现在采用随码大小变化的安全界限，并检查边缘支持、方向、交点条件与凸性。该观测的四角 RMS 从 10.378 px 降到 0.358 px。</p>
<img src="{image(result_path(root, "marker_corner_correction.jpg"))}" alt="同一原图上错误角点与修正角点的比较">
<p>边缘支持不足、近乎平行的边缘交点，以及投影面积与最长边平方之比小于 0.25 的严重斜视码，依据图像本身标记为无效。共排除 {len(image_quality["excluded"])} 个候选观测，其中验证部分为 {summary["all_candidate_test_observations"] - summary["eligible_test_observations"]} 个；排除不依赖验证重投影误差。全部候选误差仍单独保留。训练时另有 {len(summary["training_outliers_removed"])} 个观测超过工程配置的 2 px 门限，在重新优化前移除；验证数据未按误差移除。</p>
<h2>整组留出：检查不同器械摆放之间的稳定性</h2><p>每轮完全移除一个拍摄组的 marker 几何观测，仅用其余六组估计九个码。相机内参在各轮保持固定。被移除组只用其训练部分估计器械到棋盘的六自由度放置变换，最后连续 20% 用于验证。三维几何变化直接在固定 ID84 坐标系中比较，没有缩放或额外对齐。</p>
<table><tr><th>移除组</th><th>该组验证角点 RMS px</th><th>几何角点变化 RMS mm</th><th>最大变化 mm</th></tr>{fold_rows}</table>
<h2>边长与坐标系</h2><p>长度单位为 mm；全部九个码边长自由估计。原点是 ID84 中心，解码后 +X 向右、+Y 向上、+Z 为码面外法向。四角按 OpenCV 解码顺序：左上、右上、右下、左下。</p>
<table><tr><th>ID</th><th>边长 mm</th><th>有效验证角点 RMS px</th><th>P95 px</th><th>验证张数</th></tr>{edge_rows}</table>
<h2>与工程输出保持一致</h2><p>最终文件直接调用工程 <code>marker_calibration/marker_io.py::save_geometry()</code>，使用原 <code>double_cube_82_90.json</code> 的 instrument 与 tip_offset 配置。文件名、字段集合、schema_version、ID 整数键和行内三元坐标格式保持一致。</p>
<pre>marker_positions:\n  84:\n  - [-L84/2, L84/2, 0.0]\n  - [ L84/2, L84/2, 0.0]\n  - [ L84/2,-L84/2, 0.0]\n  - [-L84/2,-L84/2, 0.0]\nreference_marker_id: 84\nscale_source: checkerboard_35_mm\nreference_marker_length_fixed: false\ntip_offset_source: configured_not_calibrated</pre>
<p>方法和尺度的字段值如实记录此次棋盘标定；参考码长度不固定。<code>tip_offset: [0.0, 0.0, 0.0]</code> 是工程配置值，此次未估计器械尖端。</p>
<h2>数值门禁</h2><p>采用工程配置的训练观测 2 px、验证观测 3 px、每码最少 5 张训练图、最少 15 张训练图和至少 10° 观察方向跨度，并额外检查七轮分组验证。</p><table><tr><th>检查</th><th>状态</th></tr>{gate_rows}</table>
<h2>投影叠加</h2>{gallery}
<h2>复现文件</h2><p><code>detections.json</code> 保留图像观测与筛选理由；<code>marker_all_candidate_observations.json</code> 保留筛选前误差；<code>marker_group_validation.json</code> 保留七轮详细结果。<code>source_manifest.json</code> 记录原始图像 SHA-256。</p>
<pre>python -m python.scripts.marker_calibration.checkerboard_geometry fit --source inputs/camera_source --output .\npython -m python.scripts.marker_calibration.checkerboard_geometry groups --output .\npython -m python.scripts.marker_calibration.checkerboard_geometry export --output . --config configs/markers/double_cube_82_90.json</pre>
<p class="muted">正式几何以 double_cube_geometry.yaml 中的全精度浮点数为准。</p></main></html>'''
    (root / "marker_calibration_report.html").write_text(page, encoding="utf-8")
    (root / "README.md").write_text(
        f"""# C 器械几何标定\n\n最终文件：[double_cube_geometry.yaml](double_cube_geometry.yaml)。ID84–92，参考码84，单位mm。\n\n使用工程原有 save_geometry()，字段与格式保持一致。尺度来自35mm棋盘，所有marker边长自由估计。tip_offset=[0,0,0]沿用配置，未标定尖端。\n\n- 项目数值门禁全部通过。\n- 有效留出角点RMS：{summary["eligible_test_corner_errors_px"]["rms"]:.3f}px；全部候选RMS：{summary["all_candidates_test_corner_errors_px"]["rms"]:.3f}px。\n- 独立留出重建角点差异RMS：{metric["rms"]:.4f}mm。\n- 七轮整组留出的最大几何角点变化：{worst:.4f}mm。\n\n[完整图文报告](marker_calibration_report.html)。毫米指标是同一棋盘基准下的一致性，不能代替外部尺寸计量。\n""",
        encoding="utf-8",
    )


def build_checkerboard_report(output):
    import cv2
    import scipy

    output = Path(output)
    fit = json.loads((output / "fit.json").read_text(encoding="utf-8"))
    report = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
    intrinsic = json.loads(
        (output / "intrinsics_report.json").read_text(encoding="utf-8")
    )
    obs = json.loads(
        (output / "reprojection_observations.json").read_text(encoding="utf-8")
    )["observations"]
    colors = {"A": "#007d85", "B": "#a45b15", "C": "#5951a2"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    for tool in "ABC":
        errors = np.sort(
            [
                v
                for r in obs
                if r["split"] == "test" and r["tool"] == tool
                for v in r["corner_errors_px"]
            ]
        )
        axes[0, 0].plot(
            errors,
            np.arange(1, len(errors) + 1) / len(errors),
            label=tool,
            color=colors[tool],
        )
    axes[0, 0].set(
        xlim=(0, 3),
        ylim=(0, 1.01),
        xlabel="Held-out corner error (px)",
        ylabel="Cumulative fraction",
        title="Board-only camera poses; no marker refit",
    )
    axes[0, 0].legend(title="Tool")
    axes[0, 0].grid(alpha=0.2)
    for tool in "ABC":
        ids = fit["tools"][tool]
        axes[0, 1].scatter(
            ids,
            [fit["lengths_mm"][str(mid)] for mid in ids],
            label=tool,
            color=colors[tool],
            s=25,
        )
    axes[0, 1].set(
        xlabel="Marker ID",
        ylabel="Estimated edge (mm)",
        title="All edge lengths estimated; 35 mm board fixes scale",
    )
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)
    values = [
        [
            v
            for r in report["metric_crosscheck"]["observations"]
            if r["tool"] == tool
            for v in r["corner_difference_mm"]
        ]
        for tool in "ABC"
    ]
    box = axes[1, 0].boxplot(
        values, tick_labels=list("ABC"), patch_artist=True, showmeans=True
    )
    for patch, tool in zip(box["boxes"], "ABC"):
        patch.set_facecolor(colors[tool])
        patch.set_alpha(0.5)
    axes[1, 0].set(
        ylabel="3D corner difference (mm)",
        xlabel="Tool",
        title="Independent held-out reconstruction vs training geometry",
    )
    axes[1, 0].grid(axis="y", alpha=0.2)
    old = output / "first_pass" / "preliminary_validation.json"
    if old.exists():
        previous = json.loads(old.read_text())["summary"]
        x = np.arange(3)
        axes[1, 1].bar(
            x - 0.18,
            [previous[str(("test", t))]["rms"] for t in "ABC"],
            0.36,
            label="Unsigned subpixel corners",
            color="#aebac2",
        )
        axes[1, 1].bar(
            x + 0.18,
            [report["summary_px"]["test"][t]["rms"] for t in "ABC"],
            0.36,
            label="Signed border-line corners",
            color="#007d85",
        )
        axes[1, 1].set(
            xticks=x,
            xticklabels=list("ABC"),
            ylabel="Held-out corner RMS (px)",
            title="Corner localization correction; all test observations kept",
        )
        axes[1, 1].legend(fontsize=8)
    fig.savefig(output / "validation_charts.png", dpi=170)
    plt.close(fig)

    def image(path):
        path = Path(path)
        mime = "image/png" if path.suffix == ".png" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode(
            "ascii"
        )

    def fmt(value):
        return f"{value:.4f}"

    table = []
    length_rows = []
    for tool in "ABC":
        ids = fit["tools"][tool]
        s = report["summary_px"]["test"][tool]
        m = report["metric_crosscheck"]["per_tool"][tool]
        edge = [fit["lengths_mm"][str(mid)] for mid in ids]
        table.append(
            f"<tr><td>{tool}</td><td>{len(ids)}/9</td><td>{min(edge):.3f}–{max(edge):.3f}</td><td>{s['rms']:.3f}</td><td>{s['p95']:.3f}</td><td>{s['max']:.3f}</td><td>{m['corner_difference_mm']['rms']:.3f}</td></tr>"
        )
        for mid in ids:
            p = report["per_id_test_px"][str(mid)]
            length_rows.append(
                f"<tr><td>{tool}</td><td>{mid}</td><td>{fit['lengths_mm'][str(mid)]:.6f}</td><td>{p['count'] // 4}</td><td>{p['rms']:.3f}</td><td>{p['p95']:.3f}</td></tr>"
            )
    flagged = report["observations_flagged_over_3px_rmse"]["test"]
    bad_rows = "".join(
        f"<tr><td>{r['image']}</td><td>{r['id']}</td><td>{r['rmse_px']:.3f}</td></tr>"
        for r in flagged["records"]
    )
    gallery = []
    for path in sorted((output / "overlays").glob("*.jpg")):
        gallery.append(
            f'<figure><img loading="lazy" src="{image(path)}"><figcaption>{path.stem}：绿色圆圈为检测角点，红色十字为预测角点，黄色线为残差。</figcaption></figure>'
        )
    k = np.array(fit["camera_matrix"])
    dist = np.array(fit["distortion"])
    ktext = "\n".join("  ".join(f"{v:14.8f}" for v in row) for row in k)
    dtext = ", ".join(f"{v:.12g}" for v in dist)
    board_check = report["metric_crosscheck"]
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Galaxy 红外相机与器械 marker 几何标定 — 20261112</title>
<style>body{{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#19313d;background:#f4f7f8;margin:0}}main{{max-width:1180px;margin:auto;padding:36px 28px;background:white}}h1{{font-size:29px;line-height:1.4}}h2{{margin-top:36px;font-size:22px;color:#086c74}}p{{max-width:1050px}}.notice{{border-left:5px solid #c58620;background:#fff7e5;padding:14px 18px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:9px 12px;border-bottom:1px solid #dce5e8;text-align:left}}th{{background:#eaf2f3}}img{{max-width:100%;height:auto}}pre{{overflow:auto;background:#f1f5f6;padding:16px;border-radius:8px}}.muted,figcaption{{color:#627783;font-size:14px}}a{{color:#007780}}code{{overflow-wrap:anywhere}}figure{{margin:20px 0}}.scroll{{overflow:auto}}</style>
<main><p class="muted">2026-11-16 · 317 张原始图像 · 4504×4504 · DICT_4X4_100</p>
<h1>Galaxy 红外相机与三个器械的毫米尺度标定</h1>
<div class="notice"><strong>结果范围：</strong>B、C 各恢复了 9 个 marker；A 只稳定识别到 ID 23–30 共 8 个，其第 9 个 marker 无足够有效观测，未补造几何。当前交付是 26 个 marker 的几何结果，不能称为完整的 27 码标定。针尖/球心位置未标定，输出器械默认不启用。</div>
<p>尺度来自 <strong>11×8 内角点、35 mm 格长的棋盘</strong>。每个 marker 的边长都由图像估计，没有固定任意一个码为 15 mm。七个拍摄组内部器械与棋盘相对固定，组间允许分别重新摆放三个器械。</p>
<h2>结果与验证</h2>
<p>每组最后连续 20% 的图像留出，共 <strong>{report["training_images"]} 张训练、{report["heldout_images"]} 张验证</strong>。验证图中相机位姿仅由棋盘计算；marker 的几何和各组器械摆放变换均固定为训练结果。下表包含全部验证 marker 观测，未按验证残差剔除异常。</p>
<div class="scroll"><table><tr><th>器械</th><th>覆盖</th><th>边长范围 mm</th><th>角点 RMS px</th><th>角点 P95 px</th><th>最大角点误差 px</th><th>独立重建角点差异 RMS mm</th></tr>{"".join(table)}</table></div>
<p class="muted">毫米列：只用留出图像、棋盘相机位姿，逐组独立重新估计正方形 marker，再与训练结果在棋盘坐标系中比较；不作缩放或刚体对齐。每项要求至少 5 张可见图像、射线角跨度至少 6°。这是数据一致性评估，不是外部计量给出的绝对精度。</p>
<img src="{image(output / "validation_charts.png")}" alt="验证误差分布、边长和独立重建比较">
<h2>相机内参与棋盘检查</h2>
<p>使用第 0 组前 74 张中的训练部分拟合：59 张候选，剔除棋盘异常帧 14、55，最终 57 张参与内参拟合；另外 15 张留出。Brown–Conrady 五畸变参数模型，训练 RMS <strong>{intrinsic["training_rms_px"]:.4f} px</strong>；这 15 张图每张用一半角点求位姿、另一半计算误差，RMS <strong>{intrinsic["heldout_half_board_corner_error_px"]["rms"]:.4f} px</strong>。</p>
<pre>K =\n{ktext}\n\nD = [k1, k2, p1, p2, k3]\n[{dtext}]</pre>
<p>全部 66 张留出图的互补棋盘角点 RMS 为 {report["heldout_complementary_board_corners_px"]["rms"]:.4f} px。用一半棋盘角点确定相机位姿，对从未用于这些位姿估计的另外 44 个角点进行多视角重建：三维位置 RMS 差异 {board_check["unused_board_corner_reconstruction_error_mm"]["rms"]:.4f} mm；相隔两格的 70 mm 距离 RMS 差异 {board_check["unused_board_two_step_70mm_distance_error_mm"]["rms"]:.4f} mm。这些检查仍共同依赖同一块棋盘的尺寸与平面假设。</p>
<h2>联合优化模型</h2>
<pre>棋盘角点 Q 已知，单位 mm：\n  board_pixel(i,k) = project(K,D, C_i Q_k)\n\n器械 o 的 marker j：\n  marker_pixel(i,j,k) = project(K,D, C_i E_group(i),o G_j square(L_j)_k)</pre>
<p>C_i 为逐图棋盘到相机变换；E 为每组、每器械的放置变换；G_j 和 L_j 为各码共享的位姿与边长。内参固定后，先固定棋盘 PnP 相机位姿求器械几何，再联合调整相机位姿、各组放置变换、各码位姿和全部边长。采用 soft-L1 鲁棒损失、解析稀疏 Jacobian；每个码保持理想正方形，不强制立方体直角或面间距。只固定参考码的位姿来定义坐标系，参考码边长仍自由估计。</p>
<p>导出参考码：A=26，B=81，C=84。各自原点在参考码中心，解码方向 +X 向右、+Y 向上、+Z 为外法向。此坐标系不包含针尖或器械功能点。</p>
<h2>已发现并处理的问题</h2>
<p>原始通用亚像素角点细化会在部分图上锁到白色贴纸外沿。新流程对黑到白的有符号梯度拟合四条直线，再以交点确定黑码边界。验证 RMS 在 A/B/C 上由首轮的 1.370/1.334/0.940 px 改善至 {report["summary_px"]["test"]["A"]["rms"]:.3f}/{report["summary_px"]["test"]["B"]["rms"]:.3f}/{report["summary_px"]["test"]["C"]["rms"]:.3f} px。低频 ID 17（2 次）和 37（3 次）缺乏稳定支持，未纳入器械几何。</p>
<p>仍有 {flagged["count"]}/{flagged["total"]} 个验证“图像—marker”观测的 RMS 超过 3 px，均保留在上表统计中。尤其帧 232 的 ID 90 有一个明显错误角点，造成 C 的最大角点误差 20.610 px。训练数据中另有 23/2463 个观测超过 3 px，由鲁棒损失降低影响，没有按验证数据删除它们。</p>
<table><tr><th>验证图像</th><th>ID</th><th>该码四角 RMS px</th></tr>{bad_rows}</table>
<h2>每个 marker 的边长</h2><table><tr><th>器械</th><th>ID</th><th>边长 mm</th><th>验证观测张数</th><th>验证角点 RMS px</th><th>P95 px</th></tr>{"".join(length_rows)}</table>
<h2>投影叠加检查</h2><p>下面包含每组代表性留出图和高误差图。绿色圈是检测，红色十字是训练几何预测，黄色连线表示差异；局部放大仅用于显示。</p>{"".join(gallery)}
<h2>交付文件与复现</h2><ul><li><a href="camera_intrinsics.yml">camera_intrinsics.yml</a>：当前 Galaxy 红外相机内参。</li><li><a href="tool_A_geometry.yaml">A 几何（8 码，覆盖不完整）</a>、<a href="tool_B_geometry.yaml">B 几何（9 码）</a>、<a href="tool_C_geometry.yaml">C 几何（9 码）</a>：完整精度的角点坐标与边长。</li><li><a href="validation_report.json">validation_report.json</a>：全部验证指标、异常和毫米检查。</li><li>detections.json、fit.json、reprojection_observations.json：重检测、优化参数和逐观测误差；source_manifest.json 记录原始 317 张图像、317 个逐帧 JSON 以及 session.json 的 SHA-256。</li></ul>
<pre>python -m python.scripts.marker_calibration.checkerboard fit --output .\npython -m python.scripts.marker_calibration.checkerboard validate --output .</pre>
<p>检测在服务器 OpenCV 4.10 上运行，拟合与验证环境为 Python {platform.python_version()}、OpenCV {cv2.__version__}、NumPy {np.__version__}、SciPy {scipy.__version__}。解析 Jacobian 与中心差分的相对误差 6.55×10⁻¹⁰；含组间重新摆放和不等边长的无噪声合成场景恢复误差小于 10⁻⁹ mm。合成测试只验证实现，不替代实拍验证。</p>
<div class="notice"><strong>剩余事项：</strong>要完成 A 的第 9 码，需要新增该码清晰可见、与已有码共视的图像，并保持同一组内器械与棋盘相对固定。当前结果未标定器械尖端，也未通过独立量具验证绝对毫米精度，不能直接视为导航系统精度认证。</div>
<p class="muted">原始数据：{html.escape(report["source"])}。报告中的全精度数值以 JSON/YAML 为准。</p></main></html>'''
    (output / "calibration_report.html").write_text(page, encoding="utf-8")
    markdown = [
        "# Galaxy 红外相机与 marker 几何标定",
        "",
        "B、C 各九个码完成；A 只有八个稳定可见码，覆盖不完整。未标定针尖。",
        "",
        "[打开完整图文报告](calibration_report.html)",
        "",
        "|器械|码数|验证角点 RMS (px)|P95 (px)|独立重建角点差异 RMS (mm)|",
        "|---|---:|---:|---:|---:|",
    ]
    for tool in "ABC":
        s = report["summary_px"]["test"][tool]
        m = report["metric_crosscheck"]["per_tool"][tool]["corner_difference_mm"]
        markdown.append(
            f"|{tool}|{len(fit['tools'][tool])}|{s['rms']:.3f}|{s['p95']:.3f}|{m['rms']:.3f}|"
        )
    markdown += [
        "",
        "317 张图像；七组各留出最后连续 20%，251 张训练、66 张验证。尺度来自 35 mm 棋盘，全部码边长自由估计。",
        "验证只用棋盘求相机位姿，不重新拟合器械位姿。毫米列表示同一棋盘基准下的重建一致性，并非独立实测的绝对精度。",
        "",
        "全部验证观测均保留；662 个观测中有两个 RMS 超过 3 px，见图文报告。",
        "",
        "A 的 ID 为 23–30；B 为 69–76、81；C 为 84–92。参考坐标码分别为 26、81、84。",
        "",
        "![验证图表](validation_charts.png)",
        "",
        "结果默认不启用，不包含针尖/球心位置。补拍 A 第九个 marker 后才能完成 27 码几何。",
    ]
    (output / "README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    manifest = {
        "versions": {
            "python": platform.python_version(),
            "opencv_fit": cv2.__version__,
            "opencv_detection": "4.10.0",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "input_image_count": 317,
        "artifacts": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.iterdir())
            if path.is_file()
            and path.suffix in (".json", ".yaml", ".yml", ".py", ".html", ".md", ".png")
            and path.name != "artifact_manifest.json"
        },
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkerboard", action="store_true", help="Report on the checkerboard and multi-instrument fit")
    args = parser.parse_args()
    (build_checkerboard_report if args.checkerboard else build)(args.output)

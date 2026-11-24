"""Persist calibration residuals without claiming measured metric repeatability."""
from __future__ import annotations

import json
import csv
import io
import math
from pathlib import Path
import uuid

import numpy as np

from ..session_storage import commit_staged_files


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _image_diagnostics(sensor, options):
    training = {entry["image"]: entry for entry in sensor.get("training_views", [])}
    validation = {entry["path"]: entry for entry in sensor.get("validation_views", [])}
    rejected = {entry["image"]: entry for entry in sensor.get("rejected_training_views", [])}
    excluded = {entry["path"]: entry for entry in sensor.get("excluded_validation_views", [])}
    rows = []
    for detection in sensor.get("detections", []):
        image = detection["image"]
        issues = []
        stage = "训练" if detection.get("split") == "training" else "留出验证"
        error = None
        limit = options.get("max_view_rmse_px", options.get("max_rms_px", 1.0))
        if detection["status"] != "detected":
            issues.append(detection["status"] + (": " + detection["detail"] if detection.get("detail") else ""))
        if image in rejected:
            stage = "剔除后验证"
            issues.append(f"训练阶段剔除：{rejected[image]['rmse_px_before_removal']:.6f} px > "
                          f"{rejected[image]['threshold_px']:.6f} px")
            entry = excluded.get(image, {})
            error = entry.get("view_rms_px")
            limit = options.get("max_validation_rmse_px", 2.0)
            if not entry:
                issues.append("未执行剔除后验证")
            elif entry.get("status") != "ok":
                issues.append("剔除后验证：" + entry.get("status", "unknown"))
        elif image in training:
            error = training[image]["rmse_px"]
        elif image in validation:
            error = validation[image].get("view_rms_px")
            limit = options.get("max_validation_rmse_px", 2.0)
            if validation[image].get("status") != "ok":
                issues.append("留出验证：" + validation[image].get("status", "unknown"))
        if error is not None and error > limit:
            issues.append(f"{stage}误差超限：{error:.6f} px > {limit:.6f} px")
        rows.append({"image": image, "stage": stage, "detector": detection.get("detector"),
                     "rmse_px": error, "threshold_px": limit,
                     "issues": issues, "status": "issue" if issues else "ok"})
    return rows


def save_quality_report(output_dir: Path, result: dict, *, replace_existing: bool = False) -> list[str]:
    """Write JSON details and a readable summary, including failed-fit diagnostics."""
    paths = [output_dir / "calibration_quality.json", output_dir / "calibration_quality.md",
             output_dir / "excluded_images.csv"]
    result["report_files"] = [str(path) for path in paths]
    for sensor in result["sensors"].values():
        sensor["image_diagnostics"] = _image_diagnostics(sensor, result.get("options", {}))
    clean = _json_value(result)
    items = list(clean["sensors"].items()) + [
        (f"{pair['left_sensor']} -> {pair['right_sensor']}", pair) for pair in clean["pairs"]]
    records = []
    for label, sensor in clean["sensors"].items():
        for entry in sensor.get("rejected_training_views", []):
            records.append({"stage": "intrinsics", "sensor_or_pair": label, "image": entry["image"],
                            "paired_image": "", "rmse_px": entry["rmse_px_before_removal"],
                            "threshold_px": entry["threshold_px"], "reason": entry["reason"]})
    for pair in clean["pairs"]:
        for entry in pair.get("rejected_pairs", []):
            records.append({"stage": "extrinsics", "sensor_or_pair": f"{pair['left_sensor']} -> {pair['right_sensor']}",
                            "image": entry["left_image"], "paired_image": entry["right_image"],
                            "rmse_px": "", "threshold_px": "", "reason": entry["reason"]})
    lines = render_markdown(clean, items)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=["stage", "sensor_or_pair", "image", "paired_image",
                                                  "rmse_px", "threshold_px", "reason"])
    writer.writeheader()
    writer.writerows(records)
    payloads = [json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                "\n".join(lines) + "\n", csv_buffer.getvalue()]
    staged = []
    try:
        for path, payload in zip(paths, payloads):
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            staged.append((temporary, path))
            temporary.write_text(payload, encoding="utf-8")
        commit_staged_files(staged, replace_existing=replace_existing)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return result["report_files"]


def render_markdown(result, items):
    def number(value):
        return f"{value:.6f}" if isinstance(value, (int, float)) else "未测得"

    lines = ["# 标定质量报告", "", f"数据目录：`{result['session']}`", "",
             f"像素质量检查：{'PASS' if result['passed'] else 'FAIL'}", "",
             "## 最终标定结果", "",
             "下表使用剔除异常图后重新拟合的参数。误差单位为 px；"
             "留出验证图不参与拟合，验证均值是逐图 RMSE 的平均值。", "",
             "| 对象 | 拟合图/配对数 | 留出验证数 | 标定 RMS | 验证平均 RMSE | 验证最大 RMSE |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, item in items:
        label = label.replace(" -> ", " → ")
        validation = item.get("validation", {})
        count = item.get("training_image_count", item.get("training_pair_count_used", item.get("training_pair_count")))
        lines.append(f"| {label} | {count} | {validation.get('count', 0)} | "
                     f"{number(item.get('training_rms_px', item.get('stereo_rms_px')))} | "
                     f"{number(validation.get('mean_rmse_px'))} | {number(validation.get('max_rmse_px'))} |")
    for pair in result["pairs"]:
        if pair.get("baseline_mm") is not None:
            lines.extend(["", f"{pair['left_sensor']} → {pair['right_sensor']}：基线 {pair['baseline_mm']:.6f} mm。"])
    lines.extend(["", "双目标定 RMS 合并两台相机各自原图像素误差；外参验证 RMSE 按第二相机像素计算。"])

    anomalies = []
    for label, sensor in result["sensors"].items():
        for row in sensor.get("image_diagnostics", []):
            if row["status"] != "issue":
                continue
            path = (Path(result["session"]) / row["image"]).as_posix()
            stage = "已剔除训练图，使用最终参数复核" if row["stage"] == "剔除后验证" else row["stage"]
            anomalies.append((f"[{row['image']}](<{path}>)", stage, "；".join(row["issues"])))
        for warning in sensor.get("image_warnings", []):
            path = (Path(result["session"]) / warning["image"]).as_posix()
            stage = warning["stage"] + (f"（第 {warning['iteration']} 轮）" if "iteration" in warning else "")
            anomalies.append((f"[{warning['image']}](<{path}>)", stage, warning["message"]))
    for pair in result["pairs"]:
        for warning in pair.get("image_warnings", []):
            anomalies.append((f"{warning['left_image']} / {warning['right_image']}",
                              warning["stage"], warning["message"]))
        for row in pair.get("rejected_pairs", []):
            reason = row["reason"]
            if reason.startswith("intrinsic_training_outlier:"):
                reason = "因 " + reason.split(":", 1)[1].strip() + " 的训练图异常，配对退出外参拟合"
            anomalies.append((f"{row['left_image']} / {row['right_image']}", "已剔除外参配对", reason))
        for field, stage in (("validation_views", "留出外参验证"),
                             ("excluded_validation_views", "剔除配对复核")):
            for row in pair.get(field, []):
                issues = []
                if row.get("status", "ok") != "ok":
                    issues.append(row.get("reason", row["status"]))
                if row.get("ordering_ambiguous"):
                    issues.append("角点编号有歧义")
                error = row.get("right_reprojection_rmse_px")
                limit = result.get("options", {}).get("max_validation_rmse_px", 2.0)
                if error is not None and error > limit:
                    issues.append(f"RMSE {error:.6f} px 超过 {limit:.6f} px 阈值")
                if issues:
                    anomalies.append((f"{row['left_image']} / {row['right_image']}", stage, "；".join(issues)))
    if anomalies:
        lines.extend(["", "## 图像警告与失败原因", "",
                      "保留标定各阶段出现的逐图警告，包括后来通过检查的图片。"
                      "剔除图的复核误差不计入上表的标定 RMS 或留出验证。", "",
                      "| 图像/配对 | 处理阶段 | 异常与原因 |", "| --- | --- | --- |"])
        for image, stage, issue in dict.fromkeys(anomalies):
            lines.append(f"| {image} | {stage} | {issue.replace('|', '/')} |")

    lines.extend(["", "重复性精度尚未测量；像素误差不等同于毫米级精度。逐图明细保存在 JSON，剔除清单保存在 CSV。", ""])
    return lines

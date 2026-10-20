"""Persist calibration residuals without claiming measured metric repeatability."""
from __future__ import annotations

import json
import csv
import io
import math
from pathlib import Path
import uuid

import numpy as np

from ..sensor_intrinsics.core import commit_staged_files
from .report_markdown import render_markdown


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

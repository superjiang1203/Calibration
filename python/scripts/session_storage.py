"""Calibration session paths and atomic artifact storage."""
from pathlib import Path
import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Sequence

import cv2


PROJECT_ROOT = Path(
    os.environ.get("CALIBRATION_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()


def sensor_image_directories(sensors):
    """Name image folders by vendor/channel, independently of display labels."""
    bases = {}
    for sensor in sensors:
        label = sensor['label']
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', label):
            raise ValueError(f'Invalid sensor label: {label!r}')
        vendor, stream = sensor['sensor_type'], sensor['stream']
        if vendor not in ('orbbec', 'galaxy', 'opencv', 'vcamera') or stream not in ('ir_left', 'ir_right', 'color', 'gray'):
            raise ValueError(f'Unsupported sensor image directory: {vendor}/{stream}')
        channel = {'ir_left': 'IR_left', 'ir_right': 'IR_right'}.get(stream, stream)
        bases[label] = f'{vendor}_{channel}'
    if len(bases) != len(sensors):
        raise ValueError('Sensor labels must be unique')
    counts = {base: list(bases.values()).count(base) for base in bases.values()}
    directories = {label: base if counts[base] == 1 else f'{base}_{label}'
                   for label, base in bases.items()}
    if len({name.casefold() for name in directories.values()}) != len(directories):
        raise ValueError('Sensor image directories must be unique (case-insensitive)')
    return directories


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.stem}.{uuid.uuid4().hex}{path.suffix}')
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise OSError(f'Cannot encode image: {path}')
    temporary = path.with_name(f'.{path.stem}.{uuid.uuid4().hex}.png')
    try:
        encoded.tofile(str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def timestamp_directory(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        destination = parent / datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        try:
            destination.mkdir()
            return destination
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate a timestamp directory under {parent}")


def session_path(value: Path) -> Path:
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def commit_staged_files(
    pairs: Sequence[tuple[Path, Path]], *, replace_existing: bool
) -> None:
    """Commit a related set of staged files with backup-based rollback."""

    normalized = [(Path(source), Path(target)) for source, target in pairs]
    target_keys = [str(target.resolve()).casefold() for _, target in normalized]
    if len(target_keys) != len(set(target_keys)):
        raise ValueError("staged file targets must be unique")
    for source, target in normalized:
        if not source.is_file():
            raise FileNotFoundError(f"staged file is missing: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)

    token = uuid.uuid4().hex
    backups: dict[Path, Path] = {}
    committed: set[Path] = set()
    try:
        # Move every old target out of the way before publishing any new file.
        for _, target in normalized:
            if not target.exists():
                continue
            if not replace_existing:
                raise FileExistsError(f"output already exists: {target}")
            backup = target.with_name(f".{target.name}.{token}.backup")
            os.replace(target, backup)
            backups[target] = backup

        for source, target in normalized:
            os.replace(source, target)
            committed.add(target)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for target in committed:
            if target not in backups:
                try:
                    target.unlink(missing_ok=True)
                except OSError as rollback_exc:
                    rollback_errors.append(f"remove {target}: {rollback_exc}")
        for target, backup in backups.items():
            if not backup.exists():
                continue
            try:
                os.replace(backup, target)
            except OSError as rollback_exc:
                rollback_errors.append(f"restore {target}: {rollback_exc}")
        if rollback_errors:
            raise RuntimeError(
                "output commit failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        raise
    else:
        for backup in backups.values():
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                pass

"""Atomic JSON and lossless image storage, including non-ASCII Windows paths."""
from pathlib import Path
import json
import os
import re
import uuid
import cv2


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

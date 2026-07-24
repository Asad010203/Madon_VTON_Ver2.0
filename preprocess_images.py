"""Normalize person + garment images to Leffa's expected input format.

Leffa VITON-HD is trained on 768x1024 portrait (aspect 3:4). Feeding images with
different aspect ratios causes:
    - Human-parsing artifacts (mask bleeds beyond body silhouette)
    - Phantom garments in output (padding regions filled with garment)
    - Garment stretching/warping

This module provides a single ``prepare_pair()`` that:
    1. Loads person + garment as RGB
    2. Resizes each to fit within 768x1024, preserving aspect ratio
    3. Pads with white to exactly 768x1024 (matches Leffa training convention)
    4. Returns the two normalized images as bytes ready to send to Modal
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

from PIL import Image


TARGET_W, TARGET_H = 768, 1024   # Leffa VITON-HD canonical size


def fit_and_pad(img: Image.Image, target_w: int = TARGET_W, target_h: int = TARGET_H,
                background: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Resize ``img`` to fit inside ``target_w x target_h`` keeping aspect, pad white."""
    img = img.convert("RGB")
    src_w, src_h = img.size

    # Scale so that whichever dimension is limiting hits the target first.
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), background)
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    canvas.paste(resized, (off_x, off_y))
    return canvas


def _to_jpeg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def prepare_pair(person_path: str | Path, garment_path: str | Path,
                 save_debug_to: Path | None = None) -> Tuple[bytes, bytes]:
    """Return (person_bytes, garment_bytes) both normalized to 768x1024."""
    person  = fit_and_pad(Image.open(person_path))
    garment = fit_and_pad(Image.open(garment_path))

    if save_debug_to is not None:
        save_debug_to.mkdir(parents=True, exist_ok=True)
        person.save(save_debug_to / "_input_person.jpg",  "JPEG", quality=95)
        garment.save(save_debug_to / "_input_garment.jpg", "JPEG", quality=95)

    return _to_jpeg_bytes(person), _to_jpeg_bytes(garment)

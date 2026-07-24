"""Merge all vton_gallery_run_*.docx fallback files into the main vton_gallery.docx.

Runs when Word was open during some test runs, causing the appends to fall
back to per-run files. This script:

  1. Finds every outputs/vton_gallery_run_*.docx (sorted by timestamp)
  2. Appends each to outputs/vton_gallery.docx in chronological order
  3. Deletes the fallback files once merged

Close the gallery in Word/WPS before running this.

Usage:
    .\.venv\Scripts\python.exe merge_gallery.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docxcompose.composer import Composer


OUT_DIR = Path(__file__).parent / "outputs"
GALLERY = OUT_DIR / "vton_gallery.docx"
FALLBACK_GLOB = "vton_gallery_run_*.docx"
FALLBACK_RE = re.compile(r"^vton_gallery_run_(\d{8})_(\d{6})\.docx$")


def main() -> None:
    if not GALLERY.exists():
        print(f"No main gallery found at {GALLERY}"); sys.exit(1)

    fallbacks = sorted(
        (p for p in OUT_DIR.glob(FALLBACK_GLOB) if FALLBACK_RE.match(p.name)),
        key=lambda p: p.name,
    )
    if not fallbacks:
        print("No fallback files to merge — gallery is already whole."); return

    print(f"Merging {len(fallbacks)} fallback file(s) into {GALLERY.name}:")
    for p in fallbacks:
        print(f"  + {p.name}")

    # Try to open the master exclusively — will fail with PermissionError if
    # Word/WPS holds it open.
    try:
        with GALLERY.open("rb+"):
            pass
    except PermissionError:
        print(f"\n⚠ {GALLERY.name} is open in Word/WPS. Close it and rerun."); sys.exit(2)

    master   = Document(str(GALLERY))
    composer = Composer(master)
    for p in fallbacks:
        composer.append(Document(str(p)))

    composer.save(str(GALLERY))
    print(f"\n✓ Merged {len(fallbacks)} file(s) into {GALLERY.name}")

    # Only delete after a successful save.
    for p in fallbacks:
        p.unlink()
        print(f"  deleted {p.name}")

    print("\nDone. Reopen vton_gallery.docx to see all runs.")


if __name__ == "__main__":
    main()

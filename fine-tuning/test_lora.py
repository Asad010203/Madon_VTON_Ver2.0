"""
Interactive IDM-VTON tester with trained LoRA — pure CLI, no UI.

Prompts you for person path, garment path, garment type, then runs inference
on Modal (L4 GPU) with the trained shorts LoRA applied. Saves output image
locally and loops for next test.

Usage:
    python test_lora.py
    python test_lora.py --checkpoint idmvton_lora_shorts_20260724_180000_e20.pt
    python test_lora.py --no-lora     # baseline (no LoRA), for A/B comparison
"""

from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import modal
from PIL import Image, ImageDraw, ImageFont


STEPS          = 30
GUIDANCE_SCALE = 2.0
GARMENT_TYPES  = ["upper_body", "lower_body", "dresses"]
DEFAULT_TYPE   = "lower_body"      # dataset is shorts
IMG_TYPES      = [("Images", "*.jpg *.jpeg *.png *.webp *.bmp"),
                  ("All files", "*.*")]


def make_composite(person_path: Path, garment_path: Path, result_path: Path,
                   out_path: Path, lora_name: str) -> None:
    """Save a side-by-side composite: Person | Garment | Result, with labels."""
    try:
        p = Image.open(person_path).convert("RGB").resize((768, 1024))
        g = Image.open(garment_path).convert("RGB").resize((768, 1024))
        r = Image.open(result_path).convert("RGB").resize((768, 1024))

        header, gap = 70, 12
        comp_w = 768 * 3 + gap * 2
        comp_h = 1024 + header
        comp = Image.new("RGB", (comp_w, comp_h), color="white")
        comp.paste(p, (0, header))
        comp.paste(g, (768 + gap, header))
        comp.paste(r, (768 * 2 + gap * 2, header))

        draw = ImageDraw.Draw(comp)
        try:
            font_lg = ImageFont.truetype("arial.ttf", 22)
            font_sm = ImageFont.truetype("arial.ttf", 14)
        except (OSError, IOError):
            font_lg = ImageFont.load_default()
            font_sm = font_lg

        for label, x in zip(("Person", "Garment", "Result"),
                            (280, 768 + gap + 280, 768 * 2 + gap * 2 + 280)):
            draw.text((x, 12), label, fill="black", font=font_lg)
        draw.text((10, 44), f"LoRA: {lora_name}", fill="#555", font=font_sm)

        comp.save(str(out_path))
    except Exception as e:
        print(f"  ! composite failed: {e}")


_TK_ROOT: tk.Tk | None = None

def _get_tk_root() -> tk.Tk:
    """Return a single persistent hidden Tk root. Recreating Tk per dialog
    causes silent failures on Windows after the first close."""
    global _TK_ROOT
    if _TK_ROOT is None or not _TK_ROOT.winfo_exists():
        _TK_ROOT = tk.Tk()
        _TK_ROOT.withdraw()
        _TK_ROOT.attributes("-topmost", True)
    return _TK_ROOT


def pick_file(title: str, initial_dir: Path | None = None) -> Path | None:
    """Open a native file browser. Cancel = None (used to quit the loop)."""
    root = _get_tk_root()
    root.update()
    path = filedialog.askopenfilename(
        parent=root,
        title=title,
        filetypes=IMG_TYPES,
        initialdir=str(initial_dir) if initial_dir else None,
    )
    root.update()
    return Path(path) if path else None


def prompt_garment_type() -> str | None:
    print("  Garment type:")
    for i, t in enumerate(GARMENT_TYPES, 1):
        marker = "  (default)" if t == DEFAULT_TYPE else ""
        print(f"    {i}. {t}{marker}")
    raw = input("  choice [1-3, Enter=default]: ").strip()
    if not raw:
        return DEFAULT_TYPE
    if raw.lower() in ("q", "quit", "exit"):
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(GARMENT_TYPES):
        return GARMENT_TYPES[int(raw) - 1]
    print(f"  ! invalid: {raw}")
    return prompt_garment_type()


def pick_checkpoint(explicit: str | None) -> str:
    """Return LoRA checkpoint path on the Modal volume (or empty string for no-LoRA)."""
    if explicit == "__none__":
        return ""
    if explicit:
        return explicit if explicit.startswith("/checkpoints/") else f"/checkpoints/{explicit}"

    print("Listing available LoRA checkpoints on Modal...")
    lister = modal.Function.from_name("idm-vton-lora-infer", "list_checkpoints")
    ckpts  = lister.remote()
    if not ckpts:
        print("! No LoRA checkpoints found. Run training first "
              "(modal run modal_train_lora_real.py::train_lora).")
        sys.exit(1)
    print("\nAvailable checkpoints:")
    for i, c in enumerate(ckpts, 1):
        print(f"  {i}. {Path(c).name}")
    raw = input(f"pick [1-{len(ckpts)}, Enter=latest]: ").strip()
    if not raw:
        return ckpts[-1]
    if raw.isdigit() and 1 <= int(raw) <= len(ckpts):
        return ckpts[int(raw) - 1]
    print(f"! invalid: {raw}"); sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="LoRA checkpoint name (on /checkpoints/) or full path")
    ap.add_argument("--no-lora",    action="store_true", help="Run without LoRA (baseline)")
    ap.add_argument("--steps",      type=int, default=STEPS)
    args = ap.parse_args()

    checkpoint = pick_checkpoint("__none__" if args.no_lora else args.checkpoint)
    if checkpoint:
        print(f"\n✓ Using LoRA: {Path(checkpoint).name}")
    else:
        print("\n✓ Baseline mode — no LoRA applied")

    print("\nConnecting to Modal (idm-vton-lora-infer app)...")
    run_fn = modal.Function.from_name("idm-vton-lora-infer", "run_idm_lora")
    print("✓ Connected.\n")

    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    completed = 0
    session_start = time.time()
    last_dir: Path | None = None

    try:
        while True:
            print("=" * 70)
            print(f"  Test {completed + 1}   (session: {(time.time() - session_start)/60:.1f} min)")
            print("=" * 70)

            print("Opening file picker for PERSON image (Cancel to quit)...")
            person = pick_file("Select PERSON image (Cancel to quit)", last_dir)
            if not person:
                print("Cancelled — exiting.")
                break
            print(f"  person  = {person}")

            print("Opening file picker for GARMENT image (Cancel to quit)...")
            garment = pick_file("Select GARMENT image (Cancel to quit)", person.parent)
            if not garment:
                print("Cancelled — exiting.")
                break
            print(f"  garment = {garment}")

            gtype = prompt_garment_type()
            if not gtype:
                break
            last_dir = person.parent

            print(f"  type    = {gtype}")
            print(f"  steps   = {args.steps}")
            print(f"  LoRA    = {Path(checkpoint).name if checkpoint else 'NONE (baseline)'}")

            print("\nSending to Modal L4 GPU...")
            t0 = time.time()
            result = run_fn.remote(
                person.read_bytes(),
                garment.read_bytes(),
                garment_type=gtype,
                steps=args.steps,
                guidance_scale=GUIDANCE_SCALE,
                lora_checkpoint=checkpoint,
            )
            total_elapsed = time.time() - t0

            tag = "lora" if result.get("lora_applied") else "baseline"
            stamp    = time.strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"vton_{tag}_{stamp}.png"
            out_path.write_bytes(result["image"])

            comp_path = out_dir / f"vton_{tag}_{stamp}_composite.png"
            make_composite(person, garment, out_path, comp_path,
                           Path(checkpoint).name if checkpoint else "NONE (baseline)")

            t = result["timings"]
            print(f"\n=== TIMING ===")
            print(f"  Preprocess load : {t.get('preprocess_load_sec', 0):>6.2f}s (cold only)")
            print(f"  Preprocessing   : {t.get('preprocess_sec', 0):>6.2f}s")
            print(f"  Model load      : {t.get('model_load_sec', 0):>6.2f}s (cold only)")
            print(f"  GPU inference   : {t['inference_sec']:>6.2f}s ({args.steps} steps)")
            print(f"  Container total : {t['total_container_sec']:>6.2f}s")
            print(f"  Grand total     : {total_elapsed:>6.2f}s")
            print(f"\nSaved:     {out_path}")
            print(f"Composite: {comp_path}")
            print(f"LoRA applied: {result.get('lora_applied', False)}")

            import os
            try:
                # Open the composite so all three panels appear together
                os.startfile(str(comp_path if comp_path.exists() else out_path))
            except Exception:
                pass

            completed += 1
            print()

    except KeyboardInterrupt:
        print("\n\nInterrupted.\n")

    print(f"Session complete. {completed} test(s) in {(time.time()-session_start)/60:.1f} min.")


if __name__ == "__main__":
    main()

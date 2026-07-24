"""Interactive IDM-VTON tester — batch mode. Test one, see result, test next.

IDM-VTON (ECCV 2024, SDXL backbone) — better garment fidelity than Leffa.
Runs on Modal L4 GPU. No local GPU needed.

Run:
    cd "D:\\madon finalization"
    .\.venv\Scripts\python.exe test_idm.py

Flow (repeats until Cancel):
    1. Pick person image
    2. Pick garment image
    3. Pick garment type
    4. Normalize both to 768x1024 (aspect-ratio preserving, white pad)
    5. Run IDM-VTON on Modal L4 @ 30 steps
    6. Result opens in default viewer
    7. Loop back to step 1 (or Cancel to quit)

Container stays warm between runs so second+ tests skip cold-start (~40s vs ~90s).
"""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import modal

from preprocess_images import prepare_pair
from PIL import Image, ImageDraw, ImageFont


STEPS          = 30
GUIDANCE_SCALE = 2.0

IMG_TYPES = [
    ("Images", "*.jpg *.jpeg *.png *.webp *.bmp"),
    ("All files", "*.*"),
]


def pick_file(title: str, initial_dir: Path | None = None) -> Path | None:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title=title,
        filetypes=IMG_TYPES,
        initialdir=str(initial_dir) if initial_dir else None,
    )
    root.destroy()
    return Path(path) if path else None


def make_composite(
    person_path: Path,
    garment_path: Path,
    result_path: Path,
    out_path: Path,
) -> None:
    """Create a side-by-side composite: Person | Garment | Output."""
    try:
        person = Image.open(person_path).convert("RGB")
        garment = Image.open(garment_path).convert("RGB")
        result = Image.open(result_path).convert("RGB")

        # Ensure all same height (768 or closest multiple)
        target_h = 768
        person = person.resize((768, target_h)) if person.size[1] != target_h else person
        garment = garment.resize((768, target_h)) if garment.size[1] != target_h else garment
        result = result.resize((768, target_h)) if result.size[1] != target_h else result

        # Create composite: 3 cols × 1 row, plus 60px header for labels
        comp_w = 768 * 3 + 10 * 2  # 3 images + 2 gaps
        comp_h = target_h + 60
        composite = Image.new("RGB", (comp_w, comp_h), color="white")

        # Paste images
        composite.paste(person, (0, 60))
        composite.paste(garment, (768 + 10, 60))
        composite.paste(result, (768 * 2 + 20, 60))

        # Add labels at top
        draw = ImageDraw.Draw(composite)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except (OSError, IOError):
            font = ImageFont.load_default()

        labels = ["Person", "Garment", "Result (wearing garment)"]
        x_positions = [150, 768 + 150 + 10, 768 * 2 + 150 + 20]
        for label, x in zip(labels, x_positions):
            draw.text((x, 15), label, fill="black", font=font)

        composite.save(str(out_path))
    except Exception as e:
        print(f"⚠ Composite creation failed: {e}")


def pick_garment_type() -> str | None:
    root = tk.Tk()
    root.title("Garment Type")
    root.geometry("320x200")
    result = {"value": None}

    def choose(t: str) -> None:
        result["value"] = t
        root.destroy()

    tk.Label(root, text="Select garment type:", font=("Segoe UI", 11)).pack(pady=12)
    tk.Button(root, text="Upper Body (shirt, jacket, tee)",
              command=lambda: choose("upper_body"), width=32).pack(pady=3)
    tk.Button(root, text="Lower Body (pants, shorts, skirt)",
              command=lambda: choose("lower_body"), width=32).pack(pady=3)
    tk.Button(root, text="Full Dress",
              command=lambda: choose("dresses"), width=32).pack(pady=3)
    root.mainloop()
    return result["value"]


def main() -> None:
    print("=== IDM-VTON — Batch Test Mode ===")
    print("Test one, see result, test next. Cancel any picker to quit.\n")

    project_dir = Path(__file__).parent
    out_dir = project_dir / "outputs"
    out_dir.mkdir(exist_ok=True)

    print("Connecting to Modal (idm-vton app) ...")
    run_idm = modal.Function.from_name("idm-vton", "run_idm")
    print("✓ Connected.\n")

    completed = 0
    last_dir: Path | None = None
    session_start = time.time()

    try:
        while True:
            print(f"{'=' * 70}")
            print(f"  Test {completed + 1}   (session: {(time.time() - session_start) / 60:.1f} min)")
            print(f"{'=' * 70}")

            print("Opening file picker for PERSON image (Cancel to quit)...")
            person = pick_file("Select PERSON image (Cancel to quit)", last_dir or project_dir)
            if not person:
                print("\nCancelled — exiting.\n"); break
            print(f"  person  = {person}")

            print("Opening file picker for GARMENT image (Cancel to quit)...")
            garment = pick_file("Select GARMENT image (Cancel to quit)", person.parent)
            if not garment:
                print("\nCancelled — exiting.\n"); break
            print(f"  garment = {garment}")

            garment_type = pick_garment_type()
            if not garment_type:
                print("\nCancelled — exiting.\n"); break
            print(f"  type    = {garment_type}")
            print(f"  model   = IDM-VTON (SDXL)  |  steps = {STEPS}\n")

            composite_path = None  # will be set if composite creation succeeds
            print("Normalizing person + garment to 768x1024 ...")
            person_bytes, garment_bytes = prepare_pair(person, garment, save_debug_to=out_dir)

            print("Sending to Modal L4 GPU ...")
            t0 = time.time()
            result = run_idm.remote(
                person_bytes,
                garment_bytes,
                garment_type=garment_type,
                steps=STEPS,
                guidance_scale=GUIDANCE_SCALE,
            )
            total_elapsed = time.time() - t0

            result_bytes = result["image"]
            timings      = result["timings"]
            steps        = result["steps"]

            stamp    = time.strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"vton_idm_{stamp}.png"
            out_path.write_bytes(result_bytes)

            # Create composite image (person | garment | result)
            person_input  = out_dir / "_input_person.jpg"
            garment_input = out_dir / "_input_garment.jpg"
            comp_path = out_dir / f"vton_idm_{stamp}_composite.png"
            if person_input.exists() and garment_input.exists():
                make_composite(person_input, garment_input, out_path, comp_path)
                composite_path = comp_path

            # ── Timing breakdown ──────────────────────────────────────────────────────
            infer_sec    = timings["inference_sec"]
            model_load   = timings["model_load_sec"]
            preproc_load = timings.get("preprocess_load_sec", 0)
            preproc_sec  = timings.get("preprocess_sec", 0)
            container    = timings["total_container_sec"]
            network_sec  = total_elapsed - container

            print(f"\n=== TIMING BREAKDOWN ===")
            print(f"  Preprocessor load  : {preproc_load:>6.2f}s  (cold start only)")
            print(f"  Preprocessing      : {preproc_sec:>6.2f}s")
            print(f"  Model load         : {model_load:>6.2f}s  (cold start only)")
            print(f"  GPU inference      : {infer_sec:>6.2f}s  ({steps} steps, {infer_sec/steps*1000:.0f} ms/step)")
            print(f"  Container total    : {container:>6.2f}s")
            print(f"  Network + overhead : {network_sec:>6.2f}s")
            print(f"  Grand total        : {total_elapsed:>6.2f}s")
            print(f"\nSaved: {out_path}")
            if composite_path.exists():
                print(f"Composite: {composite_path}")

            import os
            # Open composite if it exists, otherwise fall back to single result
            to_open = composite_path if composite_path.exists() else out_path
            os.startfile(str(to_open))
            print(f"Opened in viewer.\n")

            last_dir = person.parent
            completed += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by Ctrl-C.\n")

    print(f"Session complete. {completed} test(s) in {(time.time() - session_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()

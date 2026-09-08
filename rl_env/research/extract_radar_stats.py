"""
Extract real radar parameters from the Turing Synthetic Radar Dataset
(Scan Mode) to ground Chahek's ScanEnv emitter models in real simulated
data instead of public-estimate guesses.

Setup (one-time):
    pip install huggingface_hub jaxtyping torch python-dotenv h5py
    git clone https://github.com/alan-turing-institute/turing-deinterleaving-challenge.git
    cd turing-deinterleaving-challenge && pip install -e .
    cp .env.example .env   # then edit .env: HUGGING_FACE_TOKEN=hf_your_token_here

Usage:
    python extract_radar_stats.py
"""
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # reads HUGGING_FACE_TOKEN from .env — never hardcode it in this file


# ---------------------------------------------------------------------------
# Step 1: Download (only runs the actual network fetch if data isn't there yet)
# ---------------------------------------------------------------------------
def ensure_dataset_downloaded(save_dir: Path) -> Path:
    if any(save_dir.rglob("*.h5")):
        print(f"Data already present under {save_dir}, skipping download.")
        return save_dir

    from turing_deinterleaving_challenge import download_dataset

    print("Downloading dataset (train / scan mode only)...")
    dataset_path = download_dataset(
        save_dir=save_dir,
        subsets=["train"],
        modes=["scan"],  # scan mode = realistic sweeping receiver, what you actually need
    )
    print("Downloaded to:", dataset_path)
    return dataset_path


# ---------------------------------------------------------------------------
# Step 2: Extract real receiver + transmitter stats from one config file
# ---------------------------------------------------------------------------
def extract_stats(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        dwell_centres_mhz = f["metadata/receiver/dwell_centres_mhz"][:]
        dwell_times_s = f["metadata/receiver/dwell_times_s"][:]
        freq_range_mhz = f["metadata/receiver/freq_range_mhz"][:]

        tx_group = f["metadata/transmitters"]
        n_freqs_per_tx = []
        all_pris_us = []
        all_pws_us = []

        for tx_name in tx_group.keys():
            freqs = tx_group[f"{tx_name}/frequency_config/freqs_mhz"][:]
            pris = tx_group[f"{tx_name}/pri_config/pris_us"][:]
            pws = tx_group[f"{tx_name}/pulse_width_config/pws_us"][:]
            n_freqs_per_tx.append(len(freqs))
            all_pris_us.extend(pris.tolist())
            all_pws_us.extend(pws.tolist())

        n_freqs_per_tx = np.array(n_freqs_per_tx)
        all_pris_us = np.array(all_pris_us)
        all_pws_us = np.array(all_pws_us)

        n_fixed = int((n_freqs_per_tx == 1).sum())
        n_multi = int((n_freqs_per_tx > 1).sum())

        return {
            "source_file": str(h5_path),
            "num_bands": int(len(dwell_centres_mhz)),
            "band_freq_mhz": np.round(dwell_centres_mhz, 1).tolist(),
            "freq_range_mhz": freq_range_mhz.tolist(),
            "dwell_time_s_min": float(dwell_times_s.min()),
            "dwell_time_s_max": float(dwell_times_s.max()),
            "num_transmitters": int(len(n_freqs_per_tx)),
            "num_fixed_freq_emitters": n_fixed,
            "num_multi_freq_emitters": n_multi,
            "fixed_ratio": round(n_fixed / len(n_freqs_per_tx), 3),
            "multi_freq_ratio": round(n_multi / len(n_freqs_per_tx), 3),
            "max_freqs_single_emitter": int(n_freqs_per_tx.max()),
            "pri_us_min": float(all_pris_us.min()),
            "pri_us_max": float(all_pris_us.max()),
            "pri_us_mean": float(all_pris_us.mean()),
            "pulse_width_us_min": float(all_pws_us.min()),
            "pulse_width_us_max": float(all_pws_us.max()),
            "pulse_width_us_mean": float(all_pws_us.mean()),
        }


# ---------------------------------------------------------------------------
# Step 2b: Export one row per transmitter to CSV (the actual values, not
# just aggregate stats) — this is what Chahek/Roshni can open directly in
# Excel or paste into the architecture doc.
# ---------------------------------------------------------------------------
def export_transmitters_csv(h5_path: Path, out_path: Path):
    with h5py.File(h5_path, "r") as f:
        tx_group = f["metadata/transmitters"]

        rows = []
        for tx_name in sorted(tx_group.keys(), key=lambda n: int(n.split("_")[-1])):
            freqs = tx_group[f"{tx_name}/frequency_config/freqs_mhz"][:]
            pris = tx_group[f"{tx_name}/pri_config/pris_us"][:]
            pws = tx_group[f"{tx_name}/pulse_width_config/pws_us"][:]

            rows.append({
                "transmitter": tx_name,
                "num_freqs": len(freqs),
                "freqs_mhz": ";".join(str(round(v, 1)) for v in freqs),
                "num_pris": len(pris),
                "pris_us": ";".join(str(round(v, 1)) for v in pris),
                "num_pulse_widths": len(pws),
                "pulse_widths_us": ";".join(str(round(v, 2)) for v in pws),
                "emitter_type_guess": "fixed" if len(freqs) == 1 else "multi_freq",
            })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved per-transmitter CSV to {out_path.resolve()} ({len(rows)} rows).")


# ---------------------------------------------------------------------------
# Step 3: Run it and save a summary Chahek can drop straight into scan_env.py
# ---------------------------------------------------------------------------
def main():
    data_dir = Path("./data")
    ensure_dataset_downloaded(data_dir)

    h5_files = sorted(data_dir.rglob("*.h5"))
    print(f"Found {len(h5_files)} H5 files.")
    if not h5_files:
        print("No files found — check the download step above.")
        return

    # Use the first file as the reference config (add more files here later
    # if you want averaged stats across several configs instead of one).
    reference_file = h5_files[0]

    stats = extract_stats(reference_file)
    print(json.dumps(stats, indent=2))

    json_path = Path("radar_stats_summary.json")
    json_path.write_text(json.dumps(stats, indent=2))
    print(f"\nSaved summary to {json_path.resolve()}")

    csv_path = Path("radar_transmitters.csv")
    export_transmitters_csv(reference_file, csv_path)

    print(f"\nHand both files to Chahek: {json_path.name} (aggregate stats) "
          f"and {csv_path.name} (per-transmitter detail).")


if __name__ == "__main__":
    main()

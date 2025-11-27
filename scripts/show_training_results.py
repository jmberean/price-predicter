"""Display training results and metrics for all SOTA components."""

import sys
sys.path.insert(0, "src")

from pathlib import Path

print("=" * 70)
print("SOTA Model Training Results")
print("=" * 70)

artifact_dir = Path("artifacts")

# Check which models are trained
models = {
    "NCC Calibration": "ncc_calibration.pt",
    "FM-GP Residuals": "fmgp_residuals.pt",
    "Neural Jump SDE": "neural_jump_sde.pt",
    "Differentiable Greeks": "diff_greeks.pt",
    "Neural Rough Vol": "neural_rough_vol.pt",
    "MambaTS Trend": "mamba_trend.pt",
}

print("\n✅ Trained Models:")
for name, filename in models.items():
    filepath = artifact_dir / filename
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  {name:25s} - {filename:30s} ({size_mb:.2f} MB)")
    else:
        print(f"  {name:25s} - NOT TRAINED ❌")

print("\n" + "=" * 70)
print("Training Configuration (from .env)")
print("=" * 70)

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

config = {
    "Lookback Days": os.getenv("TRAINING_LOOKBACK_DAYS", "90"),
    "Sample Count (FM-GP)": os.getenv("TRAINING_SAMPLES_FMGP", "150"),
    "Sample Count (Neural Vol)": os.getenv("TRAINING_SAMPLES_NEURAL_VOL", "150"),
    "Sample Count (Jump SDE)": os.getenv("TRAINING_SAMPLES_NEURAL_JUMP", "150"),
    "Sample Count (MambaTS)": os.getenv("TRAINING_SAMPLES_MAMBA", "150"),
    "Sample Count (NCC)": os.getenv("TRAINING_SAMPLES_NCC", "150"),
    "Sample Count (Diff Greeks)": os.getenv("TRAINING_SAMPLES_DIFF_GREEKS", "100"),
    "Window Days": os.getenv("TRAINING_WINDOW_DAYS", "90"),
    "Device": os.getenv("TORCH_DEVICE", "cpu"),
}

for key, value in config.items():
    print(f"  {key:30s}: {value}")

print("\n" + "=" * 70)
print("To Retrain or View Training Metrics")
print("=" * 70)
print("\n1. Train specific component with visible metrics:")
print("   .venv/Scripts/python -m aetheris_oracle.pipeline.train_sota \\")
print("     --component mamba --asset BTC-USD --epochs 40")
print("\n2. Train all components:")
print("   .venv/Scripts/python -m aetheris_oracle.pipeline.train_sota \\")
print("     --component all --asset BTC-USD")
print("\n3. Validate all models:")
print("   .venv/Scripts/python scripts/validate_all_sota.py")
print("\n" + "=" * 70)

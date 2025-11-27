"""Quick training test with minimal samples for verification."""

import sys
sys.path.insert(0, "src")

from aetheris_oracle.data.free_connectors import FreeDataConnector
from aetheris_oracle.pipeline.train_sota import (
    train_ncc_calibration,
    train_neural_jump_sde,
    train_differentiable_greeks,
)

print("="*60)
print("Quick SOTA Training Test - Minimal Samples")
print("="*60)

connector = FreeDataConnector(enable_cache=True)

# Test 1: NCC with minimal samples
print("\n[1/3] Training NCC (minimal)...")
try:
    metrics = train_ncc_calibration(
        connector=connector,
        asset_id="BTC-USD",
        epochs=2,  # Just 2 epochs for test
        batch_size=8,
        device="cpu",
        artifact_path="artifacts/ncc_calibration_test.pt",
    )
    print(f"✓ NCC training completed: {metrics}")
except Exception as e:
    print(f"✗ NCC training failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Neural Jump SDE
print("\n[2/3] Training Neural Jump SDE (minimal)...")
try:
    metrics = train_neural_jump_sde(
        connector=connector,
        asset_id="BTC-USD",
        epochs=2,
        batch_size=8,
        device="cpu",
        artifact_path="artifacts/neural_jump_sde_test.pt",
    )
    print(f"✓ Neural Jump SDE training completed: {metrics}")
except Exception as e:
    print(f"✗ Neural Jump SDE training failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Differentiable Greeks
print("\n[3/3] Training Differentiable Greeks (minimal)...")
try:
    metrics = train_differentiable_greeks(
        connector=connector,
        asset_id="BTC-USD",
        epochs=2,
        batch_size=4,
        device="cpu",
        artifact_path="artifacts/diff_greeks_test.pt",
    )
    print(f"✓ Diff Greeks training completed: {metrics}")
except Exception as e:
    print(f"✗ Diff Greeks training failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Training test complete!")
print("="*60)

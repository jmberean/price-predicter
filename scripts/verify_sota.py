"""
Quick verification script to check if SOTA models are enabled and loaded.
"""
import sys
sys.path.insert(0, "src")

from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

print("=" * 70)
print("🔍 SOTA Configuration Verification")
print("=" * 70)

# Check environment variables
print("\n📋 Environment Variables:")
print(f"  USE_NEURAL_ROUGH_VOL: {os.getenv('USE_NEURAL_ROUGH_VOL')}")
print(f"  USE_FM_GP_RESIDUALS: {os.getenv('USE_FM_GP_RESIDUALS')}")
print(f"  NEURAL_ROUGH_VOL_PATH: {os.getenv('NEURAL_ROUGH_VOL_PATH')}")
print(f"  FMGP_RESIDUALS_PATH: {os.getenv('FMGP_RESIDUALS_PATH')}")
print(f"  TORCH_DEVICE: {os.getenv('TORCH_DEVICE')}")

# Check if artifact files exist
print("\n📦 Artifact Files:")
neural_vol_path = Path(os.getenv('NEURAL_ROUGH_VOL_PATH', 'artifacts/neural_rough_vol_sota.pt'))
fmgp_path = Path(os.getenv('FMGP_RESIDUALS_PATH', 'artifacts/fmgp_residuals_sota.pt'))

print(f"  Neural Rough Vol: {neural_vol_path} - {'✅ EXISTS' if neural_vol_path.exists() else '❌ MISSING'}")
print(f"  FM-GP Residuals: {fmgp_path} - {'✅ EXISTS' if fmgp_path.exists() else '❌ MISSING'}")

# Try to load models
print("\n🧪 Model Loading Test:")
try:
    from aetheris_oracle.modules.neural_rough_vol import NeuralRoughVolWrapper
    wrapper = NeuralRoughVolWrapper.load(neural_vol_path, device="cpu")
    print(f"  ✅ Neural Rough Vol loaded successfully")
    print(f"     Horizon: {wrapper.config.horizon}")
except Exception as e:
    print(f"  ❌ Failed to load Neural Rough Vol: {e}")

try:
    from aetheris_oracle.modules.fm_gp_residual import FMGPResidualEngine
    engine = FMGPResidualEngine.load(fmgp_path, device="cpu")
    print(f"  ✅ FM-GP Residuals loaded successfully")
    print(f"     Horizon: {engine.config.horizon}")
except Exception as e:
    print(f"  ❌ Failed to load FM-GP Residuals: {e}")

# Check if ForecastEngine will use them
print("\n🚀 ForecastEngine Integration:")
from aetheris_oracle.config import ForecastConfig
from aetheris_oracle.data.free_connectors import FreeDataConnector
from aetheris_oracle.pipeline.forecast import ForecastEngine

config = ForecastConfig(asset_id="BTC-USD", horizon_days=7)
connector = FreeDataConnector()

# Check if engine auto-enables SOTA based on .env
engine = ForecastEngine(connector=connector)

print(f"  Neural Rough Vol enabled: {engine.use_neural_rough_vol}")
print(f"  FM-GP Residuals enabled: {engine.use_fm_gp_residuals}")

if engine.use_neural_rough_vol or engine.use_fm_gp_residuals:
    print("\n  ✅ SOTA models are ENABLED and will be used in forecasts!")
else:
    print("\n  ⚠️  SOTA models are NOT enabled. Using legacy models.")

print("\n" + "=" * 70)

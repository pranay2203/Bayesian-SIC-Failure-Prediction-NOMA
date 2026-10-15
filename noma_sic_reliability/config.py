"""Central configuration for the reproducible NOMA experiment."""
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
DATA_DIR, MODEL_DIR, METRICS_DIR, PLOTS_DIR = [OUTPUT / x for x in ("data", "models", "metrics", "plots")]

N_SAMPLES = 7000
TEST_SIZE = 0.20
BATCH_SIZE = 128
EPOCHS = 55
LEARNING_RATE = 1e-3
MC_SAMPLES = 40
N_SCHEDULING_USERS = 80
NOISE_POWER_DBM = -100.0

FEATURE_COLUMNS = [
    "snr_db", "channel_gain_db", "received_power_dbm", "interference_dbm",
    "channel_estimation_error", "power_gap_db", "mobility_kmh", "user_distance_m",
]

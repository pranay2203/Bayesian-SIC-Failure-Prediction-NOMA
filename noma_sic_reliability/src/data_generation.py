"""Create a physically motivated, synthetic SIC decoding dataset."""
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def generate_dataset(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Generate one row per SIC decoding attempt with a stochastic failure label."""
    rng = np.random.default_rng(seed)
    distance = rng.uniform(40, 500, n_samples)
    path_loss = 35 + 30 * np.log10(distance) + rng.normal(0, 4, n_samples)
    channel_gain_db = -path_loss
    tx_power_dbm = rng.uniform(18, 32, n_samples)
    received_power = tx_power_dbm + channel_gain_db
    interference = rng.uniform(-112, -78, n_samples) + rng.normal(0, 2, n_samples)
    noise_dbm = -100.0
    # SINR reflects both thermal noise and inter-user interference.
    linear_interference = 10 ** (interference / 10) + 10 ** (noise_dbm / 10)
    # A 10 dB implementation margin represents practical receiver losses.
    snr_db = received_power - 10 * np.log10(linear_interference) - 10.0
    estimation_error = rng.beta(2, 10, n_samples) * 0.32
    power_gap = rng.uniform(1, 18, n_samples)  # SIC is easier with a clear power separation.
    mobility = rng.exponential(25, n_samples).clip(0, 140)

    # Latent probability: intentionally combines link quality and non-ideal SIC effects.
    logit = (-0.34 * snr_db + 0.115 * (interference + 100) + 8.0 * estimation_error
             - 0.15 * power_gap + 0.012 * mobility + rng.normal(0, 0.45, n_samples) - 0.6)
    true_failure_probability = sigmoid(logit)
    failure = rng.binomial(1, true_failure_probability)
    return pd.DataFrame({
        "snr_db": snr_db, "channel_gain_db": channel_gain_db,
        "received_power_dbm": received_power, "interference_dbm": interference,
        "channel_estimation_error": estimation_error, "power_gap_db": power_gap,
        "mobility_kmh": mobility, "user_distance_m": distance,
        "true_failure_probability": true_failure_probability, "sic_failure": failure,
    })


def make_pair_feature(strong: pd.Series, weak: pd.Series) -> dict:
    """Derive a model input describing SIC by a stronger receiver for a weak user's signal."""
    return {
        "snr_db": float(strong.snr_db), "channel_gain_db": float(strong.channel_gain_db),
        "received_power_dbm": float(strong.received_power_dbm),
        "interference_dbm": float((strong.interference_dbm + weak.interference_dbm) / 2),
        "channel_estimation_error": float(strong.channel_estimation_error),
        "power_gap_db": float(abs(strong.received_power_dbm - weak.received_power_dbm)),
        "mobility_kmh": float((strong.mobility_kmh + weak.mobility_kmh) / 2),
        "user_distance_m": float(strong.user_distance_m),
    }

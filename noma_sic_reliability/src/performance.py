"""Evaluate pair schedules with a stochastic imperfect-SIC NOMA link model."""
import numpy as np
import pandas as pd
from .data_generation import sigmoid


def evaluate_schedule(pairs, users, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    by_id = users.set_index("user_id")
    for _, pair in pairs.iterrows():
        strong, weak = by_id.loc[pair.strong_id], by_id.loc[pair.weak_id]
        snr = strong.snr_db
        gap = abs(strong.received_power_dbm - weak.received_power_dbm)
        # Independent link approximation gives an evaluation truth separate from ML output.
        pair_interference = (strong.interference_dbm + weak.interference_dbm) / 2
        actual_p = sigmoid(-0.31 * snr + 0.11 * (pair_interference + 100)
                           + 8.2 * strong.channel_estimation_error - 0.14 * gap
                           + 0.012 * strong.mobility_kmh - 0.55)
        failed = rng.random() < actual_p
        effective_snr = snr - (10.0 if failed else 0.8) - 0.22 * strong.mobility_kmh / 10
        ber = 0.5 * np.exp(-max(effective_snr, -8) / 2.2)
        ber = float(np.clip(ber, 1e-6, 0.5))
        outage = float(effective_snr < 3.0 or failed)
        spectral_efficiency = np.log2(1 + 10 ** (max(effective_snr, -10) / 10))
        throughput = 0.0 if outage else 10.0 * spectral_efficiency  # Mbps in a nominal 10 MHz channel
        rows.append({"actual_sic_failure": int(failed), "actual_failure_probability": actual_p, "ber": ber,
                     "outage": outage, "throughput_mbps": throughput, "reliability": 1 - outage})
    detail = pd.DataFrame(rows)
    metric = {"pairs": len(detail), "predicted_sic_failure_probability": pairs.predicted_failure_probability.mean(),
              "mean_uncertainty": pairs.uncertainty.mean(), "sic_failure_rate": detail.actual_sic_failure.mean(),
              "ber": detail.ber.mean(), "outage_probability": detail.outage.mean(),
              "throughput_mbps": detail.throughput_mbps.mean(), "reliability": detail.reliability.mean()}
    return metric, detail

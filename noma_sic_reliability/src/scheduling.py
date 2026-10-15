"""User-pairing policies for conventional and reliability-aware NOMA."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .data_generation import make_pair_feature


def _pair_risk(a, b, scaler, model, feature_columns, predictor):
    # Stronger (higher SNR) user performs SIC; assess risk for its cancellation.
    strong, weak = (a, b) if a.snr_db >= b.snr_db else (b, a)
    x = pd.DataFrame([make_pair_feature(strong, weak)])[feature_columns]
    probability, uncertainty, _ = predictor(model, scaler.transform(x.values), mc_samples=35)
    return float(probability[0]), float(uncertainty[0]), strong, weak


def conventional_pairs(users, scaler, model, feature_columns, predictor):
    """Classic NOMA policy: pair best channel user with worst channel user."""
    ordered = users.sort_values("snr_db").reset_index(drop=True)
    records = []
    for i in range(len(ordered) // 2):
        risk, unc, strong, weak = _pair_risk(ordered.iloc[-(i + 1)], ordered.iloc[i], scaler, model, feature_columns, predictor)
        records.append({"strong_id": int(strong.user_id), "weak_id": int(weak.user_id), "predicted_failure_probability": risk, "uncertainty": unc})
    return pd.DataFrame(records)


def reliability_aware_pairs(users, scaler, model, feature_columns, predictor):
    """Find the minimum-total-risk matching among the same near/far user pools.

    Keeping the conventional strong and weak pools makes this a fair comparison:
    the only changed decision is which weak user is matched to each strong user.
    """
    ordered = users.sort_values("snr_db").reset_index(drop=True)
    weak_pool = ordered.iloc[:len(ordered) // 2]
    strong_pool = ordered.iloc[len(ordered) // 2:]
    costs = np.zeros((len(strong_pool), len(weak_pool)))
    cached = {}
    for i, (_, strong) in enumerate(strong_pool.iterrows()):
        for j, (_, weak) in enumerate(weak_pool.iterrows()):
            risk, unc, selected_strong, selected_weak = _pair_risk(strong, weak, scaler, model, feature_columns, predictor)
            cached[i, j] = (risk, unc, selected_strong, selected_weak)
            costs[i, j] = risk + 0.50 * unc
    strong_indices, weak_indices = linear_sum_assignment(costs)
    records = []
    for i, j in zip(strong_indices, weak_indices):
        risk, unc, strong, weak = cached[i, j]
        records.append({"strong_id": int(strong.user_id), "weak_id": int(weak.user_id), "predicted_failure_probability": risk, "uncertainty": unc})
    return pd.DataFrame(records)

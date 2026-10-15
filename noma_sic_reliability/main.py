"""Run: python main.py. Generates dataset, trains Bayesian SIC predictor, compares schedules."""
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from config import *
from src.data_generation import generate_dataset
from src.preprocessing import prepare_data
from src.bayesian_model import train_model, predict_with_uncertainty, set_seed
from src.scheduling import conventional_pairs, reliability_aware_pairs
from src.performance import evaluate_schedule
from src.visualization import save_plots


def main():
    for directory in (DATA_DIR, MODEL_DIR, METRICS_DIR, PLOTS_DIR): directory.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    data = generate_dataset(N_SAMPLES, SEED); data.to_csv(DATA_DIR / "synthetic_sic_dataset.csv", index=False)
    x_train, x_test, y_train, y_test, scaler = prepare_data(data, FEATURE_COLUMNS, TEST_SIZE, SEED)
    model, losses = train_model(x_train, y_train, EPOCHS, BATCH_SIZE, LEARNING_RATE, SEED)
    np.savez(MODEL_DIR / "mc_dropout_sic_model.npz", w1=model.w1, b1=model.b1, w2=model.w2,
             b2=model.b2, feature_names=np.array(FEATURE_COLUMNS))
    test_prob, test_unc, _ = predict_with_uncertainty(model, x_test, MC_SAMPLES)
    prediction_metrics = {"accuracy": float(accuracy_score(y_test, test_prob >= .5)), "roc_auc": float(roc_auc_score(y_test, test_prob)),
                          "brier_score": float(brier_score_loss(y_test, test_prob)), "mean_uncertainty": float(test_unc.mean())}
    rng = np.random.default_rng(SEED + 9)
    users = generate_dataset(N_SCHEDULING_USERS, SEED + 1).drop(columns=["sic_failure", "true_failure_probability"])
    users["user_id"] = np.arange(N_SCHEDULING_USERS); users = users.sample(frac=1, random_state=SEED).reset_index(drop=True)
    conventional = conventional_pairs(users, scaler, model, FEATURE_COLUMNS, predict_with_uncertainty)
    aware = reliability_aware_pairs(users, scaler, model, FEATURE_COLUMNS, predict_with_uncertainty)
    conventional.to_csv(METRICS_DIR / "conventional_pairs.csv", index=False); aware.to_csv(METRICS_DIR / "reliability_aware_pairs.csv", index=False)
    c_metrics, _ = evaluate_schedule(conventional, users, SEED); r_metrics, _ = evaluate_schedule(aware, users, SEED)
    comparison = pd.DataFrame([{**c_metrics, "scheduler": "Conventional"}, {**r_metrics, "scheduler": "Reliability-aware"}])
    comparison.to_csv(METRICS_DIR / "final_comparison.csv", index=False)
    with open(METRICS_DIR / "model_metrics.json", "w") as f: json.dump(prediction_metrics, f, indent=2)
    save_plots(losses, test_prob, test_unc, y_test, comparison, PLOTS_DIR)
    print("\nBayesian SIC prediction metrics:", json.dumps(prediction_metrics, indent=2))
    print("\nFinal scheduling comparison:\n", comparison.round(4).to_string(index=False))
    print(f"\nArtifacts saved under: {OUTPUT}")


if __name__ == "__main__": main()

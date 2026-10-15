# Bayesian Deep Learning for Imperfect-SIC NOMA

This college-project simulation predicts the probability that successive interference cancellation (SIC) fails, estimates prediction uncertainty with Monte Carlo (MC) dropout, and uses the prediction to construct lower-risk NOMA user pairs.

## Pipeline

`NOMA parameters → synthetic data → preprocessing → MC-dropout predictor → failure probability + uncertainty → pairing → link simulation → metrics + plots`

## Setup and execution

From this folder, create and activate a virtual environment, then install the packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

On macOS/Linux, activate with `source .venv/bin/activate` instead. The first PyTorch installation may take a little longer.

## What the code does

- `config.py`: experiment sizes, seeds, model training choices, and output paths.
- `src/data_generation.py`: samples link/channel/user conditions and a stochastic imperfect-SIC outcome.
- `src/preprocessing.py`: makes a stratified split and standardizes inputs.
- `src/bayesian_model.py`: NumPy neural network with dropout; repeated inference produces probability mean and uncertainty standard deviation.
- `src/scheduling.py`: compares classic strongest-with-weakest pairing against a risk-and-uncertainty-minimizing scheduler.
- `src/performance.py`: evaluates schedules using independently sampled imperfect-SIC outcomes, BER, outage, throughput, and reliability.
- `src/visualization.py`: writes training, uncertainty, and comparison graphics.

## Outputs

Running `main.py` creates:

- `outputs/data/synthetic_sic_dataset.csv` — the generated training data.
- `outputs/models/mc_dropout_sic_model.npz` — trained MC-dropout model weights.
- `outputs/metrics/model_metrics.json` — classifier quality and mean uncertainty.
- `outputs/metrics/final_comparison.csv` — conventional versus reliability-aware performance.
- `outputs/plots/*.png` — visual evidence for the report or presentation.

All random seeds are fixed. The numerical results are produced at runtime; no result values are hard-coded.

## Notes for presentation

The MC-dropout standard deviation measures model uncertainty: higher values signal less confidence. The reliability-aware scheduler penalizes both predicted SIC-failure probability and this uncertainty, while using the same near/far candidate pools as the conventional method for a fair comparison.

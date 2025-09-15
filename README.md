# Jefferson DS Postdoc — Panel Problems (TD3, GAN, CL)

Reproducible code and docs for the Jefferson Lab Data Scientist Postdoc panel on **September 18**.
This repo contains:
- **Problem 1**: TD3 on `Gymnasium` `Pendulum-v1` with learning-behavior plots.
- **Problem 2**: Keras GAN reproducing the eICU age distribution; includes comparison plots (hist/KDE, CDF, Q–Q).
- **Problem 3**: Short literature review and an efficiency-aware solution blueprint for task-incremental Continual Learning with known drifts.

> Tools: **Python 3**, **TensorFlow/Keras**, **Gymnasium**.

## Quickstart

### Create environment
```bash
# (Option A) Conda
conda env create -f environment.yml
conda activate jlab-postdoc

# (Option B) venv + pip
python -m venv .venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Problem 1: TD3 on Pendulum-v1

### Train
```bash
python problem1_td3_pendulum/train_td3.py --total-steps 150000 --seed 0 --log-every 1000
```

### Evaluate
```bash
python problem1_td3_pendulum/eval_td3.py --checkpoint problem1_td3_pendulum/results/td3_actor.keras
```

Outputs:
- `problem1_td3_pendulum/plots/learning_returns.png`
- `problem1_td3_pendulum/plots/losses.png`

---

## Problem 2: GAN for eICU Age Distribution

`train_gan.py` will download `eICU_age.npy` automatically into the folder if not present.

### Train
```bash
python problem2_gan_age/train_gan.py --epochs 2000 --batch-size 256 --latent-dim 64
```

### Evaluate & plots
```bash
python problem2_gan_age/eval_gan.py --num-samples 100000
```
Outputs:
- `problem2_gan_age/figures/hist_overlay.png`
- `problem2_gan_age/figures/cdf_overlay.png`
- `problem2_gan_age/figures/qq_plot.png`

---

## Problem 3: Continual Learning (CL)

See PDFs in `problem3_continual_learning/`:
- `CL_review.pdf`
- `CL_solution_blueprint.pdf`

---

## Reproducibility

- Fixed random seeds where applicable.
- Save checkpoints under each `results/` directory.
- Environment files included (`environment.yml`, `requirements.txt`).

## License
MIT

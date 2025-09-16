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

## Problem 1 — TD3 on Pendulum-v1

### Quick Start

#### 0) Setup
Open **Anaconda Prompt** and activate your project environment:
```cmd
conda activate jlab-postdoc
cd %USERPROFILE%\jefferson-postdoc
````

---

#### 1) Fast demo run (5–10 min on CPU)

Train and generate the core plots:

```cmd
set MPLBACKEND=Agg
python problem1_td3_pendulum\train_td3.py --total-steps 50000 --start-steps 1000 --batch-size 128 --log-every 2000
```

Evaluate the trained policy:

```cmd
python problem1_td3_pendulum\eval_td3.py --checkpoint problem1_td3_pendulum\results\td3_actor.keras
```

Open plots:

```cmd
explorer problem1_td3_pendulum\plots
```

Outputs:

* `learning_returns.png`
* `losses.png`

---

#### 2) Optional: Action statistics

Generate extra figures from the trained actor:

```cmd
set MPLBACKEND=Agg
python problem1_td3_pendulum\make_action_stats.py --checkpoint problem1_td3_pendulum\results\td3_actor.keras --episodes 5
explorer problem1_td3_pendulum\plots
```

Outputs:

* `action_stats.png`
* `action_episode_means.png`

---

#### 3) Better learning (if you have more time)

Run longer training for smoother curves:

```cmd
set MPLBACKEND=Agg
set TF_NUM_INTRAOP_THREADS=4
set TF_NUM_INTEROP_THREADS=2
set OMP_NUM_THREADS=4

python problem1_td3_pendulum\train_td3.py ^
  --total-steps 150000 ^
  --start-steps 8000 ^
  --batch-size 256 ^
  --exploration-noise 0.1 ^
  --log-every 5000 ^
  --seed 0
```

---

#### 4) Push results to GitHub

From the repo root:

```cmd
git add problem1_td3_pendulum\plots\*.png problem1_td3_pendulum\results\td3_actor.keras
git commit -m "TD3 results: learning curves + (optional) action stats"
git push
```

---

#### 5) Troubleshooting

* **No plots / TkAgg error**
  Use `set MPLBACKEND=Agg` (or make it permanent with

  ```cmd
  conda env config vars set MPLBACKEND=Agg
  conda deactivate & conda activate jlab-postdoc
  ```

  ).
* **Missing gymnasium**
  Ensure `(jlab-postdoc)` is active:

  ```cmd
  python -c "import gymnasium; print(gymnasium.__version__)"
  ```
* **Stop long run**
  Press **Ctrl+C** once. Checkpoints are saved before plotting.

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

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

---

## Problem 2 — GAN reproducing the eICU age distribution

This experiment trains a small GAN to model the empirical age distribution and produces three diagnostic figures:
`hist_overlay.png`, `cdf_overlay.png`, and `qq_plot.png`.

> **Outputs**  
> - Models/params → `problem2_gan_age/results/generator.keras`, `problem2_gan_age/results/norm_params.npz`  
> - Figures → `problem2_gan_age/figures/hist_overlay.png`, `cdf_overlay.png`, `qq_plot.png`

---

### 0) Setup

Open **Anaconda Prompt**:

```cmd
conda activate jlab-postdoc
cd %USERPROFILE%\jefferson-postdoc
````

---

### 1) Train (fast demo)

This writes the generator + normalization parameters into `results/`.

```cmd
set MPLBACKEND=Agg
python problem2_gan_age\train_gan.py --epochs 2000 --batch-size 256 --latent-dim 64
```

Check that files were saved:

```cmd
dir problem2_gan_age\results
```

You should see **non-zero** sizes for:

* `generator.keras`
* `norm_params.npz`

---

### 2) Evaluate & make plots

The evaluator uses a headless Matplotlib backend and always writes proper PNGs.

```cmd
python problem2_gan_age\eval_gan.py --num-samples 100000
dir problem2_gan_age\figures
explorer problem2_gan_age\figures
```

You should now have:

* `hist_overlay.png` — histogram/KDE overlay (true vs. generated)
* `cdf_overlay.png` — CDF curves overlay
* `qq_plot.png` — Q–Q plot (diagonal = perfect match)

---

### 3) Push results to GitHub

```cmd
git add problem2_gan_age\train_gan.py ^
        problem2_gan_age\results\generator.keras ^
        problem2_gan_age\results\norm_params.npz ^
        problem2_gan_age\figures\hist_overlay.png ^
        problem2_gan_age\figures\cdf_overlay.png ^
        problem2_gan_age\figures\qq_plot.png
git commit -m "Problem 2: GAN results (generator + norm params + figures)"
git push
```

---

### 4) Better training presets (optional)

If you have more time and want smoother figures:

**Balanced (panel-ready):**

```cmd
python problem2_gan_age\train_gan.py --epochs 4000 --batch-size 256 --latent-dim 64
python problem2_gan_age\eval_gan.py --num-samples 200000
```

**Stronger:**

```cmd
python problem2_gan_age\train_gan.py --epochs 8000 --batch-size 256 --latent-dim 64
python problem2_gan_age\eval_gan.py --num-samples 300000
```

Tips:

* If training is unstable, try a larger batch (e.g., `--batch-size 512`) or lower `--lr-g/--lr-d` inside `train_gan.py`.
* For quick checks, reduce `--num-samples` in eval (e.g., `50000`) to render faster.

---

### 5) Troubleshooting

**PNG files are 0 bytes**

* Likely `results/generator.keras` or `results/norm_params.npz` is missing. Re-run training and confirm they exist (non-zero size) before evaluating.
* Ensure headless plotting is enabled: `set MPLBACKEND=Agg`.

**`FileNotFoundError: norm_params.npz`**

* Train first; those params are written at the end of `train_gan.py`.

**No `results/` files after training**

* Make sure you’re using the revised `train_gan.py` that saves:

  * `problem2_gan_age/results/generator.keras`
  * `problem2_gan_age/results/norm_params.npz`

**Matplotlib/Tk errors**

* Use the non-GUI backend: `set MPLBACKEND=Agg` (or make it permanent with
  `conda env config vars set MPLBACKEND=Agg` → deactivate & reactivate the env).

---

### 6) File layout (after a successful run)

```
problem2_gan_age/
├─ data_loader.py
├─ train_gan.py
├─ eval_gan.py
├─ results/
│  ├─ generator.keras
│  └─ norm_params.npz
└─ figures/
   ├─ hist_overlay.png
   ├─ cdf_overlay.png
   └─ qq_plot.png
```

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

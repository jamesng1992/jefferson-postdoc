# One-Command Snippets

## TD3
```
python problem1_td3_pendulum/train_td3.py --total-steps 150000 --seed 0
python problem1_td3_pendulum/eval_td3.py --checkpoint problem1_td3_pendulum/results/td3_actor.keras
```

## GAN
```
python problem2_gan_age/train_gan.py --epochs 2000 --batch-size 256 --latent-dim 64
python problem2_gan_age/eval_gan.py --num-samples 100000
```

## Create/Push GitHub Repo
```
git init
git branch -M main
git add .
git commit -m "Initial commit: TD3, GAN, CL docs"
git remote add origin https://github.com/<YOUR_USERNAME>/jefferson-postdoc.git
git push -u origin main
```

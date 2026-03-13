# CODI interpretability

Based on
* Original paper [CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://arxiv.org/abs/2502.21074)

Used added additional experiments for 
https://www.lesswrong.com/editPost?postId=tLWrk9H2pmx2rJ7hC&key=369426330689dd659ac783dbba3196

## Setup

```
codi-env.yml
```

Set required environment variables in a `.env` file.

## Train a CODI model

```bash
python train.py configs/llama1b_gsm8k-aug-nl.yaml
```

This repository supports also a distributed training with torchrun.
```bash
torchrun --nproc_per_node=4 train.py configs/llama1b_gsm8k-aug-nl.yaml
```

## Evaluate a CODI model

```bash
bash scripts/test_llama1b.sh
```

## Reproduce the experiments in the post

Use experiments run ones after 10 those are the ones I made tuned logit lens and use train_tuned_logitlens.py to created tuned logit lens

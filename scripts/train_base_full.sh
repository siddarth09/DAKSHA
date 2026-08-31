#!/usr/bin/env bash
# SmolVLA fine-tune WITHOUT LoRA -- the reference recipe. Run: bash scripts/train_base_full.sh

set -eo pipefail


STEPS="${1:-130000}"
RUN="${2:-cross_full_c25}"
DATA="${3:-$HOME/zero_data/cross_base}"

echo "free space on \$HOME: $(df -h "$HOME" | awk 'NR==2{print $4}')"

# ⚠️ ABSOLUTE PATH, NOT A BARE `lerobot-train`. There are two installs: this venv's 0.5.1 and an
# editable 0.4.2 under ~/.local -> /home/sid/lerobot. A bare command resolves by PATH, so forgetting
# to activate the venv silently trains with 0.4.2 -- which has no SmolVLA PEFT support and a torch
# built for the wrong GPU arch. Hard-code it so the script cannot pick the wrong one.
/home/sid/lerobot_env/bin/lerobot-train \
  --dataset.repo_id=zero/cross \
  --dataset.root="$DATA" \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.resize_imgs_with_padding="[256,256]" \
  --policy.chunk_size=25 \
  --policy.n_action_steps=25 \
  --policy.scheduler_decay_steps="$STEPS" \
  --batch_size=16 \
  --steps="$STEPS" \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=4 \
  --wandb.enable=false \
  --output_dir="$HOME/zero_runs/$RUN"

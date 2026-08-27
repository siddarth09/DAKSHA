#!/usr/bin/env bash
# SmolVLA fine-tune WITHOUT LoRA -- the reference recipe. Run: bash scripts/train_base_full.sh
#
# WHY NO PEFT HERE. SmolVLAConfig already defaults to freeze_vision_encoder=True and
# train_expert_only=True, i.e. the authors fine-tune the ~100M-parameter action expert with the
# VLM frozen. That is parameter-efficient in the way that matters and is what smolvla_base was
# designed for. The r=32 LoRA run trained 2.9M params -- 34x fewer -- and additionally damped the
# adapter by lora_alpha/r = 8/32 = 0.25. It reached 31-42 mm position error UNIFORMLY across all
# five task phases on its own training data, against 17 mm of jaw clearance on the can, and in a
# closed-loop rollout it hovered at the start pose and never closed the gripper.
# Measured: 100M trainable fits at batch 16 in 5.76 GiB of 7.53 GiB.
#
# TWO FIXES CARRIED OVER FROM THE LoRA RUN:
#   scheduler_decay_steps=steps  -- the previous run set steps=50000 while decay_steps stayed at
#                                   30000, so the last 20k steps (40% of 6.5 hours) ran at the
#                                   2.5e-6 floor. Auto-scaling only triggers when steps < decay.
#   chunk_size=25                -- 50 steps at 10 fps is 5 SECONDS of open-loop action, and
#                                   measured chunk error grew 31 mm (k=0) -> 47 mm (k=49). The
#                                   default assumes ~30 fps data.
#
# DISK: without PEFT each checkpoint is the full ~1.8 GB model, not a 12 MB adapter.
# save_freq=5000 over 30000 steps is 6 checkpoints, ~11 GB. Check you have room.
set -eo pipefail

EPISODES="$(python3 -c "print('['+','.join(str(i) for i in range(60) if i not in {18,31,33,34,35,55})+']')")"
STEPS="${1:-30000}"
RUN="${2:-base_full_c25}"

echo "free space on \$HOME: $(df -h "$HOME" | awk 'NR==2{print $4}')"

lerobot-train \
  --dataset.repo_id=zero/base \
  --dataset.root="$HOME/zero_data/zero_base" \
  --dataset.episodes="$EPISODES" \
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

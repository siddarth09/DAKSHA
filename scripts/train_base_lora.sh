#!/usr/bin/env bash
# SmolVLA + LoRA on the ZERO baseline view. Run this in a terminal:
#     bash scripts/train_base_lora.sh
#
# ---------------------------------------------------------------------------------------------
# WHAT PEFT IS DOING HERE
# ---------------------------------------------------------------------------------------------
# smolvla_base is 451M parameters. Full fine-tuning on 54 demos would overfit and needs optimiser
# state for every weight. LoRA instead FREEZES all 451M and injects a low-rank update into chosen
# layers: for a frozen weight W, it learns A (r x in) and B (out x r) and uses W + BA. With r=32
# that is ~0.65% of the model trainable (2.93M params), so the pretrained visual-language
# representation is preserved and only the task-specific mapping is learned.
#
# --peft.method_type=LORA        the adapter family (lerobot also exposes MISS etc.)
# --peft.r=32                    rank of the update. Higher = more capacity + more VRAM.
#                                r=16 gave only 743K trainable and grad-norm 0.47 -- too rigid to
#                                express a new action space. r=32 -> 2.93M, grad-norm ~7.
# --peft.target_modules          REGEX of layers that get an adapter. Only the action expert's
#                                query/value projections: that is where the policy reasons about
#                                what to do, and adapting q/v is the standard minimal choice.
#                                The vision tower and the language model stay untouched.
# --peft.full_training_modules   layers trained FULLY (peft's `modules_to_save`), not adapted.
#                                The state/action projections must be here, not in LoRA: they are
#                                the input/output boundary of a 20-dim EEF action space that
#                                smolvla_base never saw, so they need to move freely. A low-rank
#                                nudge to a projection whose meaning changed entirely is not
#                                enough -- that is exactly what the r=16 grad-norm was telling us.
#                                NOTE: a module may not be in both target_modules and this list;
#                                that is why target_modules is narrowed to just q/v above.
#
# ---------------------------------------------------------------------------------------------
# OTHER CHOICES
# ---------------------------------------------------------------------------------------------
# resize_imgs_with_padding=256   the cameras record 224x224; SmolVLA's 512 default UPSCALES for
#                                nothing. 256 is 3.6x faster per step (0.54s vs 1.96s at bs=16).
# episodes                       the 54 complete demos. 18/31/34 never grasp, 33/35/55 grasp but
#                                never hand over, so they would teach a truncated task.
# dataset.root=zero_base         the no-force view. lerobot types EVERY observation.* key as a
#                                policy input, so the view IS the input contract (see
#                                scripts/make_train_view.py). zero_vlfa adds force for the VLFA arm.
set -euo pipefail

EPISODES="$(python3 -c "print('['+','.join(str(i) for i in range(60) if i not in {18,31,33,34,35,55})+']')")"
RUN="${1:-base_lora_r32}"

# ⚠️ ABSOLUTE PATH, NOT A BARE `lerobot-train`. There are two installs: this venv's 0.5.1 and an
# editable 0.4.2 under ~/.local -> /home/sid/lerobot. A bare command resolves by PATH, so forgetting
# to activate the venv silently trains with 0.4.2 -- which has no SmolVLA PEFT support and a torch
# built for the wrong GPU arch. Hard-code it so the script cannot pick the wrong one.
/home/sid/lerobot_env/bin/lerobot-train \
  --dataset.repo_id=zero/base \
  --dataset.root="$HOME/zero_data/zero_base" \
  --dataset.episodes="$EPISODES" \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.resize_imgs_with_padding="[256,256]" \
  --peft.method_type=LORA \
  --peft.r=32 \
  --peft.target_modules='model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj' \
  --peft.full_training_modules="['state_proj','action_in_proj','action_out_proj','action_time_mlp_in','action_time_mlp_out']" \
  --batch_size=16 \
  --steps=50000 \
  --save_freq=2500 \
  --log_freq=100 \
  --num_workers=4 \
  --wandb.enable=true \
  --output_dir="$HOME/zero_runs/$RUN"

# ZERO — research notes

Papers bearing on project ZERO: **an EEF-space bimanual policy taught on two Seeed reBot arms and deployed on a simulated Unitree G1**.

## Why these notes exist

Not a bibliography. Each note ends with a **"Why it matters for ZERO"** section that either changes the plan or explicitly doesn't. A paper that gets read and produces no decision should say so — that is useful information and stops it being re-read.

Two rules, both learned from the previous project:

1. **Mark what has actually been read.** Every note carries a `Status:` line — `abstract read` vs `full paper read`. Notes written from an abstract get a ⚠️ on anything unverified. It is very easy to end up citing a number nobody checked.
2. **Separate the mechanism from the optimisation.** Our transfer mechanism is the explicit shared EEF representation plus per-robot IK. Most cross-embodiment papers propose something else — latent actions, shared dynamics, action priors. Those are usually *additions* to a mechanism, not replacements for one. Note which, every time.

## Index

| Paper | Date | Status | Bearing on the plan |
|---|---|---|---|
| [Learning Action Priors for Cross-embodiment Robot Manipulation](2606.26095-action-priors.md) | 2026-06 | abstract read | Stage-1 pretrains a flow-matching action module on **actions only** — usable before we have good demos, and aimed squarely at data scarcity |

## Queue — surfaced, not yet read

Recorded with links only. **No claims here have been verified; do not cite these lines.**

### Our mechanism is the field standard — read these before claiming novelty

- **Open X-Embodiment / RT-X** — [arXiv:2310.08864](https://arxiv.org/html/2310.08864v4). 60 datasets, 21 institutions. Maps actions from robots with "widely divergent actuation" onto **normalised 7-D end-effector actions: 3 translation + 3 rotation + 1 gripper scalar**. That is our design, per arm. ZERO's shared-EEF space is the canonical approach, not an invention — which is good news for validation and means we should not oversell it. Our one deviation is rot6d instead of discretised rotation (continuity, Zhou et al.).
- **KITE: Decoupling Kinematics and Interaction for Zero-Shot Cross-Embodiment Manipulation** — [arXiv:2606.22113](https://arxiv.org/pdf/2606.22113). The kinematics/interaction split sounds very close to our shared-EEF + per-robot-IK decomposition.
- **AnyBody: A Benchmark Suite for Cross-Embodiment Manipulation** — [arXiv:2505.14986](https://arxiv.org/pdf/2505.14986). A *benchmark*. Potentially removes the need to invent our own transfer metric.

### The "robot's own arms in frame" problem — named, but the published fix does NOT fit us

This is ZERO's Phase-2 risk. There is a literature for it, and reading it changed the plan twice.

- ⚠️ **Mirage: Cross-Embodiment Zero-Shot Policy Transfer with Cross-Painting** — [arXiv:2402.19249](https://arxiv.org/abs/2402.19249). Masks the *target* robot out of the frame, inpaints the hole, then renders the *source* robot in its place, so the policy believes it is still driving the arm it trained on. Zero-shot across different arms and grippers.
  **DEMOTED 2026-08-13.** Mirage is manipulator↔manipulator, and it assumes source and target are interchangeable *at the same base pose* — that is what makes painting source pixels over target pixels geometrically coherent. reBot→G1 violates it: the reBot arms are bolted to a table, the G1's hang off a torso standing beside it, and the frame gains a torso and head with no source-side counterpart. Worse, with the G1's own `head_camera` (which moves with the torso) there is no reBot-equivalent viewpoint at all, so cross-painting is not merely lossy but undefined. Keep it in mind only for a fixed third-person view.
- **RoVi-Aug: Robot and Viewpoint Augmentation** — [rovi-aug.github.io](https://rovi-aug.github.io/). Robot-to-robot diffusion + video inpainting; tolerates camera-pose change and permits finetuning, i.e. Mirage without the constraints. Same base-pose assumption though.
- **Cloak: Zero-Shot Cross-Embodiment Manipulation by Masking the End-Effector from the VLA** — [arXiv:2606.22836](https://arxiv.org/pdf/2606.22836). Masks only the end-effector. The most likely of the three to survive a manipulator→humanoid jump, precisely because it removes the smallest region.

**Where that leaves us: WRIST CAMERAS.** A camera at each wrist looking at the object puts the base, torso and head out of frame entirely, so the manipulator-vs-humanoid difference stops being visible. The residual gap is parallel jaw vs 3-finger, which is far smaller than a whole body. Fixed third-person becomes a secondary view / ablation rather than the primary.

### Latent-action alternatives to our explicit space

- **Latent Action Diffusion for Cross-Embodiment Manipulation** — [arXiv:2506.14608](https://arxiv.org/html/2506.14608v4). Unifies diverse EEF action spaces into one contrastively-aligned latent space, and factorises the policy into an embodiment-agnostic part plus **embodiment-specific action decoders**. The latent counterpart to what we are doing explicitly with IK.
- **DyPES-VLA: Learning Shared Dynamics Priors and Embodiment-Specific Control** — [arXiv:2608.06374](https://arxiv.org/abs/2608.06374).
- **LAP: Language-Action Pre-Training Enables Zero-shot Cross-Embodiment Transfer** — [arXiv:2602.10556](https://arxiv.org/pdf/2602.10556). Claims *zero-shot*; a baseline to compare our transfer ratio against.

Also worth pulling in when the relevant phase arrives:

- Teleoperation rigs, for Phase 1 — [OPEN TEACH](https://arxiv.org/pdf/2403.07870), [Bunny-VisionPro](https://arxiv.org/pdf/2407.03162), [AnyTeleop](https://arxiv.org/pdf/2307.04577), [GELLO](https://wuphilipp.github.io/gello_site/)
- 6D rotation representation, since our action uses rot6d — Zhou et al. 2019, *On the Continuity of Rotation Representations in Neural Networks*

## Note template

```markdown
# <Title>

- **arXiv:** [id](url) (date)
- **Authors:** ...
- **Status:** abstract read | full paper read

## Problem it attacks
## Method
## Claimed results          (⚠️ mark anything unverified)
## Why it matters for ZERO  (or explicitly: why it doesn't)
## Open questions
```

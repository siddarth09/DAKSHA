# DAKSHA — bimanual loco-manipulation 

Extends the single-arm VLA work (PRANA) to dual-arm coordinated manipulation on the
full Unitree G1. *Dakṣa* (Sanskrit) = "skilled / dexterous." Earth gravity (-9.81).

## Layout
```
daksha/
├── scripts/
│   ├── gen_robocasa_kitchen.py   # generate a REAL RoboCasa kitchen (+objects), render / view
│   └── browse_robocasa.py        # contact sheets of all layouts & styles
├── scenes/
│   ├── robocasa_kitchen.xml      # last generated RoboCasa scene
│   ├── robocasa_kitchen_*.png    # last renders (overview + counter)
│   ├── kitchen_env.xml           # hand-authored primitive kitchen (no robocasa needed)
│   └── home_env.xml              # minimal room + table + two-hand box
├── assets/objects/robosuite/     # robosuite meshes (bottle/can/milk) for primitive scene
└── robots/                       # drop g1_with_hands.xml here (from src/bheema/unitree_g1/)
```

## Quick start — generate a RoboCasa kitchen

Always activate the venv first (robocasa lives there):
```bash
source ~/bheema_rl_env/bin/activate
cd ~/projects25/src/daksha/scripts
```

Generate layout 2 / style 2 with 3 random RoboCasa objects on the counter, and open it:
```bash
python gen_robocasa_kitchen.py --layout 2 --style 2 --objects 3 --view
```

Flags:
| flag | meaning |
|------|---------|
| `--layout N`  | kitchen layout, 1–60 |
| `--style N`   | kitchen style, 1–60 |
| `--objects N` | number of random RoboCasa objects placed on the largest clear counter strip |
| `--seed N`    | reshuffles which objects are sampled and where they land |
| `--view`      | open the interactive MuJoCo viewer (see note below) |
| `--no-render` | skip writing the PNGs |
| `--out NAME`  | output basename (default `robocasa_kitchen`) |

Outputs: `scenes/<out>.xml`, `scenes/<out>_overview.png`, `scenes/<out>_counter.png`.

##  Why the kitchen looks all RED if you open the raw XML

RoboCasa gives every fixture two geoms: a **collision** geom (dark red, `rgba="0.5 0 0 0.5"`,
MuJoCo geom **group 0**) and a **visual/textured** geom (**group 1**). The plain
`python -m mujoco.viewer --mjcf=scenes/robocasa_kitchen.xml` shows group 0 by default, so the
whole kitchen renders red, and the enclosing walls stay opaque.

**Do NOT open the raw XML in the plain viewer.** Instead use `--view`, which hides the
collision group and makes the enclosing walls transparent — matching the PNG renders:
```bash
python gen_robocasa_kitchen.py --layout 2 --style 2 --objects 3 --view
```
(If you ever do open it in the plain viewer, press key **`0`** to toggle off the collision
geom group.)

## Browse layouts & styles before choosing
```bash
python browse_robocasa.py --fixed-style 1 --fixed-layout 1
# -> scenes/browse_layouts.png (all 60 layouts) and scenes/browse_styles.png (all 60 styles)
```

## Objects

Objects are sampled from RoboCasa's library via `sample_kitchen_object(...)` and placed on the
counter using RoboCasa's own reset regions (avoids appliances), then dropped to settle.
Currently only the **lightwheel** registry is used (the pack that's downloaded). To unlock the
full library (objaverse, thousands of objects):
```bash
yes | python ~/projects25/src/robocasa/robocasa/scripts/download_kitchen_assets.py --type objs_objaverse
```
then add `"objaverse"` to `OBJ_REGISTRIES` at the top of `gen_robocasa_kitchen.py`.

## Environment notes
- RoboCasa is installed editable in `~/bheema_rl_env` and hard-pins `mujoco==3.3.1`.
  ⚠️ This downgraded the env from mujoco 3.8 / torch 2.11, which breaks mjlab / mujoco-warp
  (the HANUMAN locomotion stack). If you use those, restore mujoco 3.8 / torch 2.11 (or move
  robocasa to its own venv).
- Downloaded RoboCasa asset packs live under `~/projects25/src/robocasa/robocasa/models/assets/`
  (`textures`, `fixtures`, `objects/lightwheel`).

## Status
Phase 0: real RoboCasa kitchen environment loads/renders without a robot, with library objects
on the counter. Next → copy G1 MJCF into `robots/`, merge into the scene, add the standing policy.
```

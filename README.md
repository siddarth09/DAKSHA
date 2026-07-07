# DAKSHA — bimanual loco-manipulation 

https://github.com/user-attachments/assets/420457ae-cec6-47b6-8f63-33b3e6751fe1



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



## Objects

Objects are sampled from RoboCasa's library via `sample_kitchen_object(...)` and placed on the
counter using RoboCasa's own reset regions (avoids appliances), then dropped to settle.
Currently only the **lightwheel** registry is used (the pack that's downloaded). To unlock the
full library (objaverse, thousands of objects):
```bash
yes | python ~/projects25/src/robocasa/robocasa/scripts/download_kitchen_assets.py --type objs_objaverse
```
then add `"objaverse"` to `OBJ_REGISTRIES` at the top of `gen_robocasa_kitchen.py`.


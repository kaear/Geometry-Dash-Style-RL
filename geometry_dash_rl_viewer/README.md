# CS 175 Geometry Dash RL — first milestone

This is a **minimal original Python training simulator** for testing an RL pipeline, **not** a port or licensed derivative of the JavaScript clone. The proposal references https://github.com/anaygoenka/GeometryDash as a possible visual/gameplay source. Review and attribute any code you subsequently copy from it.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python smoke_test.py
python train_ppo.py
```

## Current behavior
- Auto-run, ground + gravity, single binary jump, three fixed spikes.
- Gymnasium `reset` / `step`, terminal collision and completion, time-limit truncation.
- Sparse or progress-shaped rewards (initial coefficient of 0.01).
- Random/heuristic baseline script and first PPO training script.

## What is intentionally NOT implemented yet
- Procedural map generation, level difficulty settings, curriculum scheduling, validation/test seeds, plotting, rendering, DQN.
- Precision sprite hitboxes and detailed Geometry Dash physics; calibrate or replace physics if you adapt a clone.

## Milestone checklist
1. Run the smoke test and verify the heuristic survives the fixed level more often than random actions.
2. Run `train_ppo.py` and confirm saved checkpoint; evaluate it with a separate deterministic evaluation script before drawing conclusions.
3. Commit baseline to GitHub; then implement reproducible seeded procedural levels with feasibility checks.
4. Add direct-vs-curriculum and sparse-vs-shaped experimental configuration.

**Limitations:** Code passed Python syntax compilation at packaging time, but Gymnasium / Stable-Baselines3 were unavailable in that environment, so full execution was not tested. Install dependencies and run `smoke_test.py` locally. Observation features and physics are starting assumptions to refine, not final experimental specifications.

## Visual viewer (new)

Install the extra Pygame dependency with `python -m pip install -r requirements.txt`.

```bash
python view_game.py --mode manual
python view_game.py --mode random
python view_game.py --mode heuristic
python view_game.py --mode ppo --model models/ppo_fixed_level.zip
```

In manual mode hold **Space** or **Up** to jump when grounded. In every mode press **R** to restart, **P** to pause and **Esc** to quit. Add `--fps 20` to slow simulation down. The viewer deliberately uses the exact `GeometryDashEnv` state and `step()` used for training; **no physics, obstacles, rewards, or saved models are modified**. This viewer draws basic shapes rather than the original Geometry Dash art. Your previously trained checkpoint is not bundled: copy `models/ppo_fixed_level.zip` from your existing project folder to the new folder (or run `python train_ppo.py`).

The starter fixed level is deterministic, so repeated deterministic PPO episodes will often produce the same result. Procedural level generation is a later milestone.

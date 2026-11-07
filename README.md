# BlinkMimic

**Whole-body motion tracking for the Unitree G1 in about 3 minutes on one RTX 4090.**

BlinkMimic trains one policy for all 40 LAFAN1 clips (2.45 h of motion) to an evaluation normalized return of 0.82
in **2.9 min of training on one RTX 4090** (mean of 3 seeds), and to 0.82 with 0 of 40 clips failed for 3
checkpoints in a row in **3.4 min**.

Like SuperMimic and InstantTrack, BlinkMimic learns one differentiable world model and trains the policy by backpropagation through
imagined rollouts, on [mjlab](https://github.com/mujocolab/mjlab) / MuJoCo Warp. BlinkMimic changes how the policy
is optimized and where the simulator spends its resets ([details](#what-blinkmimic-changes)).

![Equal training time (3.1 min). SuperMimic and InstantTrack v1.2 fall on fallAndGetUp1_subject1; BlinkMimic tracks it.](docs/static/hero.gif)

*Each policy after 3.1 min of training (seed 0), on `fallAndGetUp1_subject1`: the first 8 s, then 30–42 s,
simulated in MuJoCo Warp under evaluation conditions. Blue ghost: reference motion. A faded panel has fallen.*

## Results

The measurements below are from the original release. They have not been remeasured with this branch's
relative reset rewards and random robot yaw/XY translation. See [the current workflow](#current-workflow-training-to-rendering)
for this branch's commands and evaluation settings.

All runs on one RTX 4090 (driver 580, CUDA 12.8, torch 2.9.0), the GPU otherwise idle. 40 LAFAN1 clips; every
checkpoint (one per 100 iterations) evaluated on all 40 clips from their first frame with 4 replicas; return and
failed clips are averaged over the replicas. Minutes are training wall-clock, including compilation, CUDA-graph capture, probe sweeps and
checkpointing, excluding process start-up and data loading.

![Training time to return ≥ 0.82 (left) and to the strict target (right)](docs/static/time_to_target.svg)

### Time to target

| | seeds | first return ≥ 0.82 (min) | ≥ 0.82 and 0 failed, 3 checkpoints in a row (min) | iterations to first ≥ 0.82 | ms / iteration |
|---|---|---|---|---|---|
| SuperMimic | (1) | ≈ 41 (estimated) | ≈ 86 (estimated) | 7898 | 310 |
| SuperMimic + EMA | (3) | ≈ 30 (estimated) | ≈ 32 (estimated) | 5749 / 6000 / 5856 | 310 |
| InstantTrack v1.2 | 3 | 5.59 / 6.31 / 5.78 (**5.89**) | 7.29 / 8.52 / 6.28 (**7.36**) | 2500 / 2800 / 2500 | 141 |
| **BlinkMimic** | 3 | 2.89 / 3.28 / 2.64 (**2.94**) | 3.32 / 3.28 / 3.53 (**3.38**) | 1300 / 1500 / 1200 | 133 |

"First return ≥ 0.82" is SuperMimic's metric, the strict column InstantTrack's. ms / iteration = training
wall-clock / iterations.
SuperMimic times are estimated: the iterations at which InstantTrack's SuperMimic runs reach the target (strict:
16665; 5749 / 6000 / 6971) × 310 ms per iteration, measured for SuperMimic on our RTX 4090.

BlinkMimic is 6% cheaper per iteration than InstantTrack v1.2 and needs about half the iterations.

With 6 checkpoints in a row instead of 3, BlinkMimic reaches the target in 2 of 3 seeds (5.36, 3.53 min) and
InstantTrack v1.2 in 1 of 3 within 4500 iterations.

### Equal training time

Mean return / failed clips (of 40) at the last checkpoint within the budget. These SuperMimic runs took 331–348 ms
per iteration (310 ms above).

| training time | SuperMimic (1 seed) | SuperMimic + EMA (3 seeds) | InstantTrack v1.2 (3 seeds) | **BlinkMimic** (3 seeds) |
|---|---|---|---|---|
| 3.1 min | 0.179 / 29.2 | 0.002 / 40 | 0.639 / 12.3 | **0.821 / 1.5** |
| 4.5 min | 0.367 / 16.8 | 0.231 / 28.9 | 0.767 / 5.0 | **0.851 / 0** |
| 7.8 min | 0.495 / 12.2 | 0.480 / 14.0 | 0.846 / 0.1 | **0.861 / 0** |

![Return and failed clips over training time](docs/static/curves.svg)

### Tracking quality at equal training time

Mean of 3 seeds, one deterministic rollout per clip (`tools/eval_metrics.py`), at iteration 2000 (4.3–4.6 min) and at
the last checkpoint within 7.8 min (v1.2: iteration 3400; BlinkMimic: 3500).

| training time | method | return ↑ | failed clips ↓ | body pos (cm) ↓ | body ori (°) ↓ | joints (°) ↓ | foot slip (cm/s) ↓ | contact agreement ↑ | penetration (mm) ↓ |
|---|---|---|---|---|---|---|---|---|---|
| 4.4 min | InstantTrack v1.2 | 0.765 | 5.3 | 5.15 | 10.3 | 4.85 | 7.75 | 0.746 | 0.079 |
| 4.4 min | **BlinkMimic** | **0.851** | **0.3** | **3.74** | **7.37** | **3.99** | **6.87** | **0.757** | **0.061** |
| 7.8 min | InstantTrack v1.2 | 0.847 | 0 | 4.07 | 8.29 | 4.77 | 7.18 | **0.775** | 0.061 |
| 7.8 min | **BlinkMimic** | **0.862** | 0 | **3.44** | **7.02** | **4.19** | **6.66** | 0.765 | **0.058** |

Foot slip of the reference motion itself: 10.2 cm/s.

## What BlinkMimic changes

| | SuperMimic ([supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)) | InstantTrack v1.2 | BlinkMimic | flag |
|---|---|---|---|---|
| policy EMA, resets before recent failure points, contact reward, world model 3 × 1024, learning rate 4e-4 | – | ✓ | ✓ | `PRESET[instanttrack]` |
| no host synchronization in the update loop, fused Adam, fused rotations, more of the step compiled | – | ✓ | ✓ | |
| Muon on the hidden weight matrices of policy and world model (Jordan et al. 2024; Liu et al. 2025) | – | ✓ | ✓ | `--pi_opt muon --wm_opt muon` |
| world-model and policy updates as CUDA graphs | – | ✓ | ✓ | `--cuda_graph` |
| goal encoding batched before each imagined rollout (same result, fewer kernels) | – | ✓ | – | |
| action built on the reference pose | – | ✓ | ✓ | `--pi_ref_skip` |
| policy encoder / decoder | 2048-1024-512-256 | 1024-512-256 | 2048-1024-512-256 | `--enc_hidden --dec_hidden` |
| world-model-only warm-up | 500 iterations | 500 iterations | none | `--wm_only_iters 0` |
| semi-implicit world model: predicts velocity increments and integrates positions | – | – | ✓ | `--wm_structured` |
| left-right mirror augmentation of half of each world-model batch | – | – | ✓ | `--wm_mirror 0.5` |
| reset-statistics fixes (duplicate bins; per-clip contact floor of the contact reward) | – | – | ✓ | `--bin_fix --ref_contact_fix` |
| evaluation-guided resets: from iteration 1000, every 125 iterations the deployed policy is swept over all clips under evaluation conditions; 15% of the resets go to the 10 s before the places where it fails | – | – | ✓ | `--eprobe_*` |
| imagined-rollout batch (policy and critic) | 4096 | 4096 | 2048, policy learning rate 4×, decayed to 1× over 300 iterations once the training return reaches 0.75 (by iteration 1400) | `--pi_batch --pi_lr_mult --lr_decay_*` |
| collection step, episode resets and probe sweep as CUDA graphs; sweep on a side stream; checkpoints written by a background thread | – | – | ✓ | `--graph_early --collect_graph --reset_masked --eprobe_graph --eprobe_async --ckpt_light` |

The released recipe is `PRESET[main]` in [`scripts/presets.sh`](scripts/presets.sh). `python -m blinkmimic.train`
without flags uses the SuperMimic optimizer settings. On this branch, the simulator and MDP also use the
relative reset rewards and robot yaw randomization described below.

## Setup

Linux x86_64, an NVIDIA GPU with a driver for CUDA 12.8, [uv](https://docs.astral.sh/uv/), `git`, `curl`, `python3`
(and `ffmpeg` for videos).

```bash
git clone https://github.com/fuyichen0725/blinkmimic.git
cd blinkmimic
uv sync                          # .venv with mjlab, MuJoCo Warp, Warp, torch 2.9.0 (CUDA 12.8)
tools/download_data.sh           # 40 G1-retargeted LAFAN1 clips + G1 description -> data/
uv run python tools/test_env.py  # sanity check: FK vs simulator, state round-trip, reference playback
```

## Current workflow: training to rendering

Run these commands from the repository root after completing Setup. Activate the environment so the same
Python installation is used for training, evaluation and rendering:

```bash
source .venv/bin/activate
```

### Full BONES-SEED on a single 32 GiB GPU

This branch stores every reference frame on the GPU as root XYZ in FP32 plus a packed FP16 state
(quaternion, world linear/angular velocity, joint position/velocity). G1 needs **148 bytes/frame**, plus
small clip/bin metadata and two bytes/frame when calibrated foot contacts are enabled. Queries decode
to FP32 and normalize the quaternion before computing matrices, FK, commands and rewards. The storage
change introduces FP16 quantization in those channels; root XYZ stays FP32. There is no full-library
body cache or per-frame clip/bin index table. Reference FK runs only for requested batches. Initialization
uses two passes over the files and writes each clip directly into the final allocation, with host memory
bounded by clip size rather than retaining all decoded motions. Reference-contact preprocessing streams
bounded FK batches and keeps only foot scalars on the CPU, then a boolean contact cache on the GPU.
Code that reads references uses `motion.ref_state(frames)`, `motion.body_state(frames)`,
`motion.clip_of(frames)` and `motion.bin_of(frames)` rather than full-library derived tensors.

At 43 million frames, frame storage is approximately 6.36 GB, versus approximately 58 GB in the former
FP32/body-cache layout. This is motion storage, not total training memory. Do not estimate GPU use from
compressed NPZ disk size: source frame rates, frame counts and compression ratios differ.

```bash
PYTHON=python bash scripts/train_bones32.sh logs/bones_s0 \
  --motion_dir /path/to/bones --motion_quat_order wxyz --seed 0
```

The entry point loads **all matching files**, including subdirectories. It uses 2048 simulator worlds,
500 replay steps, world-model/policy batches of 1024, a 200-step training episode limit, and disables
eprobe. Extra flags override these defaults. Actor commands, reset-relative diffs and reward coefficients
retain their definitions. Reward settings remain in `reward_cfg.py`. Without this entry point, eprobe
now caps simultaneous probe worlds at `--eprobe_envs 4096`, sampling tiles from the full candidate set
per sweep. Use `--eprobe_share 0 --eprobe_async 0` to disable it entirely.
Terminal-only reset bins reuse the last valid transition; imagined references clamp to the clip end and
returns/bootstrap stop there, so short clips and tail frames cannot index into the next motion.

Startup prints actual motion/replay storage. TensorBoard `memory/*` records PyTorch allocated, reserved
and peak memory, plus sampled whole-device use (which also includes MuJoCo/Warp). Training failures are
propagated through `scripts/train.sh` rather than hidden by its `tee` pipeline.

Validate the **entire dataset on the target GPU** before a long run:

```bash
python tools/check_bones_memory.py \
  --motion_dir /path/to/bones --motion_quat_order wxyz \
  --gpu 0 --log_dir logs/bones32_check --iterations 250
```

Use an otherwise idle GPU. The check loads the full library and runs real training through graph captures,
replay fill and episode resets; `nvidia-smi` samples whole-device memory, while TensorBoard records the
PyTorch allocator peaks. It writes `memory_check.json`, checks that at least 250 iterations were logged,
and exits nonzero on training failure or observed use reaching the 32 GiB budget/device capacity.
Whole-device peaks are sampled, so this is a runtime smoke check rather than proof for every future policy
state or long-run allocation. **This workspace has no CUDA GPU or full BONES-SEED copy: the target-card
check has not been run here.** CPU regression tests verify storage size, streaming/FK behavior, precision,
compiled gradients and reward consistency. Full-library evaluation should also use bounded rollout batches;
`evaluateran --episodes 20` bounds its simulator size.

### 1. Train with relative reset rewards and random robot yaw/position

```bash
PYTHON=python bash scripts/train.sh logs/run_no_rel_s0 \
  --seed 0 --iterations 3500 --ckpt_every 100 --yaw_degrees 90 --xy_range 0 0.1
```

This uses `PRESET[main]`, writes logs to `logs/run_no_rel_s0`, and saves policy checkpoints such as
`logs/run_no_rel_s0/ckpt_003500.pt`. On a shared GPU, append `--eprobe_async 0` as noted below.

Use `--max_episode_steps 200` to cap training episodes at 4 seconds (50 Hz). The default is 500 steps
(10 seconds); motion endings or failures can end an episode earlier. This setting only limits training
collection: evaluation continues to the motion's end, and `--wm_window` / `--horizon` retain their values.
Each new episode samples a fresh reset pose and reference frame. Compare long evaluation rollouts when
checking drift, since shorter training episodes also shorten the time available for position error to accumulate.

`--motion_dir` recursively loads CSV and NPZ files, including subdirectories. NPZ motions use the supplied
packed format: `qpos` is `(T, 7+J)` with root XYZ, root quaternion and joint angles; `fps` is the source
frame rate. Joint columns must follow the robot's `joint_names` order (29 joints for G1).
Use `--motion_quat_order wxyz` (default) or `--motion_quat_order xyzw` to declare the NPZ quaternion order;
this applies to all NPZ files in that motion set. CSV keeps its existing robot-specific interpretation
(G1 LAFAN1: xyzw, 30 FPS), including when CSV and NPZ files share a directory. All motions use the same
50 Hz resampling, finite-difference velocities, FK, commands and rewards after loading.
Clip names include their relative subdirectory, for example `jumps/jump_and_land_heavy_001__A001`.

```bash
PYTHON=python bash scripts/train.sh logs/run_npz_s0 \
  --motion_dir /path/to/motions --motion_quat_order wxyz \
  --seed 0 --yaw_degrees 90 --xy_range 0 0.1 --pi_ref_skip \
  --iterations 4000 --ckpt_every 1000 --max_episode_steps 200
```

Both evaluators accept the same `--motion_dir` and `--motion_quat_order` options. Pass the same values
when evaluating this motion set. `tools/eval_batch.py` reads them from the run's `args.json`.
ONNX export/playback also accept `--motion_quat_order` with a single NPZ supplied through `--motion`.

Training samples a motion time bin using return/failure statistics, then a frame inside that bin. Each reset
adds an independent, continuous uniform robot yaw offset in **[-90°, +90°]**, on top of the original reset noise.
The world linear and angular velocities rotate with that yaw. Use `--yaw_degrees 90` to set this half-range,
`--yaw_degrees 30` for [-30°, +30°], or `--yaw_degrees 0` to disable the extra yaw rotation. Omitting the flag
preserves `DomainRandCfg.reset_yaw` in [`src/blinkmimic/dr.py`](src/blinkmimic/dr.py), whose default is pi/2
radians (90°). The existing small `reset_rot` noise remains enabled even when this extra yaw rotation is disabled.

Training also adds independent world-X/Y translations, each uniform in **[0, 0.1] meters**, matching the
reference initialization. This is added to the existing `reset_pos` noise; it does not add a new Z offset.
Use `--xy_range LOW HIGH` to override `DomainRandCfg.reset_xy`; for example, `--xy_range -1 1` samples each
horizontal component in [-1, 1] meters. `--xy_range 0 0` disables only this additional translation, while
preserving the existing small reset noise. The saved robot reset position includes both noise and translation.

Actor commands still come from the original `motion.goal(f, K)`. The history contains only proprioception:
`[q-q0-offset, qd, local angular velocity, projected gravity, previous action]`. The old 6D
`robot_R.T @ motion_R` observation has been removed from direct and sensed observations, noise, scale and delay
layouts. Each history frame has `3J+6` values: G1 uses 93 values instead of 99 (930 instead of 990 for 10 frames).
Future motion commands retain their reference-relative orientation features. Root position and orientation
rewards compare robot and expert changes in their respective initial root frames: the robot uses its pose after reset randomization, and the expert uses the sampled frame
before randomization. The reset context is also retained in replay and imagined rollouts.

The critic uses those same four base diffs (position, quaternion rotation, local linear velocity and local
angular velocity; 12 components total). They replace its previous base height, tilt, raw local velocities
and absolute root pose errors. Both the actor's value bootstrap and critic targets receive each replay
sample's reset context. Motion commands and world-model features retain their layouts. Removing history
`rel` changes the actor input/normalizer dimensions too: use a new training run. Checkpoints and ONNX models
trained with the former 99-value G1 observation cannot be loaded by the 93-value policy. ONNX export and playback
use the new proprioception layout; the exported next-reference quaternion remains a reference output only.

Body rewards and critic features share four link diffs, each with shape `(batch, links, 3)`. The reference
uses its current base frame; the robot uses a hybrid frame with its own current yaw and the reference's
current roll/pitch. Positions subtract each own base position. Linear/angular velocities are rotated into
those frames without subtracting base velocity. Rotation is a shortest-path 3D rotation vector, in radians.
At a vertical forward axis, where yaw is undefined, hybrid construction uses a zero heading.

### Reward configuration

All reward settings live in [`src/blinkmimic/reward_cfg.py`](src/blinkmimic/reward_cfg.py). The active
`REWARD_CFG` enables the preceding repository training recipe (the formerly commented configuration,
calibrated for G1's 14 tracked links). The supplied reference configuration is now commented out; keep only
one assignment active. There is no reward preset flag or automatic checkpoint override. Edit the file and restart training/evaluation to use new values.

Sigma is used directly in `weight * exp(-error / sigma)`:

- Base position/rotation/velocity and joint velocity errors sum their vector/joint components.
- Link position and 3D rotvec errors sum every link/component.
- Link velocity errors average every link/component, including XYZ.

The diagnostic `e_body_*` values remain mean per-link squared vector norms for comparable error metrics;
`mdp.reward()` applies the link-count / XYZ factors explicitly to obtain the configured sum/mean rewards.
No sigma-to-std conversion is needed in the configuration file.

The active positive tracking weights sum to 6.0, giving `MAX_STEP_REWARD = 6.0 * CONTROL_DT` (0.12 at 50 Hz).
This is derived from the active weights and used for training/evaluation normalization and reset sampling.
The active action-rate weight is -0.1, joint-limit weight -2.0 and training-only contact weight 0.25;
action-magnitude and joint-velocity reward weights are zero. The contact bonus is excluded from that tracking
upper bound, so the imagined reward with contact can exceed a normalized value of 1.
The optional training-only contact weight is also configured here, rather than through a training flag.
Optimizer/system presets do not override reward weights.

Training and evaluation always read the current file, including when evaluating an older checkpoint.
Changing it also changes the reward used to score older policies. Use a fresh training directory after a
reward change, and keep the same file version when comparing runs. Renderers only display recorded states.
Actor observations and commands, base/body diff definitions and critic layout are unaffected by changing
reward coefficients. The critic receives all four body diffs and all four base diffs; old critic layouts
still require a new training run.

### TensorBoard: rewards, tracking errors and world-model accuracy

Training automatically writes events to `<log_dir>/tensorboard` at `--log_every` (default 50 iterations).
The training command stays the same. After installing/syncing the project dependencies, open the dashboard:

```bash
tensorboard --logdir logs --port 6006
```

- `train/*`: the existing CSV metrics, including simulator/imagined returns, world-model loss and throughput.
- `reward_sim/*`, `reward_model/*`: all nine tracking contributions, action-rate/action-magnitude/joint-limit
  penalties, contact bonus, tracking subtotal and total. These are **weighted rewards per control step,
  including CONTROL_DT**, before return normalization. Zero-weight terms remain visible.
- `score_sim/*`, `score_model/*`: the unweighted tracking exponentials (0 to 1), independent of reward weights.
  `saturated_*/*` is the fraction with a score below 0.05; a large fraction identifies a nearly flat reward.
- `diff_sim/*`, `diff_model/*`: RMS base/link position, rotation and velocity errors, plus base XYZ signed biases
  and the 90th percentile of each base diff norm. Base position is in meters in each reset frame; velocities
  use the reward's local frames. `base_angle_deg` converts the base quaternion chord to a physical angle;
  `root_ori_rms`/`base_rotation_p90` retain the actual quaternion-vector diff, not radians. Link rotation is
  a 3D rotvec in radians. Link RMS errors average links; joint-velocity error sums joints before the RMS.
- `motion_sim/base_forward_speed`, `motion_model/base_forward_speed`, `motion_reference/base_forward_speed`:
  compare forward speed in each current base frame when the apparent stride is too small.
- `wm/loss_group/*`: mean normalized squared error within each state group; `wm/loss_contribution/*` also
  applies its dimension fraction, and the six contributions sum to `wm/loss`. Normalization floors and masks
  match training. G1 has 12 base components and 58 joint components, so the aggregate loss alone can hide
  poor base prediction. The model predicts simulator dynamics, rather than directly learning imitation rewards.
- `wm/rmse_1step/*`, `wm/rmse_8step/*`, `wm/rmse_32step/*` (default horizons): physical prediction RMSE per
  component, without training normalization. These distinguish good short predictions from accumulating drift.
  Actual horizon names follow `--wm_window`/`--horizon`; `wm/valid_*step` records the valid sample counts.
  An RMSE with no valid samples is NaN rather than a misleading zero.

Diagnostics sample 128 replay windows and replay the **same recorded actions** through the current world
model without teacher forcing. Simulator/model reward plots use these paired windows, rather than the actor's
separate imagined training rollout or an average over the logging interval. Reset/push crossings are masked;
terminal transitions themselves remain valid. `reward_model/contact` includes the training-only bonus;
`reward_sim/contact` is zero. `wm/reward_gap_without_contact` subtracts that bonus for a fair comparison:
a positive gap means the model scores these actions more favorably than the simulator does.
Sampling restores the training RNG state, does not update normalizers/optimizers, and runs outside CUDA graphs
only at logging intervals. The dashboard also stores the active reward configuration and arguments as text.

For sigma tuning, first inspect `score_sim/base_pos`, `score_sim/base_rotation` and their saturated fractions.
If most samples have already fallen below 0.05, try a broader denominator before raising weights again:
`sigma_base_pos=0.5**2` (previously `0.3**2`) and `sigma_base_rotation=0.6**2` (previously `0.4**2`) are a
starting comparison, keeping weights fixed. At 0.6 m error the position exponential rises from 0.018 to 0.237;
at 60 degrees the rotation exponential rises from 0.0019 to 0.062. This restores gradients farther from the
target but makes close tracking less strict; tighten sigma once tracking improves. If scores are already high
despite unacceptable error, use a smaller sigma instead. For a representative squared error E and desired
score r, the denominator is `sigma = -E / log(r)`; base rotation uses `E = 4*sin(angle/2)**2`.
These are manual tuning suggestions; logging does not change the active reward configuration.

### 2. Evaluate and record multiple random episodes

```bash
python -m blinkmimic.evaluateran \
  --run_dir logs/run_no_rel_s0 \
  --ckpt logs/run_no_rel_s0/ckpt_003500.pt \
  --record logs/run_no_rel_s0/traj_ran.npz \
  --episodes 20 --yaw_degrees 90 --xy_range 0 0.1 --seed 0
```

For each episode, this entry point uniformly samples a motion, uniformly samples a frame before its final
frame, and samples robot yaw and world-X/Y offsets. The last frame is excluded because evaluation needs at least one
transition. Evaluation runs until termination or the motion end, with the return normalized by the number of
transitions remaining after the sampled start. It retains the original actor commands and observation
references, just as training does. Training's adaptive sampling and additional reset noise are not used here;
`--dr_from` separately enables the existing robustness evaluation options.

| Option | Meaning |
|---|---|
| `--episodes 20` | Record 20 independent random starts; the default is 1. Motions may repeat. |
| `--repeats n` | Multiply the episode count by `n`; this does not guarantee coverage of every motion. |
| `--yaw_degrees 90` | Continuous uniform yaw offset in [-90°, +90°]; `0` disables the yaw offset. |
| `--xy_range 0 0.1` | Each X/Y offset is uniform in [0, 0.1] meters; `0 0` disables translation. No extra Z offset. |
| `--seed 0` | Reproduce the same sampled motions, frames and yaw values for the same settings. |
| `--record path.npz` | Save robot trajectories, original references, sampled start frames, yaw/position offsets and the actual robot reset pose. |

The output includes `root_pos_rmse_m`, `root_pos_max_m`, `per_rollout_root_pos_rmse_m` and
`per_rollout_root_pos_final_m`. These measure the **3D root displacement error**, including horizontal drift,
in meters. Statistics include only active episode transitions. `starts` lists each motion, its local start
frame and yaw offset (in radians). Use many episodes when comparing checkpoints; a single random start is
not a full-library evaluation. `starts` also reports each `pos_offset_m` vector.

To isolate the effect of yaw, repeat the same command with `--yaw_degrees 0` and a different recording filename.
Keeping `--seed` and the episode count unchanged preserves the sampled motions and start frames.
Likewise, compare `--xy_range 0 0` against another position range while holding yaw, seed and episode count
fixed to isolate translation. Position sampling has a separate reproducible RNG stream.

### 3. Render the recorded episodes

```bash
MUJOCO_GL=egl python tools/render_ran.py \
  logs/run_no_rel_s0/traj_ran.npz logs/run_no_rel_s0/result_ran.mp4 \
  --robot g1 --seconds 10
```

By default, all recorded episodes are concatenated in recording order, including repeated motions.
`--seconds 10` means **10 seconds per episode**, so 20 episodes produce a 200-second video. Episodes that end
early display their final recorded state for the remaining video time; rendering does not sample new episodes.

The ghost uses the sampled expert root position `p0`, actual robot reset position `robot_p0`, and sampled yaw
rotation `Y`: `p_ghost(f) = robot_p0 + Y @ (p_expert(f) - p0)` and `R_ghost(f) = Y @ R_expert(f)`.
This is a display transform; it does not change actor commands or the evaluation reference motion.
The default ghost follows its full transformed trajectory, so base drift stays visible. The text overlay
reports the root distance in meters; `--clean` and `--white` omit the numerical overlay.

```bash
# Render selected episodes (zero-based indices) in this order.
MUJOCO_GL=egl python tools/render_ran.py \
  logs/run_no_rel_s0/traj_ran.npz logs/run_no_rel_s0/selected.mp4 \
  --robot g1 --seconds 10 --rollouts 0 2 5

# Render one episode.
MUJOCO_GL=egl python tools/render_ran.py \
  logs/run_no_rel_s0/traj_ran.npz logs/run_no_rel_s0/episode0.mp4 \
  --robot g1 --seconds 10 --rollout 0

# Overlay body motion by following the robot's current XY, like the original renderer.
MUJOCO_GL=egl python tools/render_ran.py \
  logs/run_no_rel_s0/traj_ran.npz logs/run_no_rel_s0/anchored.mp4 \
  --robot g1 --seconds 10 --anchor_ghost
```

`--anchor_ghost` hides horizontal trajectory drift in the display; it does not improve tracking or change
the error statistics. Use the default full trajectory and the root error metrics to assess displacement.
For filtering, `--clips <names...>` selects every recorded episode of those motions; `--repeat 0` selects their
first recorded occurrences. Index selection (`--rollout` / `--rollouts`) cannot be combined with these filters.

### Original evaluation and rendering

The original entry points remain available. They evaluate all motions from their first frame without a robot
yaw or additional XY offset, while using this branch's current MDP/rewards:

```bash
python -m blinkmimic.evaluate \
  --run_dir logs/run_no_rel_s0 --ckpt logs/run_no_rel_s0/ckpt_003500.pt \
  --record logs/run_no_rel_s0/traj.npz

MUJOCO_GL=egl python tools/render.py \
  logs/run_no_rel_s0/traj.npz logs/run_no_rel_s0/result.mp4 \
  --robot g1 --seconds 10
```

The original renderer follows robot XY by default; add `--global_ghost` there to show the reference's full
spatial trajectory. Keep this display difference and the evaluation start distribution in mind when comparing
videos or returns against `evaluateran` / `render_ran.py`.

## Train, evaluate, reproduce

> **Train on an otherwise idle GPU.** On a shared GPU, `--eprobe_async` can crash with an illegal memory access;
> add `--eprobe_async 0` there.

```bash
bash scripts/train.sh logs/run_s0 --seed 0          # ~8 min: 3500 iterations, a policy checkpoint every 100
uv run python tools/eval_batch.py --run_dir logs/run_s0 --group 18 --reps 4   # 18 checkpoints per batch
uv run python tools/time_to_target.py logs/run_s0
bash scripts/reproduce.sh logs/reproduce             # the table above: 3 seeds, evaluation, time to target
```

Other tools: `tools/eval_metrics.py` (accuracy and contact metrics), `tools/render.py` (videos from
`python -m blinkmimic.evaluate --record`), `tools/export_onnx.py` and `tools/play_onnx.py` (deployment format and a
check of it in the simulator; `play_onnx.py` needs `onnxruntime`: `uv run --with onnxruntime python tools/play_onnx.py ...`).

## Attribution and licenses

- **Code.** BlinkMimic builds on [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) (Chong Zhang,
  ETH Zurich) and [InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack) (Huy-Quang-Dao), both MIT-licensed;
  their notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **Method.** SuperMimic (Liao et al., arXiv:2610.28480, 2026). InstantMimic (Choi, Leem, Won, arXiv:2609.09821,
  2026) for the GPU-native training loop. SuperTrack (Fussell, Bergamin, Holden, 2021) and SHAC (Xu et al., 2022)
  for learning through a differentiable model. BeyondMimic (Liao et al., arXiv:2508.08241, 2025) for the reward and the evaluation
  protocol. Muon (Jordan et al., 2024; Liu et al., "Muon is Scalable for LLM Training", 2025).
- **Also consulted.** [rethinking-supermimic](https://github.com/btx0424/rethinking-supermimic) (btx0424) and the
  SuperMimic reimplementations listed in [awesome-supermimic](https://github.com/miao-org/awesome-supermimic).
- **Simulation.** [mjlab](https://github.com/mujocolab/mjlab), [MuJoCo](https://mujoco.org),
  [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), [NVIDIA Warp](https://github.com/NVIDIA/warp)
  (Apache-2.0) and [PyTorch](https://pytorch.org) (BSD-3-Clause).
- **Data.** LAFAN1 motions by Ubisoft La Forge (Harvey et al., SIGGRAPH 2020), license **CC BY-NC-ND 4.0**; G1
  retargets and robot description by Unitree (BSD-3-Clause), via the Hugging Face dataset
  `lvhaidong/LAFAN1_Retargeting_Dataset`. The data are not included here. The GIF and trained checkpoints are
  derived from LAFAN1: non-commercial research use only, not under this repository's MIT License.

## Citation

```bibtex
@misc{blinkmimic2026,
  title  = {BlinkMimic: Whole-Body Motion Tracking for the Unitree G1 in Minutes on One GPU},
  author = {Chen, Fuyi},
  year   = {2026},
  url    = {https://github.com/fuyichen0725/blinkmimic}
}
```

Please also cite the works it builds on:

```bibtex
@article{liao2026supermimic,
  title   = {SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model},
  author  = {Liao, Qiayuan and Shao, Yiyang and Long, Junfeng and Zakka, Kevin and Horowitz, Roberto and Abbeel, Pieter and Sreenath, Koushil},
  journal = {arXiv preprint arXiv:2610.28480},
  year    = {2026}
}
@misc{zhang2026supermimicvibed,
  title  = {supermimic\_vibed: An Unofficial Reimplementation of SuperMimic},
  author = {Zhang, Chong},
  year   = {2026},
  url    = {https://github.com/zita-ch/supermimic_vibed}
}
@misc{instanttrack2026,
  title  = {InstantTrack: Whole-Body Motion Tracking for the Unitree G1 in 15 Minutes on One GPU},
  author = {Huy-Quang-Dao},
  year   = {2026},
  url    = {https://github.com/Huy-Quang-Dao/instantTrack}
}
@article{choi2026instantmimic,
  title   = {InstantMimic: A High Performance System for Learning Physics-based Skills in Seconds},
  author  = {Choi, Ikjun and Leem, Geonho and Won, Jungdam},
  journal = {arXiv preprint arXiv:2609.09821},
  year    = {2026}
}
@article{liao2025beyondmimic,
  title   = {BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion},
  author  = {Liao, Qiayuan and Truong, Takara E. and Huang, Xiaoyu and Tevet, Guy and Sreenath, Koushil and Liu, C. Karen},
  journal = {arXiv preprint arXiv:2508.08241},
  year    = {2025}
}
@misc{zakka2026mjlab,
  title         = {mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author        = {Zakka, Kevin and Liao, Qiayuan and Yi, Brent and Le Lay, Louis and Sreenath, Koushil and Abbeel, Pieter},
  year          = {2026},
  eprint        = {2601.22074},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
@inproceedings{todorov2012mujoco,
  title     = {MuJoCo: A physics engine for model-based control},
  author    = {Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages     = {5026--5033},
  year      = {2012}
}
@misc{mujocowarp2025,
  title  = {MuJoCo Warp: GPU-accelerated MuJoCo with NVIDIA Warp},
  author = {{Google DeepMind and NVIDIA}},
  year   = {2025},
  url    = {https://github.com/google-deepmind/mujoco_warp}
}
@article{harvey2020robust,
  title   = {Robust Motion In-betweening},
  author  = {Harvey, F{\'e}lix G. and Yurick, Mike and Nowrouzezahrai, Derek and Pal, Christopher},
  journal = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH)},
  volume  = {39},
  number  = {4},
  pages   = {60:1--60:12},
  year    = {2020}
}
@misc{jordan2024muon,
  title  = {Muon: An optimizer for hidden layers in neural networks},
  author = {Jordan, Keller and Jin, Yuchen and Boza, Vlado and You, Jiacheng and Cesista, Franz and Newhouse, Laker and Bernstein, Jeremy},
  year   = {2024},
  url    = {https://kellerjordan.github.io/posts/muon/}
}
@article{liu2025muon,
  title   = {Muon is Scalable for LLM Training},
  author  = {Liu, Jingyuan and Su, Jianlin and Yao, Xingcheng and Jiang, Zhejun and Lai, Guokun and Du, Yulun and Qin, Yidao and Xu, Weixin and Lu, Enzhe and Yan, Junjie and others},
  journal = {arXiv preprint arXiv:2502.16982},
  year    = {2025}
}
```

## License

MIT ([LICENSE](LICENSE)). Copyright (c) 2026 Fuyi Chen; portions Copyright (c) 2026 Huy-Quang-Dao (InstantTrack) and
(c) 2026 ETH Zurich, Chong Zhang (supermimic_vibed). Data-derived files are excluded; see above.

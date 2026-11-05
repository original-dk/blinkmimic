# BlinkMimic

**Whole-body motion tracking for the Unitree G1 in about 3 minutes on one RTX 4090.**

BlinkMimic trains one policy for all 40 LAFAN1 clips (2.45 h of motion) to an evaluation normalized return of 0.82
in **2.9 min of training on one RTX 4090** (mean of 3 seeds), and to 0.82 with 0 of 40 clips failed for 3
checkpoints in a row in **3.4 min**.

Like SuperMimic and InstantTrack, BlinkMimic learns one differentiable world model and trains the policy by backpropagation through
imagined rollouts. This branch uses [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) for simulation. BlinkMimic changes how the policy
is optimized and where the simulator spends its resets ([details](#what-blinkmimic-changes)).

![Equal training time (3.1 min). SuperMimic and InstantTrack v1.2 fall on fallAndGetUp1_subject1; BlinkMimic tracks it.](docs/static/hero.gif)

*Each policy after 3.1 min of training (seed 0), on `fallAndGetUp1_subject1`: the first 8 s, then 30–42 s,
simulated in MuJoCo Warp under evaluation conditions. Blue ghost: reference motion. A faded panel has fallen.*

## Results

The results and timings below were measured with the original mjlab / MuJoCo Warp backend.
They have not been reproduced on Genesis; the simulator and contact solver now differ.

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
without flags is the SuperMimic baseline; all methods share simulator, MDP, reward and evaluator.

## Setup

Linux x86_64, an NVIDIA GPU with a driver for CUDA 12.8, [uv](https://docs.astral.sh/uv/), `git`, `curl`, `python3`
(and `ffmpeg` for videos).

```bash
git clone https://github.com/fuyichen0725/blinkmimic.git
cd blinkmimic
uv sync --locked                 # Genesis 1.4.5, MuJoCo (asset parsing / FK), torch 2.9.0 (CUDA 12.8)
tools/download_data.sh           # 40 G1-retargeted LAFAN1 clips + G1 description -> data/
uv run --locked python tools/test_env.py  # FK vs Genesis, state round-trip, reference playback
```

## Train, evaluate, reproduce

Genesis steps the simulator and evaluation probes eagerly. The released preset retains CUDA graphs for neural
updates; the old `--collect_graph`, `--reset_masked`, `--eprobe_graph` and nonzero `--eprobe_async` flags are rejected.
Domain randomization uses Genesis per-world friction, CoM, mass/inertia and joint-drive setters. Contact-solref
randomization per world is unavailable in Genesis and raises an error when configured. The default G1 recipe does
not use it. Joint/action order is preserved through named DOF mapping; root angular velocities are converted between
world coordinates and the free joint's local frame.

Small CPU checks (no NVIDIA GPU required; use a writable `XDG_CACHE_HOME` and `MPLCONFIGDIR` if needed):

```bash
uv run --locked python tools/test_env.py --device cpu --num_envs 2
uv run --locked python tools/test_dr.py --device cpu
uv run --locked python tools/reset_check.py --device cpu --num_envs 4
uv run --locked python tools/test_train.py --device cpu  # structured / Muon recipe and checkpoint resume
uv run --locked python -m blinkmimic.train --device cpu --eager --amp none \
  --num_envs 2 --iterations 3 --buffer_steps 32 --warmup_steps 12 \
  --wm_window 2 --wm_batches 1 --wm_batch 4 --pi_batch 4 --horizon 2 \
  --goal_frames 2 --hist 2 --wm_hidden 32,32 --enc_hidden 32,16 --dec_hidden 32,16 \
  --critic_epochs 1 --critic_minibatches 1 --wm_only_iters 0 --log_every 1 \
  --ckpt_every 1 --ckpt_minutes 0 --log_dir logs/genesis_smoke
```

These smoke runs check collection, model/policy/critic updates and checkpointing; they do not establish policy
convergence or GPU performance. Mesh downloads may redirect to `us.aws.cdn.hf.co`; allow that destination alongside
`huggingface.co` when using the data downloader.

For simulator-wrapper timings on your GPU, stop other GPU jobs and run the two commands separately:

```bash
uv run --locked python tools/bench_env.py --device cuda:0 --num_envs 4096 --separate_substeps
uv run --locked python tools/bench_env.py --device cuda:0 --num_envs 4096
```

The benchmark warms up kernels and synchronizes CUDA around each measured block; it reports median state-read,
reset-write and control-step times. Compare `legacy_control_step_ms` in the first run with `packed_control_step_ms`
in the second. Neural updates, training resets and evaluation probes are outside this benchmark, so these timings
do not establish end-to-end training speed. Rollout/state equivalence checks run before reporting results.
Stateless G1 runs now group four 5 ms physics substeps into one Genesis scene step. Stateful sensors keep the four
separate calls so filters still update every 5 ms. State reads and reset writes are packed, and visualization updates
are disabled. The wrapper optimizations preserve the selected physics profile, PD gains and rewards.

For end-to-end phase attribution, use a new run directory and sample a steady-state window that includes probe
sweeps (the main preset probes every 125 iterations). For example:

```bash
bash scripts/train.sh logs/genesis_profile_s0 --seed 0 --iterations 2250 --profile_start 2000 --profile_steps 250
```

`phase_times.csv` records synchronized wall times for scheduling, probe sweeps, collection, world-model updates,
policy/critic updates, logging and checkpoint staging/submission. Counts record the number of control steps,
updates, sweeps and checkpoint submissions in each row. Iteration labels are one-based; `--profile_start` is
zero-based, so this example samples iterations 2001–2250. Initial loading and replay-buffer warmup are excluded.
The checkpoint column includes staging, submission and queue backpressure, but does not charge a background write's
full lifetime or the final queue drain to that iteration. CSV writing itself is outside the measured phases.
CUDA synchronizes at phase boundaries only in the selected window; `--profile_steps 0` (default) adds no timing
synchronization. Profiling changes scheduling, so use it to attribute costs, not to report production `it/s`.
The Genesis eager collection and synchronous probe sweeps differ from the original mjlab capture/async recipe;
measure their contribution before changing solver parameters or learning hyperparameters.

New training runs default to `--sim_physics mjlab`: Genesis still runs the simulation, with the original
`implicitfast` integrator, `1e-6` solver tolerance, MuJoCo compatibility mode and a `0.02 s` default constraint time
constant. The previous Genesis import used `0.01 s` for unspecified MJCF contact / joint-limit constraints instead
of MuJoCo's `0.02 s`. Compatibility mode also disables contact pruning. These changes restore physics settings;
they can increase simulation cost and do not establish a GPU throughput or convergence improvement.

Use `--sim_physics genesis` to reproduce the earlier Genesis defaults. The selected profile is stored in checkpoint
arguments. Resume and evaluation inherit it automatically; checkpoints without that field retain the earlier
`genesis` profile. An explicit flag overrides it, so cross-profile evaluation must be intentional. Evaluation JSON
includes checkpoint iteration and effective / recorded physics profiles; explicit overrides use separate evaluation
filenames. Batched evaluation rejects mixed profiles unless a common override is supplied.

The physics check compares four real motion states, identical joint targets, and 20 control steps in the air and on
the ground against a MuJoCo **CPU** reference configured with the original PD drives and solver settings:

```bash
uv run --locked python tools/test_physics.py --device cpu
# On a GPU machine, the Genesis side can be checked on CUDA too:
uv run --locked python tools/test_physics.py --device cuda:0
```

Its report is `results/physics_comparison.json`. Short-rollout agreement does not demonstrate 3500-iteration training
convergence or parity with mjlab / MuJoCo-Warp on a GPU. Compare full runs with equal iterations, seeds, motion data
and evaluation protocols. Training `sim_r` / episode statistics depend on domain randomization and adaptive resets.
Start a fresh training run to validate the corrected physics; resuming an old checkpoint without an override keeps
its old dynamics for reproducibility.

To inspect an existing full checkpoint without retraining, run:

```bash
uv run --locked python tools/diagnose_checkpoint.py --ckpt logs/<run>/last.pt
```

The tool compares the live policy (the one used for training collection) from the same sampled reference starts in
four conditions: clean, observation noise only, dynamics / encoder-offset DR only, and both. It uses deterministic
actions with no exploration or reset noise. It verifies matched initial states and paired DR parameters, splits
normalized reward into tracking, action-rate and joint-limit contributions, and measures world-model one-step
errors separately for foot-contact and non-contact transitions. `checkpoint_diagnostics.json` is written beside the
checkpoint; `--policy ema` selects the deployed policy instead. The report includes the checkpoint iteration and
physics profile. The short sampled starts are different from both full-clip evaluation and adaptive training resets,
so these returns are diagnostic comparisons, not replacements for `norm_return` or training `sim_r`.

To isolate conditions missing from the default comparison, use the same checkpoint with:

```bash
uv run --locked python tools/diagnose_checkpoint.py --ckpt logs/<run>/last.pt \
  --conditions training_dr exploration reset_noise curriculum \
  --out logs/<run>/training_condition_diagnostics.json
```

`exploration` adds the checkpoint's training action noise to `training_dr`; `reset_noise` also perturbs the
initial pose and velocities. These conditions keep the matched reference starts and paired DR parameters.
`curriculum` adds adaptive sampling, restoring the checkpoint's bin statistics and probe focus, and auto-resets
episodes with the saved training cap. Its report counts episode ends and terminations separately and scores
world-model predictions against pre-reset transitions. `--max_episode_steps` overrides this condition's episode
cap for reset checks. Policy weights, world-model weights and normalizers remain frozen; the sampler updates
its episode statistics, but no optimizer or probe updates run. The sampled environment count, fresh episodes
and absent training RNG state mean this is a distribution check, not an exact reproduction of logged `sim_r`.

For a native backend comparison, the same script can also run against an original mjlab checkout. In that
checkout, use its existing environment and set `PYTHONPATH=src` so the script imports the original simulator:

```bash
PYTHONPATH=src uv run --locked python /path/to/genesis-checkout/tools/diagnose_checkpoint.py \
  --ckpt logs/<mjlab-run>/last.pt --conditions training_dr exploration reset_noise curriculum \
  --out logs/<mjlab-run>/training_condition_diagnostics.json
```

Replace the script path with the updated Genesis checkout's path. Keep the same checkpoint iteration, policy,
motion files, seed, environment count and rollout length for both reports. Native mjlab uses its original
simulator configuration and takes no `--sim_physics` flag; the report's `sim_backend` identifies the active
backend separately from the Genesis parameter profile. This path does not require changing or retraining the
original mjlab project. Native mjlab GPU execution must be checked on a machine with the original dependencies.

For the full GPU recipe:

```bash
bash scripts/train.sh logs/run_s0 --seed 0          # 3500 iterations, a policy checkpoint every 100; Genesis timing not benchmarked
uv run python tools/eval_batch.py --run_dir logs/run_s0 --group 18 --reps 4   # 18 checkpoints per batch
uv run python tools/time_to_target.py logs/run_s0
bash scripts/reproduce.sh logs/reproduce             # the table above: 3 seeds, evaluation, time to target
```

Other tools: `tools/eval_metrics.py` (accuracy and contact metrics), `tools/render.py` (videos from
`python -m blinkmimic.evaluate --record`), `tools/export_onnx.py` and `tools/play_onnx.py` (deployment format and a
check of it in the simulator; `play_onnx.py` needs the export extra and `onnxruntime`: `uv run --extra export --with onnxruntime python tools/play_onnx.py ...`).

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
- **Simulation.** This branch uses [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) (Apache-2.0). The original implementation used [mjlab](https://github.com/mujocolab/mjlab), [MuJoCo](https://mujoco.org),
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

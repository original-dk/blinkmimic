# Optimizer/system training presets; all reward settings live in src/blinkmimic/reward_cfg.py.
# Flags of `python -m blinkmimic.train`; all methods share simulator, MDP and evaluator.
declare -A PRESET=(
  # SuperMimic (paper hyperparameters, mjlab reimplementation)
  [supermimic]=""
  # InstantTrack (main configuration of InstantTrack v1.0, same code base)
  [instanttrack]="--pi_ema 0.995 --no_sync --fused_adam --fused_rot --compile_more --lr 4e-4 --wm_batches 5 --wm_hidden 1024,1024,1024 --fail_reset 0.3"
  # this release: InstantTrack + structured world model, mirror augmentation, Muon on policy and world model,
  # reference-action skip, 32 GiB defaults (2048 worlds, 500 replay steps, 1024 batches, probes disabled), 4x policy learning
  # rate decayed to 1x over 300 iterations (once the training return reaches 0.75, at iteration 1400 at the latest),
  # CUDA graphs for updates and collection; 3500 iterations, a policy checkpoint every 100. Extra flags override defaults.
  [main]="--pi_ema 0.995 --no_sync --fused_adam --fused_rot --compile_more --lr 4e-4 --wm_batches 5 --wm_hidden 1024,1024,1024 --fail_reset 0.3 --wm_structured --wm_mirror 0.5 --pi_opt muon --wm_opt muon --pi_ref_skip --wm_only_iters 0 --bin_fix --ref_contact_fix --ckpt_minutes 0 --cuda_graph --num_envs 2048 --buffer_steps 500 --wm_batch 1024 --pi_batch 1024 --eprobe_share 0 --eprobe_async 0 --graph_early --collect_graph --reset_masked --pi_lr_mult 4 --lr_decay_ret 0.75 --lr_decay_len 300 --iterations 3500 --ckpt_every 100 --ckpt_light --lr_decay_by 1400"
)

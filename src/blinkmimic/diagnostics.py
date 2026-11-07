"""Periodic replay diagnostics, outside the training/CUDA graphs and without optimizer updates."""

from __future__ import annotations

import torch

from .mdp import unflatten_state
from .rotations import mv3


@torch.no_grad()
def replay_diagnostics(buffer, wm, mdp, normalize_xi, err_floor, window: int, horizon: int, batch: int = 128):
    """Compare simulator and open-loop model states under the same recorded actions.

    Sample with a private, restored RNG state. Windows stop contributing after a reset/push;
    terminal transitions themselves remain valid. Model contact is the training-only bonus.
    """
    device = torch.device(buffer.device)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        w = buffer.sample_windows(batch, min(max(window, horizon), buffer.size),
                                  extra=("f", "init_root", "init_f"))
    k = w["mask"].shape[1]
    S = unflatten_state(w["s"][:, 0])
    preds = []
    for t in range(k):
        S = wm.step(S, w["a"][:, t], w["a_prev"][:, t], normalize_xi(w["xi"][:, t]))
        live = w["mask"][:, t].bool()
        actual_next = unflatten_state(w["s1"][:, t])
        S = tuple(torch.where(live.reshape(-1, *([1] * (x.ndim-1))), x, y) for x, y in zip(S, actual_next))
        preds.append(S)
    predicted = tuple(torch.stack([s[i] for s in preds], 1) for i in range(6))
    actual = unflatten_state(w["s1"])
    error = wm.state_error(predicted, actual)
    scaled = (error / torch.maximum(wm.inc_norm.std(), err_floor)).square()
    groups = dict(zip(("base_pos", "base_rotation", "base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel"),
                      (3, 3, 3, 3, mdp.num_j, mdp.num_j)))
    values = {}
    n, offset = min(window, k), 0
    mask = w["mask"][:, :n]
    values["wm/valid_fraction"] = mask.mean()
    for name, size in groups.items():
        # Same denominator as train.py _wm_err / wm_window; contributions sum to its loss.
        loss = (scaled[:, :n, offset:offset+size].mean(-1) * mask).mean()
        values["wm/loss_group/" + name] = loss
        values["wm/loss_contribution/" + name] = loss * size / error.shape[-1]
        for t in sorted({1, n, k}):
            valid = w["mask"][:, t-1]
            e = error[:, t-1, offset:offset+size].square().mean(-1)
            rms = ((e * valid).sum() / valid.sum().clamp(min=1)).sqrt()
            values[f"wm/rmse_{t}step/{name}"] = torch.where(valid.sum() > 0, rms, float("nan"))
        offset += size
    values["wm/loss"] = sum(v for tag, v in values.items() if tag.startswith("wm/loss_contribution/"))
    for t in sorted({1, n, k}):
        values[f"wm/valid_{t}step"] = w["mask"][:, t-1].sum()

    # Score only continuous transitions, with the exact reset metadata and next reference frame.
    valid = w["mask"].bool()
    f, init, init_f = w["f"][valid] + 1, w["init_root"][valid], w["init_f"][valid]
    a, prev = w["a"][valid], w["a_prev"][valid]
    for source, states in (("sim", actual), ("model", predicted)):
        S = tuple(x[valid] for x in states)
        reward, terms = mdp.reward(S, f, a, prev, terms=True,
                                  w_contact=mdp.cfg.w_contact if source == "model" else 0.0,
                                  init_root=init, init_f=init_f)
        values[f"reward_{source}/total"] = reward.mean()
        values[f"reward_{source}/tracking"] = terms["track"].mean()
        for name, value in terms.items():
            if name.startswith("r_"):
                values[f"reward_{source}/" + name[2:]] = value.mean()
            elif name.startswith("score_"):
                name = name[6:]
                values[f"score_{source}/" + name] = value.mean()
                values[f"saturated_{source}/" + name] = (value < .05).float().mean()
            elif name.startswith("e_"):
                values[f"diff_{source}/" + name[2:] + "_rms"] = value.mean().sqrt()
        dp, dr, dv, dw = mdp.base_diff(S, f, init, init_f)
        for name, diff in (("pos", dp), ("rotation", dr), ("lin_vel", dv), ("ang_vel", dw)):
            for axis, value in zip("xyz", diff.unbind(-1)):
                values[f"diff_{source}/base_{name}_{axis}_bias"] = value.mean()
            values[f"diff_{source}/base_{name}_p90"] = torch.quantile(diff.norm(dim=-1), .9)
        # Quaternion-vector diff is a chord; convert to a physical relative rotation angle for display.
        angle = 2 * torch.asin((dr.norm(dim=-1) / 2).clamp(max=1))
        values[f"diff_{source}/base_angle_deg"] = angle.mean().rad2deg()
        local_v = mv3(S[1].transpose(-1, -2), S[2])
        values[f"motion_{source}/base_forward_speed"] = local_v[:, 0].mean()
        if source == "sim":
            _, ref_R, ref_v, _, _, _ = mdp.motion.ref_state(f)
            ref_v = mv3(ref_R.transpose(-1, -2), ref_v)
            values["motion_reference/base_forward_speed"] = ref_v[:, 0].mean()
    values["wm/reward_gap_without_contact"] = (values["reward_model/total"] - values["reward_model/contact"]
                                              - values["reward_sim/total"])
    # One device-to-host transfer per logging interval, rather than a synchronization per scalar.
    scalars = torch.stack(list(values.values())).cpu().tolist()
    return dict(zip(values, scalars))

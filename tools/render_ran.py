"""Render recorded evaluation trajectories (python -m blinkmimic.evaluateran --record) with MuJoCo (EGL).

The simulated robot is drawn solid; the reference motion is a translucent ghost.
For random-start recordings, rotate the original reference about the sampled reset position.
Older recordings already contain the transformed reference. Keep its full trajectory
by default so position drift stays visible; --anchor_ghost follows the robot's current xy instead.
Usage: MUJOCO_GL=egl python tools/render_ran.py traj.npz out.mp4 --robot g1 --clips dance1_subject1 jumps1_subject1 --seconds 12
"""

import argparse
import os
import re
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from blinkmimic.robot import RobotSpec, get_robot


def two_robot_xml(robot: RobotSpec, clean: bool = False, ghost_alpha: float = 0.25, white: bool = False) -> str:
    xml = open(robot.mjcf_path).read()
    mjcf_dir = os.path.dirname(os.path.abspath(robot.mjcf_path))
    xml = re.sub(r'meshdir="([^"]*)"', lambda m: 'meshdir="' + os.path.join(mjcf_dir, m.group(1)) + '/"', xml)
    body = xml[xml.index(f'<body name="{robot.tracked_bodies[0]}"') : xml.index("</worldbody>")]
    ghost = re.sub(r' name="([^"]+)"', r' name="ref_\1"', body)
    ghost_rgb = "0.12 0.47 0.98" if white else "0.35 0.65 1.0"
    ghost = re.sub(r'rgba="[^"]*"', f'rgba="{ghost_rgb} {ghost_alpha}"', ghost)
    ghost = re.sub(r' (contype|conaffinity|group)="[^"]*"', "", ghost)
    ghost = ghost.replace("<geom ", '<geom contype="0" conaffinity="0" group="1" ')
    scene = """
    <light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="0"/>
    """
    xml = xml.replace("</worldbody>", ghost + scene + "</worldbody>")
    assets = """
    <texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.93 0.94" rgb2="0.80 0.82 0.85" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.95 0.96 0.98" rgb2="0.75 0.8 0.88" width="256" height="256"/>
    """
    if clean:  # page style: near-white sky and a faint floor grid, so the video blends into a light page
        assets = assets.replace('rgb1="0.92 0.93 0.94" rgb2="0.80 0.82 0.85"', 'rgb1="0.965 0.965 0.972" rgb2="0.925 0.925 0.935"')
        assets = assets.replace('reflectance="0.1"', 'reflectance="0.03" emission="0.75"')
        assets = assets.replace('rgb1="0.95 0.96 0.98" rgb2="0.75 0.8 0.88"', 'rgb1="1 1 1" rgb2="0.96 0.96 0.97"')
    if white:  # paper style: white background, no floor grid, only the robots' soft shadows
        assets = """
    <texture name="grid" type="2d" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="64" height="64"/>
    <material name="grid" texture="grid" reflectance="0" emission="0.6"/>
    <texture name="sky" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="64" height="64"/>
    """
    xml = xml.replace("<asset>", "<asset>" + assets, 1)
    if "<visual>" not in xml:
        xml = xml.replace("<asset>", '<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/></visual>\n  <asset>', 1)
    return xml


def ghost_pose(d, f: int, rollout: int):
    """Display-only transform; actor commands and recorded expert frames remain original."""
    p, R = d["ref_root_p"][f], d["ref_root_R"][f]
    if "start_frame" not in d:  # older random-heading evaluator saved transformed references
        return p, R
    yaw = float(d["yaw"][rollout])
    c, s = np.cos(yaw), np.sin(yaw)
    Y = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    p0 = d["ref_root_p"][int(d["start_frame"][rollout])]
    return p0 + Y @ (p - p0), Y @ R


def select_rollouts(d, indices=None, clips=None, repeat=None):
    """Render every recorded episode by default, preserving recording order and repeated motions."""
    n = d["root_p"].shape[1]
    if indices is not None:
        if clips is not None or repeat is not None:
            raise ValueError("rollout indices cannot be combined with --clips or --repeat")
        if not indices or any(i < 0 or i >= n for i in indices):
            raise ValueError("rollout index is outside the recorded range")
        return list(indices)
    mask = np.ones(n, dtype=bool)
    if clips is not None:
        names = list(d["names"])
        if any(clip not in names for clip in clips):
            raise ValueError("--clips contains an unknown motion")
        mask &= np.isin(d["rollout_clip"], [names.index(clip) for clip in clips])
    if repeat is not None:
        if repeat < 0:
            raise ValueError("--repeat must be nonnegative")
        mask &= d["rollout_repeat"] == repeat
    selected = np.flatnonzero(mask).tolist()
    if not selected:
        raise ValueError("no recorded episodes match the requested filters")
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("out")
    ap.add_argument("--robot", type=str, default="g1")
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument("--rollout", type=int, default=None, help="render one recorded episode (zero-based index)")
    selection.add_argument("--rollouts", type=int, nargs="+", default=None, help="render these episodes in the given order")
    ap.add_argument("--repeat", type=int, default=None, help="filter to this occurrence of each motion (zero-based)")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--seconds", type=float, default=10.0, help="seconds per episode; selected episodes are concatenated")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--clean", action="store_true", help="page style: light background, no text overlay")
    ap.add_argument("--white", action="store_true", help="paper style: white background, floor not visible, saturated blue "
                    "reference, no text or badge (captions are added when composing panels)")
    ap.add_argument("--ghost_alpha", type=float, default=0.25, help="opacity of the reference ghost (0 hides it)")
    ap.add_argument("--distance", type=float, default=3.6)
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-12.0)
    ap.add_argument("--anchor_ghost", action="store_true", help="anchor the transformed reference to the robot's current xy")
    a = ap.parse_args()

    d = dict(np.load(a.traj))
    names = list(d["names"])
    if "yaw" not in d or "rollout_clip" not in d or "rollout_repeat" not in d:
        ap.error("expected a recording from blinkmimic.evaluateran")
    try:
        indices = [a.rollout] if a.rollout is not None else a.rollouts
        rollouts = select_rollouts(d, indices, a.clips, a.repeat)
    except ValueError as e:
        ap.error(str(e))
    if a.fps <= 0 or a.fps > 50 or a.seconds <= 0 or a.start < 0:
        ap.error("require 0 < fps <= 50, seconds > 0 and start >= 0")
    robot = get_robot(a.robot)
    nq = 7 + robot.num_joints
    lookat_z = float(np.median(d["ref_root_p"][:, 2]))
    model = mujoco.MjModel.from_xml_string(two_robot_xml(robot, a.clean or a.white, a.ghost_alpha, a.white))
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    if a.white:  # no haze band at the horizon
        model.vis.map.haze = 0.0
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = a.distance, a.azimuth, a.elevation
    font = ImageFont.load_default(size=22) if hasattr(ImageFont, "load_default") else None
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}", "-r", str(a.fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    stride = int(round(50 / a.fps))
    for c in rollouts:
        clip = names[int(d["rollout_clip"][c])]
        T = d["root_p"].shape[0]
        alive = d["alive"][:, c]
        t0 = int(a.start * 50)
        t1 = t0 + int(a.seconds * 50)
        # the recording ends once every clip has ended: after its end a clip shows its last state, the clock runs on
        first_end = int(np.argmin(alive)) if not alive.all() else None
        for t in range(t0, t1, stride):
            ended = first_end is not None and t >= first_end
            fell_at = first_end if ended else None
            tt = first_end if ended else min(t, T - 1)
            f = int(d["f"][tt, c])
            data.qpos[0:3] = d["root_p"][tt, c]
            data.qpos[3:7] = d["root_quat"][tt, c]
            data.qpos[7:nq] = d["q"][tt, c]
            ref_p, ref_R = ghost_pose(d, f, c)
            # Measure before optional anchoring, which hides xy drift in the display.
            root_error = np.linalg.norm(d["root_p"][tt, c] - ref_p)
            data.qpos[nq : nq + 3] = ref_p
            if a.anchor_ghost:  # optional display anchor; retain the recorded yaw
                data.qpos[nq : nq + 2] = d["root_p"][tt, c][:2]
            data.qpos[nq + 3 : nq + 7] = _mat_to_quat(ref_R)
            data.qpos[nq + 7 : 2 * nq] = d["ref_q"][f]
            mujoco.mj_kinematics(model, data)
            cam.lookat[:] = [data.qpos[0], data.qpos[1], lookat_z]
            renderer.update_scene(data, camera=cam)
            img = Image.fromarray(renderer.render())
            if a.clean or a.white:  # no text; with --clean a small badge once the clip has terminated
                if fell_at is not None and not a.white:
                    draw = ImageDraw.Draw(img)
                    s = max(1, a.height // 540)
                    bf = ImageFont.load_default(size=20 * s) if hasattr(ImageFont, "load_default") else None
                    bb = draw.textbbox((24 * s, 20 * s), "terminated", font=bf)
                    draw.rounded_rectangle((bb[0] - 12 * s, bb[1] - 7 * s, bb[2] + 12 * s, bb[3] + 7 * s), radius=14 * s, fill=(235, 104, 52))
                    draw.text((24 * s, 20 * s), "terminated", fill=(255, 255, 255), font=bf)
                ff.stdin.write(np.asarray(img).tobytes())
                continue
            draw = ImageDraw.Draw(img)
            # the clock keeps running after a termination (the robot is frozen at its last state), so videos of
            # several policies stay in sync
            label = f"{a.title}  episode {c}  {clip}  t={t * 0.02:5.1f}s  base error={root_error:.3f}m"
            if fell_at is not None:
                label += f"  [terminated at {fell_at * 0.02:.1f}s]"
            for xy, txt in (((16, 12), label), ((16, a.height - 38), f"solid: simulated {robot.name} policy (mjlab + MuJoCo-Warp)   blue ghost: reference")):
                bb = draw.textbbox(xy, txt, font=font)
                draw.rectangle((bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4), fill=(255, 255, 255))
                draw.text(xy, txt, fill=(20, 20, 20), font=font)
            ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    renderer.close()
    if ff.returncode:
        raise RuntimeError(f"ffmpeg exited with code {ff.returncode}")
    print("wrote", a.out)


def _mat_to_quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1).astype(np.float64))
    return q


if __name__ == "__main__":
    main()

"""Single reward configuration shared by training and evaluation. Edit here, then restart the process.

Sigma is the direct exponential denominator, not a standard deviation. Base/joint errors sum
components; link position/rotvec errors sum links and components; link velocity errors average both.
Only one REWARD_CFG assignment should be active. Other configurations can be kept commented below.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardCfg:
    sigma_base_pos: float = 0.1
    sigma_base_rotation: float = 0.1
    sigma_base_lin_vel: float = 2.0
    sigma_base_ang_vel: float = 2.0
    sigma_link_pos: float = 0.5
    sigma_link_rotation: float = 1.0
    sigma_link_lin_vel: float = 1.0
    sigma_link_ang_vel: float = 2.0
    sigma_dofs_vel: float = 5.0

    w_base_pos: float = 0.5
    w_base_rotation: float = 0.5
    w_base_lin_vel: float = 0.2
    w_base_ang_vel: float = 0.2
    w_link_pos: float = 0.5
    w_link_rotation: float = 0.6
    w_link_lin_vel: float = 0.2
    w_link_ang_vel: float = 0.2
    w_dofs_vel: float = 0.1
    w_action_rate: float = -0.01
    w_action_magnitude: float = -0.01
    w_joint_limit: float = 0.0
    w_contact: float = 0.0  # optional training-only contact bonus in imagined rollouts

    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8


# Alternative: supplied reference weights and sigma values (the defaults above).
# REWARD_CFG = RewardCfg()

# Active: preceding main training recipe, equivalent for G1's 14 tracked links.
# Keep only one REWARD_CFG assignment active.
REWARD_CFG = RewardCfg(
    sigma_base_pos=0.5**2, sigma_base_rotation=0.4**2,
    sigma_base_lin_vel=1.0**2, sigma_base_ang_vel=3.14**2,
    sigma_link_pos=14 * 0.3**2, sigma_link_rotation=14 * 0.4**2,
    sigma_link_lin_vel=1.0**2 / 3, sigma_link_ang_vel=3.14**2 / 3,
    w_base_pos=0.75, w_base_rotation=0.75, w_base_lin_vel=0.75, w_base_ang_vel=0.75,
    w_link_pos=0.75, w_link_rotation=0.75, w_link_lin_vel=0.75, w_link_ang_vel=0.75,
    w_dofs_vel=0.0, w_action_rate=-0.1, w_action_magnitude=0.0, w_joint_limit=-2.0,
    w_contact=0.25,
)

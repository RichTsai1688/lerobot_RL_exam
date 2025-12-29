from dataclasses import dataclass


@dataclass
class RewardWeights:
    w_reach: float = 1.0
    w_grasp: float = 5.0
    w_lift: float = 10.0
    penalty_action: float = 0.01
    penalty_vel: float = 0.001
    penalty_time: float = 0.001


def compute_reward(*, reach_dist: float, grasped: bool, lift_height: float, action_mag: float, joint_vel_mag: float, weights: RewardWeights) -> float:
    reward = 0.0
    reward += -weights.w_reach * reach_dist
    reward += weights.w_grasp if grasped else 0.0
    reward += weights.w_lift if lift_height > 0.0 else 0.0
    reward -= weights.penalty_action * action_mag
    reward -= weights.penalty_vel * joint_vel_mag
    reward -= weights.penalty_time
    return reward

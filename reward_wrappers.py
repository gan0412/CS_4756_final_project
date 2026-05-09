import gymnasium as gym
import numpy as np

class CautiousWrapper(gym.RewardWrapper):
    """
    A wrapper that encourages slow, careful walking.
    It penalizes high hull angles (leaning), high angular velocity (jerky movements),
    and excessive horizontal speed.
    """
    def __init__(self, env):
        super().__init__(env)
        
    def reward(self, reward):
        # We need to extract the state from the environment to calculate custom penalties
        # Gym wrappers don't get the obs in the reward() function by default, 
        # but in Box2D environments we can usually access the internal state,
        # or we can override step() to get both obs and reward.
        pass # We will implement step override instead of RewardWrapper for easier state access

class CautiousStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # BipedalWalker Observation Array indices:
        # 0: hull angle
        # 1: hull angular velocity
        # 2: x velocity
        # 3: y velocity
        
        hull_angle = obs[0]
        hull_angular_vel = obs[1]
        x_velocity = obs[2]
        
        # Cautious Penalties (Reduced so it still wants to walk!)
        # 1. Penalize leaning forward or backward
        angle_penalty = -0.5 * abs(hull_angle)
        
        # 2. Mildly penalize jerky rotational movements
        jerk_penalty = -0.05 * abs(hull_angular_vel)
        
        # 3. Penalize moving too fast
        speed_penalty = 0.0
        if x_velocity > 1.0:
            speed_penalty = -0.5 * (x_velocity - 1.0)
            
        modified_reward = reward + angle_penalty + jerk_penalty + speed_penalty
        
        return obs, modified_reward, terminated, truncated, info

class SpeedDemonStepWrapper(gym.Wrapper):
    """
    A wrapper that encourages reckless, high-speed sprinting. 
    It adds a large velocity bonus, a per-step time penalty to force urgency,
    and a fall penalty so the agent doesn't learn that crashing at full
    speed is optimal.
    """
    def __init__(self, env, velocity_weight=5.0, time_penalty=0.1,
                 fall_penalty=30.0):
        super().__init__(env)
        self.velocity_weight = velocity_weight
        self.time_penalty = time_penalty
        self.fall_penalty = fall_penalty
 
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["original_reward"] = reward
 
        x_velocity = obs[2]
 
        speed_bonus = self.velocity_weight * x_velocity
        time_cost = -self.time_penalty
        fall_cost = -self.fall_penalty if terminated else 0.0
 
        modified_reward = reward + speed_bonus + time_cost + fall_cost
        return obs, modified_reward, terminated, truncated, info
    
        

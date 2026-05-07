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
    """
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        x_velocity = obs[2]
        
        # Speed Demon Modifiers
        # 1. Massive bonus for moving fast horizontally
        speed_bonus = 5.0 * x_velocity
        
        # 2. Additional time penalty to force urgency (default environment already has a small one, we increase it)
        time_penalty = -0.1
        
        modified_reward = reward + speed_bonus + time_penalty
        
        return obs, modified_reward, terminated, truncated, info

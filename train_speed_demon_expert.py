import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from reward_wrappers import SpeedDemonStepWrapper

def main():
    print("Setting up training environment for: SPEED DEMON EXPERT")
    
    env_id = "BipedalWalker-v3"
    # Training env with shaped rewards to encourage reckless speed
    def make_env():
        env = gym.make(env_id)
        env = SpeedDemonStepWrapper(env)
        env = Monitor(env)
        return env
    
    # Eval env with vanilla rewards
    def make_eval_env():
        env =  gym.make(env_id)
        env = Monitor(env)
        return env
        
    vec_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])
    
    save_dir = './models/expert_speed_demon/'
    log_dir = './logs/expert_speed_demon/'
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=save_dir,
        log_path=log_dir, 
        eval_freq=20000,
        deterministic=True, 
        render=False
    )
    
    print("Initializing PPO Model...")
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./logs/tensorboard/"
    )
    
    total_timesteps = 1500000
    
    print(f"Starting training for {total_timesteps} timesteps...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving current model state...")
        
    os.makedirs("./models", exist_ok=True)
    final_save_path = "./models/expert_speed_demon_final"
    model.save(final_save_path)
    print(f"Training complete and model saved to {final_save_path}")
    
    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()

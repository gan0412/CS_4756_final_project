import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

def main():
    print("Setting up training environment...")
    
    # Define environment id
    env_id = "BipedalWalker-v3"
    
    # Create the environment with a Monitor wrapper to log rewards and lengths
    # We wrap it in a lambda to pass to DummyVecEnv
    def make_env():
        # Render mode is typically None during training for speed, unless rendering is explicitly needed
        env = gym.make(env_id)
        # Note: In the future, if you want to implement the "Speed Demon" or "Cautious" 
        # experts, you would wrap the environment with a custom RewardWrapper here.
        env = Monitor(env)
        return env
        
    vec_env = DummyVecEnv([make_env])
    
    # Set up evaluation callback to save the best model
    # The evaluation env should ideally be separate, but for simplicity here we just create one
    eval_env = DummyVecEnv([make_env])
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./models/expert_optimal/',
        log_path='./logs/expert_optimal/', 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )
    
    print("Initializing PPO Model...")
    # Initialize PPO model
    # PPO is a strong default for continuous control.
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
        ent_coef=0.0,
        tensorboard_log="./logs/tensorboard/"
    )
    
    # Set training timesteps. For testing, we can use a small number.
    # For actual training, recommend 1,000,000 to 5,000,000 for a decent expert.
    # We use a small number by default here to ensure the script runs, 
    # but this should be changed for the final run.
    total_timesteps = 100000 
    
    print(f"Starting training for {total_timesteps} timesteps...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving current model state...")
        
    # Save the final model
    os.makedirs("./models", exist_ok=True)
    model.save("./models/expert_optimal_final")
    print("Training complete and model saved.")
    
    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()

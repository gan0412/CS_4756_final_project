import os
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

def record_simulation(model_path, video_folder):
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create the environment with rgb_array render mode
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    
    # Wrap it with RecordVideo
    env = RecordVideo(env, video_folder=video_folder, name_prefix="simulation", episode_trigger=lambda x: True)
    
    print("Recording 1 episode...")
    obs, info = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        
    print(f"Episode Reward: {episode_reward:.2f}")
    env.close()
    print(f"Video saved to {video_folder}")

if __name__ == "__main__":
    model_path = "./models/expert_optimal_final"
    # Save the video directly to the artifacts directory so we can embed it
    video_folder = "/Users/Ganes/.gemini/antigravity/brain/03354b3f-6dcb-4b14-9f47-01e59be5a11d/artifacts/"
    os.makedirs(video_folder, exist_ok=True)
    record_simulation(model_path, video_folder)

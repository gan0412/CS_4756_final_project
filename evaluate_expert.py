import gymnasium as gym
from stable_baselines3 import PPO

def evaluate_model(model_path, num_episodes=5):
    print(f"Loading model from {model_path}...")
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Initialize the environment
    env = gym.make("BipedalWalker-v3")
    
    total_rewards = []
    
    print(f"Evaluating for {num_episodes} episodes...")
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Predict the action based on the observation
            # deterministic=True ensures we use the optimal action, not sampling
            action, _states = model.predict(obs, deterministic=True)
            
            # Take the step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            done = terminated or truncated
            
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    # Change this path to point to whichever expert you want to test!
    model_path = "./models/expert_optimal_final"
    evaluate_model(model_path)

import gymnasium as gym
from stable_baselines3 import PPO
import os
import json
import numpy as np

def evaluate_model(policy, num_episodes=20):
    print(f"Evaluating for {num_episodes} episodes...")
    
    # Initialize the environment
    env = gym.make("BipedalWalker-v3")
    
    total_rewards = []
    episode_lengths = []
    successes = 0  # BipedalWalker: reward >= 300 is typically considered success
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Predict the action based on the observation
            # deterministic=True ensures we use the optimal action, not sampling
            action, _states = policy.predict(obs, deterministic=True)
            
            # Take the step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            done = terminated or truncated
            
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        if episode_reward >= 300:
            successes += 1
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()
    
    # Compute statistics
    rewards = np.array(total_rewards)
    results = {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "median_reward": float(np.median(rewards)),
        "success_rate": successes / num_episodes,
        "mean_length": float(np.mean(episode_lengths)),
        "all_rewards": [float(r) for r in rewards],
        "num_episodes": num_episodes,
    }
    
    print(f"\nAverage Reward over {num_episodes} episodes: {results['mean_reward']:.2f} (+/- {results['std_reward']:.2f})")
    print(f"Success Rate: {results['success_rate']*100:.1f}%")
    print(f"Min / Max: {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    
    return results


def evaluate_all(models_config):
    """Evaluate multiple models and save combined results to JSON."""
    all_results = {}
    
    for cfg in models_config:
        name = cfg["name"]
        # Load the right type of policy
        if cfg["type"] == "ppo":
            policy = PPO.load(cfg["path"])
        elif cfg["type"] == "bc":
            from behavioral_cloning import load_bc_policy
            policy = load_bc_policy(cfg["path"])
        all_results[name] = evaluate_model(policy)
    
    # Save results
    os.makedirs("./results", exist_ok=True)
    save_path = "./results/evaluation_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {save_path}")
    
    return all_results


if __name__ == "__main__":
    # Can add more entries here as we train more methods.
    # models = [
    #     {"name": "PPO (vanilla)", "path": "./models/expert_optimal_final", "type": "ppo"},
    #     {"name": "PPO (cautious)", "path": "./models/expert_cautious_final", "type": "ppo"},
    #     {"name": "PPO (speed demon)", "path": "./models/expert_speed_demon_final", "type": "ppo"},
    #     {"name": "BC",            "path": "./models/bc_policy.pt",         "type": "bc"},
    # ]
    models = [
        {"name": "PPO (optimal)",     "path": "./models/expert_optimal_final",   "type": "ppo"},
        {"name": "BC (optimal)",      "path": "./models/bc_optimal.pt",          "type": "bc"},
        {"name": "DAgger (optimal)",  "path": "./models/dagger_optimal_seed0.pt","type": "bc"},
    ]
    
    # Makes sure to evaluate models that actually exist
    available = []
    for m in models:
        path = m["path"]
        # SB3 models save as .zip
        if os.path.exists(path) or os.path.exists(path + ".zip"):
            available.append(m)
        else:
            print(f"model not found at {path}")

    if available:
        evaluate_all(available)
    else:
        print("No trained models found.")
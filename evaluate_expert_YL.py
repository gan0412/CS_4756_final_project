import gymnasium as gym
from stable_baselines3 import PPO
import os
import json
import numpy as np

def evaluate_model(policy, num_episodes=100):
    print(f"Evaluating for {num_episodes} episodes...")
    
    env = gym.make("BipedalWalker-v3")
    
    total_rewards = []
    episode_lengths = []
    successes = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _states = policy.predict(obs, deterministic=True)
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
    all_results = {}
    
    for cfg in models_config:
        name = cfg["name"]
        if cfg["type"] == "ppo":
            policy = PPO.load(cfg["path"])
        elif cfg["type"] == "bc":
            from behavioral_cloning import load_bc_policy
            policy = load_bc_policy(cfg["path"])
        all_results[name] = evaluate_model(policy)
    
    os.makedirs("./results", exist_ok=True)
    save_path = "./results/evaluation_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {save_path}")
    
    return all_results


if __name__ == "__main__":
    models = [
        # PPO experts
        {"name": "PPO (optimal)",          "path": "./models/expert_optimal_final",            "type": "ppo"},
        {"name": "PPO (cautious)",         "path": "./models/expert_cautious_final",           "type": "ppo"},
        {"name": "PPO (speed_demon)",      "path": "./models/expert_speed_demon_final",        "type": "ppo"},

        # Phase 1: 主实验 (50 demos, seed 0)
        {"name": "BC (optimal)",           "path": "./models/bc_optimal.pt",                   "type": "bc"},
        {"name": "DAgger (optimal s0)",    "path": "./models/dagger_optimal_seed0.pt",         "type": "bc"},
        {"name": "BC (cautious)",          "path": "./models/bc_cautious.pt",                  "type": "bc"},
        {"name": "DAgger (cautious s0)",   "path": "./models/dagger_cautious_seed0.pt",        "type": "bc"},
        {"name": "BC (speed_demon)",       "path": "./models/bc_speed_demon.pt",               "type": "bc"},
        {"name": "DAgger (speed_demon s0)","path": "./models/dagger_speed_demon_seed0.pt",     "type": "bc"},

        # Phase 2: Ablation (5/10 demos, seed 0)
        {"name": "BC (5 demos)",           "path": "./models/bc_optimal_5demo.pt",             "type": "bc"},
        {"name": "DAgger (5 demos s0)",    "path": "./models/dagger_optimal_5demo_seed0.pt",   "type": "bc"},
        {"name": "BC (10 demos)",          "path": "./models/bc_optimal_10demo.pt",            "type": "bc"},
        {"name": "DAgger (10 demos s0)",   "path": "./models/dagger_optimal_10demo_seed0.pt",  "type": "bc"},

        # Phase 4: Multi-seed
        {"name": "DAgger (optimal s1)",    "path": "./models/dagger_optimal_seed1.pt",         "type": "bc"},
        {"name": "DAgger (optimal s2)",    "path": "./models/dagger_optimal_seed2.pt",         "type": "bc"},
        {"name": "DAgger (cautious s1)",   "path": "./models/dagger_cautious_seed1.pt",        "type": "bc"},
        {"name": "DAgger (cautious s2)",   "path": "./models/dagger_cautious_seed2.pt",        "type": "bc"},
        {"name": "DAgger (5 demos s1)",    "path": "./models/dagger_optimal_5demo_seed1.pt",   "type": "bc"},
        {"name": "DAgger (5 demos s2)",    "path": "./models/dagger_optimal_5demo_seed2.pt",   "type": "bc"},
        {"name": "DAgger (10 demos s1)",   "path": "./models/dagger_optimal_10demo_seed1.pt",  "type": "bc"},
        {"name": "DAgger (10 demos s2)",   "path": "./models/dagger_optimal_10demo_seed2.pt",  "type": "bc"},
    ]
    
    # Makes sure to evaluate models that actually exist
    available = []
    for m in models:
        path = m["path"]
        if os.path.exists(path) or os.path.exists(path + ".zip"):
            available.append(m)
        else:
            print(f"model not found at {path}")

    if available:
        evaluate_all(available)
    else:
        print("No trained models found.")
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
from stable_baselines3 import PPO
from behavioral_cloning import BCPolicy


# Helpers 
def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_expert_demos(expert, env, n_episodes, seed):
    #Fallback bootstrap when initial-data file does not exist yet
    states, actions, ep_rewards = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = expert.predict(obs, deterministic=True)
            states.append(obs.copy())
            actions.append(action.copy())
            obs, r, term, trunc, _ = env.step(action)
            ep_reward += r
            done = term or trunc
        ep_rewards.append(ep_reward)
    print(f"  expert mean reward over {n_episodes} eps: {np.mean(ep_rewards):.2f}")
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)


def collect_student_rollouts(policy, env, n_rollouts, device):
    states, ep_rewards = [], []
    policy.eval()
    for _ in range(n_rollouts):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            states.append(obs.copy())
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                action = policy(obs_t).cpu().numpy()[0]
            obs, r, term, trunc, _ = env.step(action)
            ep_reward += r
            done = term or trunc
        ep_rewards.append(ep_reward)
    return (
        np.array(states, dtype=np.float32),
        np.array(ep_rewards, dtype=np.float32),
    )


def relabel_with_expert(states, expert):
    actions = []
    for s in states:
        a, _ = expert.predict(s, deterministic=True)
        actions.append(a)
    return np.array(actions, dtype=np.float32)


def train_bc_on_dataset(policy, states, actions, epochs, lr, batch_size, device):
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dataset = TensorDataset(
        torch.from_numpy(states).float(),
        torch.from_numpy(actions).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    epoch_losses = []
    for _ in range(epochs):
        running = 0.0
        for batch_s, batch_a in loader:
            batch_s = batch_s.to(device)
            batch_a = batch_a.to(device)
            pred = policy(batch_s)
            loss = loss_fn(pred, batch_a)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        epoch_losses.append(running / len(loader))
    return epoch_losses


def evaluate_policy(policy, env, n_episodes, device):
    policy.eval()
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                action = policy(obs_t).cpu().numpy()[0]
            obs, r, term, trunc, _ = env.step(action)
            ep_reward += r
            done = term or trunc
        rewards.append(ep_reward)
    return float(np.mean(rewards)), float(np.std(rewards))


def save_checkpoint(policy, path, obs_dim, act_dim):
    torch.save({
        "state_dict": policy.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
    }, path)


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="BipedalWalker-v3")
    parser.add_argument(
        "--expert-path",
        type=str,
        default="./models/expert_optimal_final",
        help="Path to PPO expert (no .zip suffix needed).",
    )
    parser.add_argument(
        "--initial-data",
        type=str,
        default="./data/expert_demos.npz",
        help="Initial expert demonstrations. If missing, collected automatically.",
    )
    parser.add_argument("--bootstrap-episodes", type=int, default=50,
                        help="If --initial-data missing, collect this many expert eps.")
    parser.add_argument("--output-dir", type=str, default="./models")
    parser.add_argument("--history-dir", type=str, default="./logs")
    parser.add_argument("--tag", type=str, default="optimal",
                        help="Tag for output filenames, e.g. 'optimal', 'cautious', 'speed_demon'.")
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--rollouts-per-iter", type=int, default=10)
    parser.add_argument("--epochs-per-iter", type=int, default=20)
    parser.add_argument("--initial-epochs", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.history_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.initial_data) or ".", exist_ok=True)

    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, seed: {args.seed}, tag: {args.tag}")

    train_env = gym.make(args.env)
    train_env.reset(seed=args.seed)
    train_env.action_space.seed(args.seed)
    eval_env = gym.make(args.env)
    eval_env.reset(seed=args.seed + 10_000)

    print(f"Loading PPO expert from {args.expert_path}")
    expert = PPO.load(args.expert_path)

    if os.path.exists(args.initial_data):
        print(f"Loading initial demos from {args.initial_data}")
        data = np.load(args.initial_data)
        if "observations" in data:
            all_states = data["observations"].astype(np.float32)
        elif "obs" in data:
            all_states = data["obs"].astype(np.float32)
        elif "states" in data:
            all_states = data["states"].astype(np.float32)
        else:
            raise KeyError(
                f"Expected 'observations'/'obs'/'states' in {args.initial_data}, "
                f"got {list(data.keys())}"
            )
        all_actions = data["actions"].astype(np.float32)
    else:
        print(f"{args.initial_data} not found. Bootstrapping with "
              f"{args.bootstrap_episodes} expert episodes.")
        all_states, all_actions = collect_expert_demos(
            expert, train_env, args.bootstrap_episodes, seed=args.seed
        )
        np.savez(args.initial_data, observations=all_states, actions=all_actions)
        print(f"Saved bootstrap demos to {args.initial_data}")

    print(f"Initial dataset size: {len(all_states)}")
    obs_dim = all_states.shape[1]
    act_dim = all_actions.shape[1]

    policy = BCPolicy(obs_dim, act_dim).to(device)

    print("\nIteration 0: initial BC on expert demos")
    losses = train_bc_on_dataset(
        policy, all_states, all_actions,
        args.initial_epochs, args.lr, args.batch_size, device,
    )
    eval_mean, eval_std = evaluate_policy(policy, eval_env, args.eval_episodes, device)
    print(f"  final loss: {losses[-1]:.4f}   eval reward: {eval_mean:.2f} +/- {eval_std:.2f}")

    history = {
        "iter": [0],
        "dataset_size": [len(all_states)],
        "eval_mean": [eval_mean],
        "eval_std": [eval_std],
        "final_loss": [losses[-1]],
    }

    best_reward = eval_mean
    best_path = os.path.join(args.output_dir, f"dagger_{args.tag}_seed{args.seed}.pt")
    save_checkpoint(policy, best_path, obs_dim, act_dim)
    print(f"  saved initial checkpoint -> {best_path}")

    for it in range(1, args.n_iterations + 1):
        print(f"\n=== Iteration {it}/{args.n_iterations} ===")
        new_states, train_rewards = collect_student_rollouts(
            policy, train_env, args.rollouts_per_iter, device
        )
        print(f"  collected {len(new_states)} new states "
              f"(student mean reward: {np.mean(train_rewards):.2f})")

        new_actions = relabel_with_expert(new_states, expert)

        all_states = np.concatenate([all_states, new_states], axis=0)
        all_actions = np.concatenate([all_actions, new_actions], axis=0)
        print(f"  aggregated dataset size: {len(all_states)}")

        losses = train_bc_on_dataset(
            policy, all_states, all_actions,
            args.epochs_per_iter, args.lr, args.batch_size, device,
        )

        eval_mean, eval_std = evaluate_policy(policy, eval_env, args.eval_episodes, device)
        print(f"  final loss: {losses[-1]:.4f}   eval reward: {eval_mean:.2f} +/- {eval_std:.2f}")

        history["iter"].append(it)
        history["dataset_size"].append(len(all_states))
        history["eval_mean"].append(eval_mean)
        history["eval_std"].append(eval_std)
        history["final_loss"].append(losses[-1])

        if eval_mean > best_reward:
            best_reward = eval_mean
            save_checkpoint(policy, best_path, obs_dim, act_dim)
            print(f"  -> new best ({best_reward:.2f}), saved to {best_path}")

    history_path = os.path.join(args.history_dir, f"dagger_{args.tag}_history_seed{args.seed}.npz")
    np.savez(history_path, **{k: np.array(v) for k, v in history.items()})

    print(f"\nDone. Best eval reward: {best_reward:.2f}")
    print(f"Best policy: {best_path}")
    print(f"History:     {history_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
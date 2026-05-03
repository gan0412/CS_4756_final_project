import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=str, default="./logs")
    parser.add_argument("--pattern", type=str, default="dagger_history_seed*.npz")
    parser.add_argument("--output", type=str, default="./plots/dagger_curve.png")
    parser.add_argument(
        "--expert-reward", type=float, default=None,
        help="Optional horizontal expert reference line.",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.logs_dir, args.pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No history files matching {args.pattern} in {args.logs_dir}"
        )
    print(f"Loaded {len(paths)} history files:")
    for p in paths:
        print(f"  {p}")

    # Stack histories. Assumes all runs have the same iteration schedule.
    iters = None
    rewards = []  
    for p in paths:
        h = np.load(p)
        if iters is None:
            iters = h["iter"]
        rewards.append(h["eval_mean"])
    rewards = np.stack(rewards, axis=0)

    mean = rewards.mean(axis=0)
    std = rewards.std(axis=0)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Per-seed thin lines
    for i, r in enumerate(rewards):
        ax.plot(iters, r, color="C0", alpha=0.25, linewidth=1,
                label="individual seeds" if i == 0 else None)

 
    ax.plot(iters, mean, color="C0", linewidth=2.2, label=f"DAgger mean (n={len(paths)})")
    ax.fill_between(iters, mean - std, mean + std, color="C0", alpha=0.18)

    # Expert reference
    if args.expert_reward is not None:
        ax.axhline(args.expert_reward, color="C3", linestyle="--",
                   linewidth=1.5, label=f"Expert ({args.expert_reward:.1f})")


    ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.annotate("iter 0\n(pure BC)", xy=(0, ax.get_ylim()[0]),
                xytext=(0.3, mean[0]), fontsize=8, color="gray")

    ax.set_xlabel("DAgger iteration")
    ax.set_ylabel("Eval reward (mean episodic return)")
    ax.set_title("DAgger learning curve on BipedalWalker-v3")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nSaved figure -> {args.output}")

    print("\nIter | mean reward (across seeds)")
    print("-" * 35)
    for it, m, s in zip(iters, mean, std):
        print(f"{it:>4} | {m:>7.2f} +/- {s:>5.2f}")


if __name__ == "__main__":
    main()

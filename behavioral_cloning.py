import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class BCPolicy(nn.Module):
    # A simple MLP policy for BipedalWalker continuous control
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Two hidden layers (64 units each), we use tanh activations for nonlinearity
        # Final Tanh clamps actions to [-1, 1] to match BipedalWalker's action space
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
            nn.Tanh(),
        )
        self.obs_dim = obs_dim
        self.act_dim = act_dim
    
    def forward(self, x):
        return self.net(x)
    
    def predict(self, obs, deterministic=True):
        # Matching the predict functions for model eval.
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float32)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            action = self.net(obs).squeeze(0).numpy()
        return action, None


def train_bc(data_path="./data/expert_demos.npz", epochs=50):
    print(f"Loading expert data from {data_path}...")
    data = np.load(data_path)
    observations = data["observations"]
    actions = data["actions"]
    
    obs_dim = observations.shape[1]
    act_dim = actions.shape[1]
    n = len(observations)
    
    # 90/10 train/val split
    indices = np.random.permutation(n)
    split = int(n * 0.9)
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(observations[train_idx], dtype=torch.float32),
            torch.tensor(actions[train_idx], dtype=torch.float32),
        ),
        batch_size=256, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(observations[val_idx], dtype=torch.float32),
            torch.tensor(actions[val_idx], dtype=torch.float32),
        ),
        batch_size=256,
    )
    
    policy = BCPolicy(obs_dim, act_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    history = {"train_loss": [], "val_loss": []}
    print(f"Training BC policy: {n} samples, {epochs} epochs...")

    for epoch in range(epochs):
        # Training model
        policy.train()
        train_losses = []
        for obs_batch, act_batch in train_loader:
            pred = policy(obs_batch)
            loss = loss_fn(pred, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validating
        policy.eval()
        val_losses = []
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                pred = policy(obs_batch)
                loss = loss_fn(pred, act_batch)
                val_losses.append(loss.item())
        
        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs} -- "
                  f"train loss: {avg_train:.5f}, val loss: {avg_val:.5f}")
    
    # Save the policy
    save_path = "./models/bc_policy.pt"
    os.makedirs("./models", exist_ok=True)
    torch.save({
        "state_dict": policy.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
    }, save_path)
    print(f"BC policy saved to {save_path}")
    
    return policy, history


def load_bc_policy(path="./models/bc_policy.pt"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    policy = BCPolicy(checkpoint["obs_dim"], checkpoint["act_dim"])
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy


if __name__ == "__main__":
    policy, history = train_bc()
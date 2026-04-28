import gymnasium as gym

def main():
    print("Initializing BipedalWalker-v3 environment...")
    # Initialize the environment
    # Using 'rgb_array' so it doesn't try to pop open a window on a headless setup
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    
    # Reset the environment to start
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Take a few random steps
    print("Taking 10 random steps...")
    for i in range(10):
        # Sample a random action from the action space
        action = env.action_space.sample()
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode ended early at step {i}")
            break
            
    print("Successfully ran 10 steps!")
    env.close()

if __name__ == "__main__":
    main()

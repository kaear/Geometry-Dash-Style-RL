"""Train a first PPO policy on the fixed beginner level."""
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from geometry_env import GeometryDashEnv

if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    env = Monitor(GeometryDashEnv(reward_mode="sparse"))
    model = PPO("MlpPolicy", env, seed=42, verbose=1, tensorboard_log=None)
    model.learn(total_timesteps=50_000)
    model.save("models/ppo_fixed_level")
    print("Saved models/ppo_fixed_level.zip")

# training/train_interactive.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from training.env import TronEnv
from training.model import TronTransformer
import torch as th

if __name__ == "__main__":
    env = DummyVecEnv([lambda: TronEnv()])

    policy_kwargs = dict(
        features_extractor_class=TronTransformer,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[],
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=256,  # Small for fast rollouts (fits 1-2 episodes)
        batch_size=64,
        learning_rate=3e-4,
        device="cpu",
        tensorboard_log="./logs/"
    )
    print("Starting training...")
    model.learn(total_timesteps=5_000_000)
    model.save("../models/interactive_final")
    th.save(model.policy.state_dict(), "../models/interactive_final.pt")
    print("Training complete")
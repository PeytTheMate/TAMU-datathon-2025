# training/env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game.case_closed_game import Game, Direction, GameResult
from training.utils import state_to_tensor
import random

DIR_MAP = {0: Direction.UP, 1: Direction.DOWN, 2: Direction.LEFT, 3: Direction.RIGHT}

class TronEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.game = Game()
        self.turns = 0
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10, 18, 20), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.game.reset()
        self.turns = 0
        obs = self._get_obs()
        print(f"[DEBUG RESET] Obs shape: {obs.shape}, turns: 0")
        return obs, {}

    def _get_obs(self):
        state_dict = {
            'board': self.game.board.grid,
            'agent1_trail': self.game.agent1.get_trail_positions(),
            'agent2_trail': self.game.agent2.get_trail_positions(),
            'agent1_boosts': self.game.agent1.boosts_remaining,
            'agent2_boosts': self.game.agent2.boosts_remaining,
            'turn_count': self.game.turns,
        }
        return state_to_tensor(state_dict).numpy()

    def step(self, action):
        self.turns += 1
        print(f"[DEBUG STEP] Turn {self.turns}, action {action}, agent1.alive {self.game.agent1.alive}, agent2.alive {self.game.agent2.alive}")

        # Force end after 100 turns
        if self.turns >= 100:
            print("[DEBUG] Force end at max turns")
            reward = 50 if self.game.agent1.length > self.game.agent2.length else -50
            return self._get_obs(), reward, True, False, {"result": "max_turns"}

        # Check if dead
        if not self.game.agent1.alive or not self.game.agent2.alive:
            result = GameResult.DRAW if not self.game.agent1.alive and not self.game.agent2.alive else GameResult.AGENT2_WIN if not self.game.agent1.alive else GameResult.AGENT1_WIN
            reward = self._compute_reward(result)
            print(f"[DEBUG] Episode end: {result}, reward {reward}")
            return self._get_obs(), reward, True, False, {"result": result.name}

        # Player move
        dir_idx = action % 4
        boost = action >= 4 and self.game.agent1.boosts_remaining > 0
        dir1 = DIR_MAP[dir_idx]

        # Fixed opponent (always RIGHT to avoid invalid)
        opp_dir = Direction.RIGHT

        result = self.game.step(dir1, opp_dir, boost1=boost, boost2=False)

        reward = self._compute_reward(result)
        done = result is not None
        print(f"[DEBUG] After step: done {done}, reward {reward}")

        return self._get_obs(), reward, done, False, {}

    def _compute_reward(self, result):
        if result == GameResult.AGENT1_WIN: return 100
        if result == GameResult.AGENT2_WIN: return -100
        if result == GameResult.DRAW: return -50
        return 0.2 * (self.game.agent1.length - self.game.agent2.length) - 0.01
"""Minimal Geometry Dash-style Gymnasium environment for CS 175.
Original starter scaffold; intentionally simplified and independent of browser clone.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GeometryDashEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, reward_mode="sparse", obstacles=None, max_steps=350):
        super().__init__()
        if reward_mode not in ("sparse", "shaped"):
            raise ValueError("reward_mode must be sparse or shaped")
        self.reward_mode = reward_mode
        self.speed = 0.15
        self.gravity = 0.025
        self.jump_speed = 0.37
        self.level_end = 25.0
        self.max_steps = max_steps
        # (front position, obstacle width, obstacle height) in abstract units
        self.obstacles = obstacles or [(6.0, 0.7, 0.6), (12.0, 0.7, 0.6), (18.0, 0.7, 0.6)]
        self.action_space = spaces.Discrete(2)  # 0=wait, 1=jump if grounded
        # [player height, vertical velocity, grounded, next two obstacle distances/widths/heights]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
        self.x = self.y = self.vy = 0.0
        self.steps = 0

    def _obs(self):
        upcoming = [(pos, width, height) for pos, width, height in self.obstacles
                    if pos + width >= self.x]
        features = []
        for pos, width, height in upcoming[:2]:
            features += [(pos - self.x) / 25.0, width, height]
        while len(features) < 6:
            features += [1.0, 0.0, 0.0]
        return np.asarray([self.y, self.vy, float(self.y <= 0.0)] + features,
                          dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x = self.y = self.vy = 0.0
        self.steps = 0
        return self._obs(), {"progress": 0.0, "success": False}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if int(action) == 1 and self.y <= 0.0:
            self.vy = self.jump_speed
        self.vy -= self.gravity
        self.y = max(0.0, self.y + self.vy)
        if self.y == 0.0:
            self.vy = 0.0
        previous_x = self.x
        self.x += self.speed
        self.steps += 1
        collided = any(self.x >= pos and previous_x <= pos + width and self.y < height
                       for pos, width, height in self.obstacles)
        success = self.x >= self.level_end and not collided
        terminated = bool(collided or success)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        reward = 1.0 if success else -1.0 if collided else 0.0
        if self.reward_mode == "shaped" and not terminated:
            reward += 0.01 * (self.speed / self.speed)  # fixed reference distance = speed
        info = {"progress": min(self.x / self.level_end, 1.0),
                "success": success, "collision": collided, "steps": self.steps}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        return f"x={self.x:.2f} y={self.y:.2f} progress={100*self.x/self.level_end:.1f}%"

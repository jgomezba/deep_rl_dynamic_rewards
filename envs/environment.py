import random, time
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

class GridWorld:
    def __init__(
        self, 
        grid_size=5, 
        n_gems=3, 
        start=(0,0), 
        max_steps=50, 
        reward_step=-0.1, 
        reward_gem=1.0, 
        reward_goal=10.0, 
        reward_invalid=-1.0,
        actions=None
    ):
        self.G = grid_size
        self.n_gems = n_gems
        self.start = start
        self.goal = (grid_size-1, grid_size-1)
        self.MAX_STEPS = max_steps
        self.REWARD_STEP = reward_step
        self.REWARD_GEM = reward_gem
        self.REWARD_GOAL = reward_goal
        self.REWARD_INVALID = reward_invalid
        self.ACTIONS = actions if actions is not None else [(0,1),(0,-1),(-1,0),(1,0)]
        self.reset()

    def _random_gems(self):
        pos = set()
        while len(pos) < self.n_gems:
            p = (random.randint(0,self.G-1), random.randint(0,self.G-1))
            if p != self.start and p != self.goal:
                pos.add(p)
        return set(pos)

    def reset(self):
        self.agent = self.start
        self.gems = self._random_gems()
        self.collected = 0
        self.t = 0
        return self._obs()

    def step(self, action):
        dx, dy = self.ACTIONS[action]
        x, y = self.agent
        nx, ny = x + dx, y + dy
        self.t += 1

        if nx < 0 or nx >= self.G or ny < 0 or ny >= self.G:
            reward = self.REWARD_STEP + self.REWARD_INVALID
            done = False
            return self._obs(), reward, done, {}

        self.agent = (nx, ny)
        reward = self.REWARD_STEP

        if self.agent in self.gems:
            self.gems.remove(self.agent)
            self.collected += 1
            reward += self.REWARD_GEM

        done = False

        if self.agent == self.goal:
            if self.collected == self.n_gems:
                reward += self.REWARD_GOAL
                done = True
            else:
                reward -= 2 

        if self.t >= self.MAX_STEPS:
            done = True

        return self._obs(), reward, done, {}

    def _obs(self):
        obs = np.zeros((3, self.G, self.G), dtype=np.float32)
        for (i,j) in self.gems:
            obs[0, i, j] = 1.0
        gi, gj = self.goal
        obs[1, gi, gj] = 1.0
        ai, aj = self.agent
        obs[2, ai, aj] = 1.0
        return obs

    def render(self, sleep=0.25, title=""):
        grid = np.zeros((self.G, self.G))
        for (i,j) in self.gems:
            grid[i,j] = 1
        gi,gj = self.goal
        grid[gi,gj] = 2
        ai,aj = self.agent
        grid[ai,aj] = 3
        clear_output(wait=True)
        plt.imshow(grid, cmap="coolwarm", vmin=0, vmax=3)
        plt.title(title)
        plt.show()
        time.sleep(sleep)
from collections import deque
import random

import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, s2, d):
        self.buf.append( (s, a, r, s2, d) )
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        s  = torch.tensor(np.array(s),  dtype=torch.float32, device=device)
        a  = torch.tensor(a,            dtype=torch.long,    device=device)
        r  = torch.tensor(r,            dtype=torch.float32, device=device)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=device)
        d  = torch.tensor(d,            dtype=torch.float32, device=device)
        return s, a, r, s2, d
    def __len__(self):
        return len(self.buf)
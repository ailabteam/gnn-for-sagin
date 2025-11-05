# File: src/poc2_routing/sagin_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch_geometric.data import Data

DATA_PATH = 'data/processed/sagin_simulation_dataset.pt'

class SaginRoutingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_hops=16):
        super(SaginRoutingEnv, self).__init__()
        print("Initializing SAGIN Routing Environment...")
        self.all_snapshots = torch.load(DATA_PATH)
        print(f"Loaded {len(self.all_snapshots)} graph snapshots.")
        self.max_hops = max_hops
        self.action_space = spaces.Discrete(10) # Max 10 neighbors
        # Observation space sẽ được xác định trong reset
        self.observation_space = None

    def _get_observation(self):
        graph = self.current_snapshot
        num_nodes = graph.num_nodes
        
        # Node features gốc là tọa độ xyz
        original_features = graph.x
        
        # Tạo 2 cột đặc trưng mới
        additional_features = torch.zeros(num_nodes, 2, dtype=torch.float32)
        additional_features[self.current_node, 0] = 1.0
        additional_features[self.destination_node, 1] = 1.0

        # Nối chúng lại
        final_features = torch.cat([original_features, additional_features], dim=1)
        
        return Data(x=final_features, edge_index=graph.edge_index, edge_attr=graph.edge_attr)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_snapshot_idx = self.np_random.integers(0, len(self.all_snapshots))
        self.current_snapshot = self.all_snapshots[self.current_snapshot_idx]
        
        num_nodes = self.current_snapshot.num_nodes
        nodes = list(range(num_nodes))
        self.source_node, self.destination_node = self.np_random.choice(nodes, size=2, replace=False)
        
        self.current_node = self.source_node
        self.hop_count = 0
        self.path = [self.current_node]

        observation = self._get_observation()
        info = {"source": self.source_node, "destination": self.destination_node}
        
        # Cập nhật observation space nếu chưa có
        if self.observation_space is None:
            self.observation_space = spaces.Dict({
                "x": spaces.Box(low=-np.inf, high=np.inf, shape=observation.x.shape, dtype=np.float32),
                "edge_index": spaces.Box(low=0, high=num_nodes-1, shape=observation.edge_index.shape, dtype=np.int64),
                "edge_attr": spaces.Box(low=0, high=np.inf, shape=observation.edge_attr.shape, dtype=np.float32)
            })

        return observation, info

    def step(self, action):
        neighbors = self.current_snapshot.edge_index[1, self.current_snapshot.edge_index[0] == self.current_node]
        
        terminated = False
        truncated = False
        reward = 0.0

        if action >= len(neighbors):
            reward = -10.0
            truncated = True
        else:
            next_node = neighbors[action].item()
            
            # Phạt nhỏ cho mỗi bước nhảy
            reward = -0.1 
            
            # REWARD SHAPING
            pos = self.current_snapshot.x
            dist_before = torch.linalg.norm(pos[self.current_node] - pos[self.destination_node])
            dist_after = torch.linalg.norm(pos[next_node] - pos[self.destination_node])
            shaping_reward = (dist_before - dist_after) / 1000.0
            reward += shaping_reward
            
            self.current_node = next_node
            self.path.append(self.current_node)
            self.hop_count += 1

            if self.current_node == self.destination_node:
                reward += 10.0
                terminated = True
            elif self.hop_count >= self.max_hops:
                reward -= 5.0
                truncated = True
        
        observation = self._get_observation()
        info = {"path": self.path}

        return observation, reward, terminated, truncated, info

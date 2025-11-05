import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx

from .sagin_env import SaginRoutingEnv
from .agent import GNNAgent

# --- Cấu hình Huấn luyện ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPISODES = 10000
MAX_HOPS = 16
GAMMA = 0.99
LOG_INTERVAL = 100

# --- Cấu hình Model ---
# Node features giờ là: [x, y, z, is_current, is_destination]
NODE_IN_CHANNELS = 5 
GNN_HIDDEN_CHANNELS = 64
LEARNING_RATE = 1e-4

def calculate_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

def train():
    print(f"Starting training on {DEVICE}...")
    env = SaginRoutingEnv(max_hops=MAX_HOPS)
    agent = GNNAgent(
        in_channels=NODE_IN_CHANNELS,
        hidden_channels=GNN_HIDDEN_CHANNELS,
        lr=LEARNING_RATE,
        device=DEVICE
    )

    episode_rewards = []
    episode_hops = []
    all_baseline_hops = []

    progress_bar = tqdm(range(NUM_EPISODES))
    for i_episode in progress_bar:
        state, info = env.reset()
        saved_log_probs = []
        rewards = []
        
        for t in range(MAX_HOPS):
            action, log_prob = agent.select_action(state)
            if action is None: break
            state, reward, terminated, truncated, _ = env.step(action)
            saved_log_probs.append(log_prob)
            rewards.append(reward)
            if terminated or truncated: break
        
        episode_rewards.append(sum(rewards))
        episode_hops.append(len(env.path) - 1)

        if not saved_log_probs: continue
        
        returns = calculate_returns(rewards, GAMMA).to(DEVICE)
        
        baseline = returns.mean()
        advantages = returns - baseline
        
        policy_loss = []
        for log_prob, advantage in zip(saved_log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        
        agent.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        agent.optimizer.step()

        if (i_episode + 1) % LOG_INTERVAL == 0:
            snapshot = env.current_snapshot
            G = nx.Graph()
            # Dùng .x gốc (chỉ có xyz) để tính Dijkstra
            pos = snapshot.x.cpu().numpy()
            edges = snapshot.edge_index.t().cpu().numpy()
            for i, j in edges:
                dist = np.linalg.norm(pos[i] - pos[j])
                G.add_edge(i, j, weight=dist)
            try:
                path_dijkstra = nx.dijkstra_path(G, source=info['source'], target=info['destination'])
                all_baseline_hops.append(len(path_dijkstra) - 1)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                all_baseline_hops.append(np.nan)

            avg_reward = np.mean(episode_rewards[-LOG_INTERVAL:])
            avg_hops = np.mean(episode_hops[-LOG_INTERVAL:])
            avg_baseline = np.nanmean(all_baseline_hops) if all_baseline_hops else 'N/A'
            progress_bar.set_description(f'Episode {i_episode+1} | Avg Reward: {avg_reward:.2f} | Avg Hops: {avg_hops:.2f} (Dijkstra Avg: {avg_baseline:.2f})')

    env.close()

    # --- Vẽ đồ thị kết quả ---
    plt.figure(figsize=(12, 5))
    smoothed_rewards = [np.mean(episode_rewards[max(0, i-LOG_INTERVAL):i+1]) for i in range(len(episode_rewards))]
    plt.plot(smoothed_rewards, label='Smoothed Reward', linewidth=2)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend(); plt.grid(True)
    plt.savefig('poc2_rewards.png', dpi=600)
    plt.close()

    plt.figure(figsize=(12, 5))
    smoothed_hops = [np.mean(episode_hops[max(0, i-LOG_INTERVAL):i+1]) for i in range(len(episode_hops))]
    plt.plot(smoothed_hops, label='Agent Hops', linewidth=2)
    
    if all_baseline_hops:
        avg_dijkstra = np.nanmean(all_baseline_hops)
        plt.axhline(y=avg_dijkstra, color='r', linestyle='--', label=f'Avg Dijkstra Hops ({avg_dijkstra:.2f})')

    plt.title('Average Hops to Destination vs. Dijkstra Baseline')
    plt.xlabel('Episode')
    plt.ylabel('Number of Hops')
    plt.legend(); plt.grid(True)
    plt.savefig('poc2_hops_vs_dijkstra.png', dpi=600)
    plt.close()

if __name__ == '__main__':
    try:
        import networkx
    except ImportError:
        print("NetworkX not found. Installing...")
        import os
        os.system("pip install networkx")
    
    train()

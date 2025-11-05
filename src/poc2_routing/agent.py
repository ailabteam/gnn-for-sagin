import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch.distributions import Categorical

class GNNPolicy(nn.Module):
    """
    Mạng Chính sách (Policy Network) dựa trên GNN.
    """
    def __init__(self, in_channels, hidden_channels, heads=4):
        super(GNNPolicy, self).__init__()
        
        # GNN Encoder để học biểu diễn node
        # Chúng ta sẽ không sử dụng edge_attr trực tiếp trong GNN nữa
        # vì thông tin khoảng cách đã có trong tọa độ xyz của node_features.
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, add_self_loops=False)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, add_self_loops=False)
        
        # Một lớp tuyến tính để tính "điểm" cho mỗi node, dựa trên embedding của nó
        self.action_scorer = nn.Linear(hidden_channels, 1)

    def forward(self, state: Data):
        """
        Args:
            state (Data): Một object đồ thị PyG từ môi trường.
                          state.x có shape [num_nodes, in_channels].
        
        Returns:
            action_logits (torch.Tensor): Điểm số (logits) cho các hành động khả thi.
        """
        x, edge_index = state.x, state.edge_index
        
        # 1. Mã hóa toàn bộ đồ thị để có được node embeddings
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        
        # 2. Xác định node hiện tại và các hàng xóm của nó
        # Dùng argmax vì one-hot encoding
        current_node_idx = torch.argmax(state.x[:, 3]).item()
        neighbors_mask = (edge_index[0] == current_node_idx)
        neighbor_nodes = edge_index[1, neighbors_mask]
        
        if neighbor_nodes.numel() == 0:
            return torch.tensor([], device=x.device)

        # 3. Lấy embeddings của các node hàng xóm
        neighbor_embeddings = x[neighbor_nodes]
        
        # 4. Tính điểm cho mỗi hành động (mỗi hàng xóm)
        action_scores = self.action_scorer(neighbor_embeddings).squeeze(-1)
        
        return action_scores

class GNNAgent:
    """
    Tác tử RL chứa một mạng GNNPolicy.
    """
    def __init__(self, in_channels, hidden_channels, lr=1e-4, device='cpu'):
        self.device = device
        self.policy_net = GNNPolicy(in_channels, hidden_channels).to(self.device)
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr)

    def select_action(self, state: Data):
        """
        Chọn một hành động dựa trên trạng thái hiện tại.
        """
        # Chuyển toàn bộ object Data sang device
        state = state.to(self.device)
        
        action_scores = self.policy_net(state)
        
        if action_scores.numel() == 0:
            return None, None
        
        action_probs = F.softmax(action_scores, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class EdgeGNNEncoder(nn.Module):
    """
    Một module GNN để mã hóa một đồ thị DUY NHẤT.
    Nó nhận một đồ thị và trả về một vector biểu diễn (embedding) cho MỖI CẠNH.
    """
    def __init__(self, in_channels, hidden_channels, heads=4):
        super(EdgeGNNEncoder, self).__init__()
        # Các lớp GAT để học biểu diễn node
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        # 1. Mã hóa Node
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        
        # 2. Tạo biểu diễn Cạnh từ biểu diễn Node
        node_u = x[edge_index[0]]
        node_v = x[edge_index[1]]
        edge_embedding = torch.cat([node_u, node_v], dim=-1)
        
        return edge_embedding


class GNN_GRU_Model(nn.Module):
    """
    Kiến trúc lai GNN-GRU để dự đoán thuộc tính cạnh từ một chuỗi đồ thị.
    """
    def __init__(self, node_features_dim, gnn_hidden_dim, gru_hidden_dim, heads=4):
        super(GNN_GRU_Model, self).__init__()
        
        # --- Phần GNN ---
        # GNN Encoder sẽ được áp dụng cho từng đồ thị trong chuỗi
        self.gnn_encoder = EdgeGNNEncoder(
            in_channels=node_features_dim, 
            hidden_channels=gnn_hidden_dim, 
            heads=heads
        )
        
        # Kích thước output của GNN Encoder là gnn_hidden_dim * 2 (vì nối 2 node embedding)
        gnn_output_dim = gnn_hidden_dim * 2
        
        # --- Phần GRU ---
        # GRU nhận đầu vào là một chuỗi các vector biểu diễn cạnh
        self.gru = nn.GRU(
            input_size=gnn_output_dim, 
            hidden_size=gru_hidden_dim, 
            num_layers=2, # Sử dụng 2 lớp GRU để học các quy luật phức tạp hơn
            batch_first=True # Quan trọng: đầu vào sẽ có dạng [batch, seq_len, features]
        )
        
        # --- Phần MLP Decoder ---
        # MLP nhận đầu vào là trạng thái ẩn cuối cùng của GRU
        self.decoder = nn.Sequential(
            nn.Linear(gru_hidden_dim, gru_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(gru_hidden_dim // 2, 1) # Output là 1 giá trị (khoảng cách)
        )

    def forward(self, graph_sequence, common_edge_index):
        """
        Forward pass cho toàn bộ mô hình.
        Args:
            graph_sequence (list): Một list các object đồ thị PyG (đã được batch).
            common_edge_index (torch.Tensor): Tensor edge_index chung cho tất cả đồ thị.
        """
        # Số cạnh trong mỗi đồ thị của batch
        num_edges = common_edge_index.size(1)
        
        # --- Bước 1: Áp dụng GNN Encoder cho từng time step ---
        # Tạo một list để chứa các embedding của cạnh tại mỗi time step
        edge_embeddings_over_time = []
        for graph in graph_sequence:
            # `graph.x` chứa node features của tất cả đồ thị trong batch được ghép lại
            # `common_edge_index` là cấu trúc cạnh
            edge_embedding_t = self.gnn_encoder(graph.x, common_edge_index)
            edge_embeddings_over_time.append(edge_embedding_t)
            
        # Nối các embedding lại với nhau theo chiều thời gian (sequence length)
        # Output shape: [num_edges * batch_size, seq_len, gnn_output_dim]
        sequence_tensor = torch.stack(edge_embeddings_over_time, dim=1)

        # --- Bước 2: Truyền chuỗi embedding qua GRU ---
        # GRU sẽ xử lý chuỗi này như một batch lớn, trong đó mỗi cạnh là một sample
        # `_` là output của tất cả các time step, `h_n` là hidden state cuối cùng
        # h_n shape: [num_layers, num_edges * batch_size, gru_hidden_dim]
        _, h_n = self.gru(sequence_tensor)
        
        # Lấy hidden state của lớp GRU cuối cùng
        # Shape: [num_edges * batch_size, gru_hidden_dim]
        last_hidden_state = h_n[-1]

        # --- Bước 3: Đưa vào MLP Decoder để dự đoán ---
        # Shape: [num_edges * batch_size, 1]
        prediction = self.decoder(last_hidden_state)
        
        return prediction

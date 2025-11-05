import torch
from torch.utils.data import Dataset

class GraphSequenceDataset(Dataset):
    """
    Dataset cho bài toán dự đoán trên chuỗi đồ thị.
    Mỗi sample là một chuỗi các đồ thị và target là thuộc tính cạnh của đồ thị tiếp theo.
    - Input: Một chuỗi `sequence_length` đồ thị [G_t, G_{t+1}, ..., G_{t+N-1}]
    - Target: Thuộc tính cạnh (edge_attr) của đồ thị G_{t+N}
    """
    def __init__(self, graph_snapshots, sequence_length=4):
        self.snapshots = graph_snapshots
        self.seq_len = sequence_length
        # `__len__` sẽ là tổng số snapshot trừ đi độ dài chuỗi (cho input) và 1 (cho target)
        self.num_samples = len(self.snapshots) - self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Một sample bao gồm một chuỗi các đồ thị và một target.
        # Chuỗi input bắt đầu từ `idx` và kéo dài `self.seq_len`
        # Target là đồ thị ngay sau chuỗi input.
        
        input_sequence_graphs = self.snapshots[idx : idx + self.seq_len]
        target_graph = self.snapshots[idx + self.seq_len]
        
        # --- Xử lý Target ---
        # Chúng ta cần xác định các cạnh tồn tại trong đồ thị cuối cùng của chuỗi
        # và đồ thị target để có thể so sánh.
        
        last_graph_in_sequence = input_sequence_graphs[-1]
        
        edges_last = {tuple(sorted(edge)) for edge in last_graph_in_sequence.edge_index.t().tolist()}
        edges_target = {tuple(sorted(edge)) for edge in target_graph.edge_index.t().tolist()}
        
        common_edges = sorted(list(edges_last.intersection(edges_target)))
        
        if not common_edges:
            return None, None, None # Trả về None nếu không có cạnh chung

        # Lấy index của các cạnh chung trong đồ thị target
        edge_map_target = {tuple(sorted(edge)): i for i, edge in enumerate(target_graph.edge_index.t().tolist())}
        common_edge_indices_target = [edge_map_target[edge] for edge in common_edges]
        
        # Target tensor
        target_edge_attr = target_graph.edge_attr[torch.tensor(common_edge_indices_target, dtype=torch.long)]

        # --- Xử lý Input ---
        # Chúng ta cần đảm bảo tất cả các đồ thị trong chuỗi input đều có chung
        # một tập hợp cạnh (là `common_edges`) để đưa vào model một cách nhất quán.
        
        processed_sequence = []
        for graph in input_sequence_graphs:
            # Tạo một đồ thị mới chỉ chứa các cạnh chung
            new_graph = graph.clone()
            
            # Tạo map cho đồ thị hiện tại trong chuỗi
            edge_map_current = {tuple(sorted(edge)): i for i, edge in enumerate(graph.edge_index.t().tolist())}
            
            # Kiểm tra xem tất cả common_edges có tồn tại trong đồ thị này không
            current_edges_set = set(edge_map_current.keys())
            if not all(edge in current_edges_set for edge in common_edges):
                # Nếu một đồ thị trong chuỗi thiếu mất một cạnh chung, sample này không hợp lệ
                return None, None, None

            common_edge_indices_current = [edge_map_current[edge] for edge in common_edges]
            
            new_graph.edge_index = torch.tensor(common_edges, dtype=torch.long).t().contiguous()
            new_graph.edge_attr = graph.edge_attr[torch.tensor(common_edge_indices_current, dtype=torch.long)]
            
            processed_sequence.append(new_graph)

        # `common_edges` bây giờ là cấu trúc cạnh chung cho toàn bộ sample
        common_edge_index = torch.tensor(common_edges, dtype=torch.long).t().contiguous()

        return processed_sequence, common_edge_index, target_edge_attr


def sequence_collate_fn(batch):
    """
    Hàm collate tùy chỉnh cho dataset chuỗi.
    """
    # Lọc các sample không hợp lệ
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None, None
        
    # Tách các thành phần
    sequences, edge_indices, targets = zip(*batch)
    
    # Batch các chuỗi và target
    # Lưu ý: `sequences` là một list của các list đồ thị
    # `edge_indices` là một list các tensor edge_index
    # `targets` là một list các tensor target
    
    # Chúng ta sẽ xử lý việc batching các đồ thị bên trong vòng lặp training
    # để đơn giản hóa.
    
    return sequences, edge_indices, targets

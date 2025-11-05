import torch
from torch.utils.data import Dataset

class LinkPredictionDataset(Dataset):
    """
    Dataset cho bài toán dự đoán thuộc tính của cạnh (link) trong tương lai.
    Input: Đồ thị tại thời điểm t.
    Target: Thuộc tính của các cạnh (edge_attr) tại thời điểm t+1.
    """
    def __init__(self, graph_snapshots):
        self.snapshots = graph_snapshots

    def __len__(self):
        # Chúng ta cần t và t+1, nên độ dài sẽ là N-1
        return len(self.snapshots) - 1

    def __getitem__(self, idx):
        graph_t0 = self.snapshots[idx]
        graph_t1 = self.snapshots[idx + 1]
        
        # Tạo một "khóa" duy nhất cho mỗi cạnh (node_u, node_v)
        edges_t0 = {tuple(sorted(edge)) for edge in graph_t0.edge_index.t().tolist()}
        edges_t1 = {tuple(sorted(edge)) for edge in graph_t1.edge_index.t().tolist()}
        common_edges = sorted(list(edges_t0.intersection(edges_t1)))
        
        if not common_edges or len(common_edges) == 0:
            return None # Trả về None nếu không có cạnh chung

        # Lấy chỉ số và thuộc tính cho các cạnh chung
        edge_map_t0 = {tuple(sorted(edge)): i for i, edge in enumerate(graph_t0.edge_index.t().tolist()) if tuple(sorted(edge)) in edges_t0}
        edge_map_t1 = {tuple(sorted(edge)): i for i, edge in enumerate(graph_t1.edge_index.t().tolist()) if tuple(sorted(edge)) in edges_t1}

        # Lọc ra các cạnh thực sự tồn tại
        common_edges_in_t0 = [edge for edge in common_edges if edge in edge_map_t0]
        common_edges_in_t1 = [edge for edge in common_edges if edge in edge_map_t1]
        
        final_common_edges = sorted(list(set(common_edges_in_t0).intersection(set(common_edges_in_t1))))

        if not final_common_edges:
            return None
            
        common_edge_indices_t0 = [edge_map_t0[edge] for edge in final_common_edges]
        common_edge_indices_t1 = [edge_map_t1[edge] for edge in final_common_edges]

        input_graph = graph_t0.clone()
        input_graph.edge_index = torch.tensor(final_common_edges, dtype=torch.long).t().contiguous()
        input_graph.edge_attr = graph_t0.edge_attr[torch.tensor(common_edge_indices_t0, dtype=torch.long)]
        
        target_edge_attr = graph_t1.edge_attr[torch.tensor(common_edge_indices_t1, dtype=torch.long)]
        
        return input_graph, target_edge_attr

def collate_fn(batch):
    """Hàm tùy chỉnh để xử lý batch, loại bỏ các sample None."""
    batch = [(g, t) for g, t in batch if g is not None]
    if not batch:
        return None, None
    
    graphs, targets = zip(*batch)
    return list(graphs), torch.cat(targets, dim=0)

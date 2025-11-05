import numpy as np
from skyfield.api import load, EarthSatellite, Topos
from tqdm import tqdm
import torch
from torch_geometric.data import Data

# --- Constants ---
# Sử dụng TLE của một vài vệ tinh Starlink để mô phỏng
# Nguồn: celestrak.com
STARLINK_TLE = [
    ("STARLINK-1007",
     "1 44782U 19074A   24142.50000000  .00000523  00000+0  16946-4 0  9991",
     "2 44782  53.0541 129.2133 0001014 100.2195 259.8931 15.06379435252814"),
    ("STARLINK-1012",
     "1 44787U 19074F   24142.50000000  .00000624  00000+0  19782-4 0  9997",
     "2 44787  53.0537 130.6133 0001002  99.1678 260.9463 15.06378941252839"),
    ("STARLINK-1013",
     "1 44788U 19074G   24142.50000000  .00000572  00000+0  18119-4 0  9990",
     "2 44788  53.0538 131.0125 0001002  98.7118 261.4026 15.06378906252847"),
    ("STARLINK-1014",
     "1 44789U 19074H   24142.50000000  .00000508  00000+0  16489-4 0  9997",
     "2 44789  53.0539 131.4124 0001004  98.2415 261.8732 15.06378891252854"),
    ("STARLINK-1015",
     "1 44790U 19074J   24142.50000000  .00000609  00000+0  19363-4 0  9994",
     "2 44790  53.0537 131.8122 0001002  97.7778 262.3369 15.06378819252864")
]
# Giới hạn tầm nhìn (visibility range) để tạo kết nối giữa 2 vệ tinh (km)
VISIBILITY_THRESHOLD_KM = 1500

def create_satellite_objects(tle_list):
    """Tạo các đối tượng EarthSatellite từ danh sách TLE."""
    satellites = []
    for name, line1, line2 in tle_list:
        sat = EarthSatellite(line1, line2, name, load.timescale())
        satellites.append(sat)
    return satellites

def get_positions(satellites, time):
    """Lấy vị trí (x, y, z) của các vệ tinh tại một thời điểm."""
    positions = [sat.at(time).position.km for sat in satellites]
    return np.array(positions)

def create_graph_snapshot(positions, threshold_km):
    """Tạo một snapshot đồ thị tại một thời điểm."""
    num_sats = len(positions)
    
    # Node features: vị trí (x, y, z)
    node_features = torch.tensor(positions, dtype=torch.float)
    
    # Edge index and features:
    edge_index_list = []
    edge_attr_list = []
    
    for i in range(num_sats):
        for j in range(i + 1, num_sats):
            pos_i = positions[i]
            pos_j = positions[j]
            distance = np.linalg.norm(pos_i - pos_j)
            
            if distance <= threshold_km:
                # Thêm cạnh theo cả hai hướng
                edge_index_list.append([i, j])
                edge_index_list.append([j, i])
                # Edge feature: khoảng cách (1/distance^2 là proxy cho chất lượng)
                quality = 1.0 / (distance**2) 
                edge_attr_list.append([distance])
                edge_attr_list.append([distance])

    if not edge_index_list:
        # Nếu không có kết nối nào, trả về đồ thị rỗng
        return Data(x=node_features, edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 1), dtype=torch.float))

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    
    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return graph

def generate_simulation_data(duration_hours=1, time_step_minutes=1):
    """Tạo ra một chuỗi các snapshot đồ thị."""
    ts = load.timescale()
    satellites = create_satellite_objects(STARLINK_TLE)
    
    start_time = ts.now()
    num_steps = int(duration_hours * 60 / time_step_minutes)
    
    graph_snapshots = []
    
    print("Generating simulation data...")
    for i in tqdm(range(num_steps)):
        time_offset = i * time_step_minutes / (24.0 * 60.0) # Offset in days
        current_time = ts.tt_jd(start_time.tt + time_offset)
        
        positions = get_positions(satellites, current_time)
        graph = create_graph_snapshot(positions, VISIBILITY_THRESHOLD_KM)
        
        # Thêm thông tin thời gian vào đồ thị để tiện xử lý
        graph.time_step = i
        graph_snapshots.append(graph)
        
    return graph_snapshots

if __name__ == '__main__':
    # Chạy thử và lưu dữ liệu
    dataset = generate_simulation_data(duration_hours=2, time_step_minutes=2)
    
    # Lưu dataset vào file
    # Đây sẽ là input cho quá trình training của chúng ta
    save_path = 'data/processed/sagin_simulation_dataset.pt'
    torch.save(dataset, save_path)
    print(f"\nSuccessfully generated {len(dataset)} graph snapshots.")
    print(f"Dataset saved to {save_path}")
    
    # In thông tin của snapshot đầu tiên để kiểm tra
    first_graph = dataset[0]
    print("\n--- First Graph Snapshot ---")
    print(first_graph)
    print(f"Number of nodes: {first_graph.num_nodes}")
    print(f"Number of edges: {first_graph.num_edges}")
    print(f"Node features shape: {first_graph.x.shape}")
    print(f"Edge features shape: {first_graph.edge_attr.shape}")

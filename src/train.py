import torch
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from .dataset import GraphSequenceDataset, sequence_collate_fn
from .models import GNN_GRU_Model
from torch_geometric.data import Batch

# --- Cấu hình ---
DEVICE = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
DATA_PATH = 'data/processed/sagin_simulation_dataset.pt'
EPOCHS = 300
LEARNING_RATE = 0.001 # Có thể thử tăng nhẹ LR cho model phức tạp hơn
BATCH_SIZE = 8       # Giảm batch size vì mỗi sample giờ lớn hơn (chứa cả chuỗi)
SEQUENCE_LENGTH = 4  # Độ dài của chuỗi đồ thị input

# --- Tham số Model ---
NODE_FEATURES_DIM = 3 # Chỉ có x, y, z. Vận tốc sẽ được tính ngầm trong model RNN
GNN_HIDDEN_DIM = 32
GRU_HIDDEN_DIM = 64
GNN_HEADS = 4

def get_feature_scalers(dataset):
    """Tính toán mean/std cho node features và edge_attr trên toàn bộ dataset."""
    all_node_features = []
    all_edge_attrs = []

    # Lặp qua tất cả các snapshot gốc để tính toán
    for snapshot in dataset.snapshots:
        all_node_features.append(snapshot.x)
        all_edge_attrs.append(snapshot.edge_attr)

    node_features_tensor = torch.cat(all_node_features, dim=0)
    edge_attrs_tensor = torch.cat(all_edge_attrs, dim=0)

    node_mean = node_features_tensor.mean(dim=0)
    node_std = node_features_tensor.std(dim=0)
    edge_mean = edge_attrs_tensor.mean(dim=0)
    edge_std = edge_attrs_tensor.std(dim=0)

    node_std[node_std == 0] = 1
    edge_std[edge_std == 0] = 1

    return (node_mean.to(DEVICE), node_std.to(DEVICE)), (edge_mean.to(DEVICE), edge_std.to(DEVICE))

def process_batch(batch_data, scalers, device):
    """Hàm helper để xử lý một batch từ dataloader."""
    (node_mean, node_std), (edge_mean, edge_std) = scalers
    sequences, edge_indices, targets = batch_data

    # 1. Chuẩn hóa và batch các đồ thị theo từng time step
    batched_sequence = []
    for t in range(SEQUENCE_LENGTH):
        graphs_at_t = [seq[t] for seq in sequences]
        for g in graphs_at_t:
            g.x = (g.x.to(device) - node_mean) / node_std

        batched_graph_t = Batch.from_data_list(graphs_at_t).to(device)
        batched_sequence.append(batched_graph_t)

    # 2. Xử lý edge_index và target
    # Vì cấu trúc cạnh trong 1 sample là nhất quán, ta chỉ cần lấy của sample đầu tiên trong batch
    # và điều chỉnh nó cho phù hợp với số node đã được batch
    num_nodes_per_graph = [seq[0].num_nodes for seq in sequences]
    node_offsets = torch.tensor([0] + list(torch.cumsum(torch.tensor(num_nodes_per_graph), dim=0)[:-1]), device=device)

    batched_edge_index = []
    for i, edge_index in enumerate(edge_indices):
        batched_edge_index.append(edge_index.to(device) + node_offsets[i])
    batched_edge_index = torch.cat(batched_edge_index, dim=1)

    # Gộp target và chuẩn hóa
    batched_targets = torch.cat(targets, dim=0).to(device)
    batched_targets_scaled = (batched_targets - edge_mean) / edge_std

    return batched_sequence, batched_edge_index, batched_targets_scaled, batched_targets

def train_model():
    print(f"Using device: {DEVICE}")
    snapshots = torch.load(DATA_PATH)

    full_dataset = GraphSequenceDataset(snapshots, sequence_length=SEQUENCE_LENGTH)

    print("Calculating feature scalers...")
    scalers = get_feature_scalers(full_dataset)
    print("Scalers calculated.")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=sequence_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=sequence_collate_fn)

    model = GNN_GRU_Model(
        node_features_dim=NODE_FEATURES_DIM,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        gru_hidden_dim=GRU_HIDDEN_DIM,
        heads=GNN_HEADS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5) # Giảm LR một nửa sau mỗi 50 epoch

    criterion = torch.nn.MSELoss()

    print("--- Starting Training ---")
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        num_batches = 0
        for batch_data in train_loader:
            if batch_data[0] is None: continue
            num_batches += 1

            sequence, edge_index, targets_scaled, _ = process_batch(batch_data, scalers, DEVICE)

            optimizer.zero_grad()
            out_scaled = model(sequence, edge_index)
            loss = criterion(out_scaled, targets_scaled)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0
        history['train_loss'].append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        num_batches = 0
        with torch.no_grad():
            for batch_data in val_loader:
                if batch_data[0] is None: continue
                num_batches += 1
                sequence, edge_index, targets_scaled, _ = process_batch(batch_data, scalers, DEVICE)
                out_scaled = model(sequence, edge_index)
                loss = criterion(out_scaled, targets_scaled)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
        history['val_loss'].append(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}/{EPOCHS} | Train Loss (scaled): {avg_train_loss:.6f} | Val Loss (scaled): {avg_val_loss:.6f}')
        
        scheduler.step()


    print("--- Training Finished ---")
    torch.save(model.state_dict(), 'gnn_gru_predictor.pth')
    print("Model saved to gnn_gru_predictor.pth")
    return history, val_loader, model, scalers

def plot_results(history, val_loader, model, scalers):
    print("--- Plotting Results ---")
    (_, _), (edge_mean, edge_std) = scalers

    plt.figure(figsize=(12, 5))
    plt.plot(history['train_loss'], label='Training Loss (scaled)')
    plt.plot(history['val_loss'], label='Validation Loss (scaled)')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs'); plt.ylabel('Scaled MSE Loss'); plt.legend(); plt.grid(True)
    plt.savefig('training_loss_curve.png', dpi=600)
    print("Loss curve saved to training_loss_curve.png")
    plt.close()

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_data in val_loader:
            if batch_data[0] is None: continue
            sequence, edge_index, _, targets_original = process_batch(batch_data, scalers, DEVICE)

            preds_scaled = model(sequence, edge_index)
            preds = (preds_scaled * edge_std) + edge_mean

            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets_original.cpu().numpy().flatten())

    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    min_val = min(min(all_targets), min(all_preds)); max_val = max(max(all_targets), max(all_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction (y=x)')
    plt.title('Predicted vs. Actual Distances on Validation Set')
    plt.xlabel('Actual Distance (km)'); plt.ylabel('Predicted Distance (km)'); plt.legend(); plt.grid(True); plt.axis('equal')
    plt.savefig('predictions_vs_actuals.png', dpi=600)
    print("Predictions plot saved to predictions_vs_actuals.png")
    plt.close()

if __name__ == '__main__':
    training_history, validation_loader, trained_model, scalers_tuple = train_model()
    plot_results(training_history, validation_loader, trained_model, scalers_tuple)


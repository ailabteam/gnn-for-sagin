import torch
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .dataset import LinkPredictionDataset, collate_fn
from .models import GNNLinkPredictor
from torch_geometric.data import Batch

# --- Cấu hình ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = 'data/processed/sagin_simulation_dataset.pt'
EPOCHS = 200
# GIẢM LEARNING RATE
LEARNING_RATE = 0.0005 
BATCH_SIZE = 16

def get_feature_scalers(dataset):
    """Tính toán mean và std cho node và edge features trên tập dữ liệu."""
    all_node_features = []
    all_edge_attrs = []
    
    # Chỉ tính trên tập train để tránh data leakage
    for i in range(len(dataset)):
        graph, edge_attr = dataset[i]
        if graph is not None:
            all_node_features.append(graph.x)
            all_edge_attrs.append(edge_attr)
    
    # Nối tất cả tensor lại
    node_features_tensor = torch.cat(all_node_features, dim=0)
    edge_attrs_tensor = torch.cat(all_edge_attrs, dim=0)

    # Tính mean và std
    node_mean = node_features_tensor.mean(dim=0)
    node_std = node_features_tensor.std(dim=0)
    edge_mean = edge_attrs_tensor.mean(dim=0)
    edge_std = edge_attrs_tensor.std(dim=0)

    # Xử lý trường hợp std=0
    node_std[node_std == 0] = 1
    edge_std[edge_std == 0] = 1

    return (node_mean.to(DEVICE), node_std.to(DEVICE)), (edge_mean.to(DEVICE), edge_std.to(DEVICE))

def train_model():
    print(f"Using device: {DEVICE}")
    snapshots = torch.load(DATA_PATH)
    full_dataset = LinkPredictionDataset(snapshots)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # TÍNH TOÁN SCALERS DỰA TRÊN TẬP TRAIN
    print("Calculating feature scalers on the training set...")
    (node_mean, node_std), (edge_mean, edge_std) = get_feature_scalers(train_dataset)
    print("Scalers calculated.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = GNNLinkPredictor(in_channels=3, hidden_channels=64, out_channels=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # THAY ĐỔI LOSS FUNCTION (hoặc giữ nguyên MSE để so sánh)
    criterion = torch.nn.MSELoss() 

    print("--- Starting Training ---")
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        train_batches = 0
        for data_batch, target_batch in train_loader:
            if data_batch is None: continue
            train_batches += 1
            
            # CHUẨN HÓA DỮ LIỆU
            for graph in data_batch:
                graph.x = (graph.x.to(DEVICE) - node_mean) / node_std
            
            data_batch = Batch.from_data_list(data_batch).to(DEVICE)
            target_batch_scaled = (target_batch.to(DEVICE) - edge_mean) / edge_std

            optimizer.zero_grad()
            out_scaled = model(data_batch)
            loss = criterion(out_scaled, target_batch_scaled)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0
        history['train_loss'].append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for data_batch, target_batch in val_loader:
                if data_batch is None: continue
                val_batches += 1

                for graph in data_batch:
                    graph.x = (graph.x.to(DEVICE) - node_mean) / node_std
                
                data_batch = Batch.from_data_list(data_batch).to(DEVICE)
                target_batch_scaled = (target_batch.to(DEVICE) - edge_mean) / edge_std
                
                out_scaled = model(data_batch)
                loss = criterion(out_scaled, target_batch_scaled)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
        history['val_loss'].append(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}/{EPOCHS} | Train Loss (scaled): {avg_train_loss:.6f} | Val Loss (scaled): {avg_val_loss:.6f}')

    print("--- Training Finished ---")
    torch.save(model.state_dict(), 'gnn_link_predictor.pth')
    print("Model saved to gnn_link_predictor.pth")
    return history, val_loader, model, (node_mean, node_std), (edge_mean, edge_std)

def plot_results(history, val_loader, model, scalers):
    print("--- Plotting Results ---")
    (node_mean, node_std), (edge_mean, edge_std) = scalers

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
        for data_batch, target_batch in val_loader:
            if data_batch is None: continue
            
            for graph in data_batch:
                graph.x = (graph.x.to(DEVICE) - node_mean) / node_std
            
            data_batch = Batch.from_data_list(data_batch).to(DEVICE)
            
            # DỰ ĐOÁN TRÊN DỮ LIỆU ĐÃ CHUẨN HÓA
            preds_scaled = model(data_batch)

            # "GIẢI CHUẨN HÓA" KẾT QUẢ
            preds = (preds_scaled * edge_std) + edge_mean
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(target_batch.numpy().flatten())
    
    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    min_val = min(min(all_targets), min(all_preds))
    max_val = max(max(all_targets), max(all_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction (y=x)')
    plt.title('Predicted vs. Actual Distances on Validation Set')
    plt.xlabel('Actual Distance (km)'); plt.ylabel('Predicted Distance (km)'); plt.legend(); plt.grid(True); plt.axis('equal')
    plt.savefig('predictions_vs_actuals.png', dpi=600)
    print("Predictions plot saved to predictions_vs_actuals.png")
    plt.close()

if __name__ == '__main__':
    # Cập nhật để nhận lại scalers
    training_history, validation_loader, trained_model, node_scalers, edge_scalers = train_model()
    plot_results(training_history, validation_loader, trained_model, (node_scalers, edge_scalers))

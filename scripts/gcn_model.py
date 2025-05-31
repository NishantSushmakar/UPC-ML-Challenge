import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import networkx as nx
from torch_geometric.data import Data, Batch
import joblib
import os
import config
from sklearn.metrics import f1_score

class GCNModel(nn.Module):
    def __init__(self, num_node_features: int, hidden_channels: int = 16, num_layers: int = 2):
        """
        Ultra-lightweight Graph Convolutional Network model for root node prediction.
        
        Args:
            num_node_features: Number of input node features
            hidden_channels: Number of hidden channels in GCN layers
            num_layers: Number of GCN layers
        """
        super(GCNModel, self).__init__()
        
        # Minimal GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Simple classifier
        self.classifier = nn.Linear(hidden_channels, 1)
        
        # Light dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Minimal forward pass.
        """
        try:
            # Input validation
            if x.dim() != 2:
                raise ValueError(f"Expected 2D node features tensor, got shape {x.shape}")
            if edge_index.dim() != 2:
                raise ValueError(f"Expected 2D edge index tensor, got shape {edge_index.shape}")
            if edge_index.size(0) != 2:
                raise ValueError(f"Expected edge index with 2 rows, got shape {edge_index.shape}")
            
            # Ensure edge_index is contiguous and on the same device as x
            edge_index = edge_index.contiguous()
            if edge_index.device != x.device:
                edge_index = edge_index.to(x.device)
            
            # Apply GCN layers
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = self.dropout(x)
            
            # Final classification
            x = self.classifier(x)
            return torch.sigmoid(x)
                
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shapes - x: {x.shape}, edge_index: {edge_index.shape}")
            raise

def prepare_graph_data(df: pd.DataFrame) -> List[Data]:
    """
    Prepare graph data from DataFrame for GCN training.
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    graph_data_list = []
    print(f"Preparing graph data from {len(df)} rows...")
    
    for idx, row in df.iterrows():
        try:
            # Create graph from edge list
            edge_list = eval(row['edgelist'])
            G = nx.from_edgelist(edge_list)
            
            # Create node features
            num_nodes = len(G.nodes())
            if num_nodes == 0:
                print(f"Warning: Empty graph found for row {idx}")
                continue
                
            # Initialize with 2 features per node
            node_features = np.zeros((num_nodes, 2))
            
            # Create a mapping from original node indices to consecutive indices
            node_mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
            
            # Add node features
            for node in G.nodes():
                idx = node_mapping[node]
                node_features[idx] = [
                    G.degree(node),
                    nx.clustering(G, node) if len(G[node]) > 1 else 0
                ]
            
            # Convert to PyTorch tensors
            x = torch.FloatTensor(node_features)
            
            # Update edge indices using the new mapping
            edge_index = torch.LongTensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()]).t().contiguous()
            
            # Create target (1 for root node, 0 for others)
            y = torch.zeros(num_nodes)
            root_idx = node_mapping[row['root']]  # Use the mapped index for root
            y[root_idx] = 1
            
            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            
            # Verify data integrity
            if data.x.size(0) != num_nodes:
                print(f"Warning: Node feature size mismatch for row {idx}")
                continue
            if data.edge_index.size(1) != len(G.edges()):
                print(f"Warning: Edge index size mismatch for row {idx}")
                continue
                
            graph_data_list.append(data)
            
            # if idx % 1000 == 0:
            #     print(f"Processed {idx} rows...")
            
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue
    
    print(f"Successfully prepared {len(graph_data_list)} graphs")
    return graph_data_list

def train_gcn_model(train_data: List[Data], valid_data: List[Data], 
                   model_config: Dict, fold: int) -> Tuple[GCNModel, Dict]:
    """
    Train GCN model with minimal memory usage.
    """
    if not train_data or not valid_data:
        raise ValueError("Empty training or validation data")
    
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(valid_data)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        print("Initializing model...")
        num_node_features = train_data[0].x.size(1)
        print(f"Number of node features: {num_node_features}")
        
        # Initialize minimal model
        model = GCNModel(
            num_node_features=num_node_features,
            hidden_channels=16,
            num_layers=2
        ).to(device)
        
        print("Model initialized successfully")
        
        # Use basic Adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001
        )
        
        print("Optimizer initialized successfully")
        
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': []
        }
        
        num_epochs = model_config.get('epochs', 100)
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            model.train()
            total_loss = 0
            train_preds = []
            train_targets = []
            
            # Process training data one graph at a time
            for i, data in enumerate(train_data):
                try:
                    if data.x is None or data.edge_index is None or data.y is None:
                        continue
                        
                    if data.x.size(0) == 0 or data.edge_index.size(1) == 0:
                        continue
                    
                    # Move data to device
                    x = data.x.to(device)
                    edge_index = data.edge_index.to(device)
                    y = data.y.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    out = model(x, edge_index)
                    
                    # Calculate loss
                    loss = F.binary_cross_entropy(out, y.unsqueeze(1))
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Store results
                    total_loss += loss.item()
                    train_preds.extend((out > 0.5).cpu().detach().numpy())
                    train_targets.extend(y.cpu().numpy())
                    
                    # Clear memory
                    del x, edge_index, y, out, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing training graph {i}: {str(e)}")
                    continue
                
                if i % 1000 == 0:
                    print(f"Processed {i}/{len(train_data)} training graphs")
                    # Force garbage collection
                    import gc
                    gc.collect()
            
            # Validation
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for i, data in enumerate(valid_data):
                    try:
                        if data.x is None or data.edge_index is None or data.y is None:
                            continue
                            
                        if data.x.size(0) == 0 or data.edge_index.size(1) == 0:
                            continue
                        
                        x = data.x.to(device)
                        edge_index = data.edge_index.to(device)
                        y = data.y.to(device)
                        
                        out = model(x, edge_index)
                        val_loss += F.binary_cross_entropy(out, y.unsqueeze(1)).item()
                        
                        val_preds.extend((out > 0.5).cpu().numpy())
                        val_targets.extend(y.cpu().numpy())
                        
                        # Clear memory
                        del x, edge_index, y, out
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"Error processing validation graph {i}: {str(e)}")
                        continue
                    
                    if i % 1000 == 0:
                        print(f"Processed {i}/{len(valid_data)} validation graphs")
                        # Force garbage collection
                        import gc
                        gc.collect()
            
            # Calculate metrics
            train_loss = total_loss / len(train_data)
            val_loss = val_loss / len(valid_data)
            
            train_f1 = f1_score(train_targets, train_preds, average='macro')
            val_f1 = f1_score(val_targets, val_preds, average='macro')
            
            metrics['train_loss'].append(train_loss)
            metrics['val_loss'].append(val_loss)
            metrics['train_f1'].append(train_f1)
            metrics['val_f1'].append(val_f1)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 
                          os.path.join(config.MODEL_PATH, f'gcn_model_fold_{fold}.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            print(f'Epoch {epoch + 1}:')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')
            
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return model, metrics
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        raise

def predict_gcn(model: GCNModel, test_data: List[Data]) -> np.ndarray:
    """
    Make predictions using trained GCN model.
    
    Args:
        model: Trained GCN model
        test_data: List of test graph data
        
    Returns:
        Array of predicted probabilities
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for batch in test_data:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            predictions.extend(out.cpu().numpy())
    
    return np.array(predictions)

# if __name__ == "__main__":
#     # Example usage
#     model_config = {
#         'hidden_channels': 64,
#         'num_layers': 3,
#         'learning_rate': 0.001,
#         'weight_decay': 5e-4,
#         'epochs': 100
#     }
    
#     # Load and prepare data
#     df = pd.read_csv(config.TRAINING_DATA_PATH)
#     df = create_groupkfolds(df, n_folds=5, group_col='sentence')
    
#     # Train for one fold
#     fold = 0
#     df_train = df[df.kfold != fold].reset_index(drop=True)
#     df_valid = df[df.kfold == fold].reset_index(drop=True)
    
#     train_data = prepare_graph_data(df_train)
#     valid_data = prepare_graph_data(df_valid)
    
#     model, metrics = train_gcn_model(train_data, valid_data, model_config, fold) 
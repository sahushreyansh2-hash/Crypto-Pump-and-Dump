import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

try:
    from src.features import FEATURE_REGISTRY
except ImportError:
    FEATURE_REGISTRY = {}
    
logger = logging.getLogger(__name__)

class MarketSignalEncoder(nn.Module):
    """
    Encodes market features (price, volume, volatility, price-volume)
    using an LSTM followed by an attention mechanism.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden_dim)
        
        # Self-attention mechanism
        attn_weights = torch.softmax(self.attention(rnn_out), dim=1)  # (batch, seq_len, 1)
        attended = torch.sum(attn_weights * rnn_out, dim=1)  # (batch, hidden_dim)
        return attended

class TemporalSignalEncoder(nn.Module):
    """
    Encodes temporal features (time, momentum) using an LSTM.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        _, (hn, _) = self.rnn(x)
        # hn: (num_layers, batch, hidden_dim)
        return hn[-1]  # (batch, hidden_dim)

class ContextEncoder(nn.Module):
    """
    Placeholder for future channel/history data (social media, announcements, etc.).
    Currently a simple linear layer if context features are available.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        if input_dim > 0:
            self.linear = nn.Linear(input_dim, hidden_dim)
        else:
            self.linear = None

    def forward(self, x: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if self.linear is not None and x is not None and self.input_dim > 0:
            return self.linear(x)
        else:
            # Return zeros if no context features
            return torch.zeros(batch_size, self.hidden_dim, device=device)

class FusionModule(nn.Module):
    """
    Fuses embeddings from encoders to produce a single logit.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class ETASEModel(nn.Module):
    """
    ETASE Framework Model combining Market, Temporal, and Context Encoders.
    """
    def __init__(self, market_dim: int, temporal_dim: int, context_dim: int, cfg: Dict[str, Any]):
        super().__init__()
        self.market_hidden = cfg.get('etase', {}).get('market_hidden', 64)
        self.temporal_hidden = cfg.get('etase', {}).get('temporal_hidden', 32)
        self.context_hidden = cfg.get('etase', {}).get('context_hidden', 16)
        self.fusion_hidden = cfg.get('etase', {}).get('fusion_hidden', 64)
        dropout = cfg.get('etase', {}).get('dropout', 0.2)
        num_layers = cfg.get('etase', {}).get('num_layers', 1)
        
        self.market_encoder = MarketSignalEncoder(market_dim, self.market_hidden, num_layers, dropout)
        self.temporal_encoder = TemporalSignalEncoder(temporal_dim, self.temporal_hidden, num_layers, dropout)
        self.context_encoder = ContextEncoder(context_dim, self.context_hidden)
        
        fusion_input_dim = self.market_hidden + self.temporal_hidden + self.context_hidden
        self.fusion_module = FusionModule(fusion_input_dim, self.fusion_hidden, dropout)

    def forward(self, market_seq: torch.Tensor, temporal_seq: torch.Tensor, context_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = market_seq.size(0)
        device = market_seq.device
        
        market_emb = self.market_encoder(market_seq)
        temporal_emb = self.temporal_encoder(temporal_seq)
        context_emb = self.context_encoder(context_features, batch_size, device)
        
        fused = torch.cat([market_emb, temporal_emb, context_emb], dim=-1)
        logits = self.fusion_module(fused)
        return logits

class ETASEDataset(Dataset):
    """
    Dataset for ETASE model.
    Separates features into market_features and temporal_features based on groups.
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series, market_cols: List[str], temporal_cols: List[str], seq_len: int):
        self.seq_len = seq_len
        self.market_data = torch.tensor(X[market_cols].fillna(0).values, dtype=torch.float32)
        self.temporal_data = torch.tensor(X[temporal_cols].fillna(0).values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        
        self.indices = []
        if 'symbol' in X.columns:
            symbols = X['symbol'].values
            for i in range(len(X) - seq_len + 1):
                if symbols[i] == symbols[i + seq_len - 1]:
                    self.indices.append(i)
        else:
            self.indices = list(range(len(X) - seq_len + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len
        
        market_seq = self.market_data[start_idx:end_idx]
        temporal_seq = self.temporal_data[start_idx:end_idx]
        target = self.y[end_idx - 1]
        
        return market_seq, temporal_seq, target

def get_feature_groups(feature_columns: List[str]) -> Tuple[List[str], List[str]]:
    """
    Categorizes features into market and temporal groups based on FEATURE_REGISTRY.
    """
    market_groups = {'price', 'volume', 'volatility', 'price_volume'}
    temporal_groups = {'time', 'momentum'}
    
    market_cols = []
    temporal_cols = []
    
    feature_to_group = {}
    for group, features in FEATURE_REGISTRY.items():
        if isinstance(features, list):
            for f in features:
                feature_to_group[f] = group
        elif isinstance(features, dict):
            for f in features.keys():
                feature_to_group[f] = group
                
    for col in feature_columns:
        group_found = None
        if col in feature_to_group:
            group_found = feature_to_group[col]
        else:
            # Prefix match for generated features
            for f, grp in feature_to_group.items():
                if col.startswith(f):
                    group_found = grp
                    break
                    
        if group_found in market_groups:
            market_cols.append(col)
        elif group_found in temporal_groups:
            temporal_cols.append(col)
        else:
            # Default to market if unknown
            market_cols.append(col)
            
    return market_cols, temporal_cols

def train_etase(X_train: pd.DataFrame, y_train: pd.Series, 
                X_val: pd.DataFrame, y_val: pd.Series, 
                feature_columns: List[str], cfg: Dict[str, Any]) -> nn.Module:
    """
    Train the ETASE model with early stopping.
    """
    seed = cfg.get('random_seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    market_cols, temporal_cols = get_feature_groups(feature_columns)
    logger.info(f"Market columns: {len(market_cols)}, Temporal columns: {len(temporal_cols)}")
    
    context_dim = 0
    if context_dim == 0:
        logger.info("Context/channel features not available in current dataset.")
        
    seq_len = cfg.get('sequence_length', 10)
    batch_size = cfg.get('etase', {}).get('batch_size', 64)
    epochs = cfg.get('etase', {}).get('epochs', 50)
    lr = cfg.get('etase', {}).get('lr', 1e-3)
    patience = cfg.get('etase', {}).get('patience', 10)
    
    train_dataset = ETASEDataset(X_train, y_train, market_cols, temporal_cols, seq_len)
    val_dataset = ETASEDataset(X_val, y_val, market_cols, temporal_cols, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = ETASEModel(market_dim=len(market_cols), 
                       temporal_dim=len(temporal_cols), 
                       context_dim=context_dim, 
                       cfg=cfg).to(device)
    
    # Positive weight for imbalanced classes
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    os.makedirs('models', exist_ok=True)
    save_path = 'models/etase_model.pt'
    
    logger.info("Starting ETASE training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for market_seq, temporal_seq, target in train_loader:
            market_seq = market_seq.to(device)
            temporal_seq = temporal_seq.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            logits = model(market_seq, temporal_seq, None)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * market_seq.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for market_seq, temporal_seq, target in val_loader:
                market_seq = market_seq.to(device)
                temporal_seq = temporal_seq.to(device)
                target = target.to(device)
                
                logits = model(market_seq, temporal_seq, None)
                loss = criterion(logits, target)
                val_loss += loss.item() * market_seq.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model

def predict_etase(model: nn.Module, X: pd.DataFrame, feature_columns: List[str], 
                  seq_len: int, device: torch.device) -> np.ndarray:
    """
    Predict probabilities using the ETASE model.
    """
    model.eval()
    market_cols, temporal_cols = get_feature_groups(feature_columns)
    
    # Dummy y for dataset creation
    y = pd.Series(np.zeros(len(X)))
    dataset = ETASEDataset(X, y, market_cols, temporal_cols, seq_len)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    probs = []
    with torch.no_grad():
        for market_seq, temporal_seq, _ in loader:
            market_seq = market_seq.to(device)
            temporal_seq = temporal_seq.to(device)
            
            logits = model(market_seq, temporal_seq, None)
            prob = torch.sigmoid(logits)
            
            probs.extend(prob.cpu().numpy())
            
    return np.array(probs)

def run_etase(X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame, y_val: pd.Series, 
              X_test: pd.DataFrame, y_test: pd.Series, 
              feature_columns: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper function to train, evaluate, and return predictions for the ETASE model.
    """
    logger.info("Initializing ETASE model training workflow...")
    
    model = train_etase(X_train, y_train, X_val, y_val, feature_columns, cfg)
    
    device = next(model.parameters()).device
    seq_len = cfg.get('sequence_length', 10)
    
    val_probs = predict_etase(model, X_val, feature_columns, seq_len, device)
    test_probs = predict_etase(model, X_test, feature_columns, seq_len, device)
    
    # Align targets with predictions due to sequence length reduction
    val_targets = y_val.values[seq_len - 1:]
    test_targets = y_test.values[seq_len - 1:]
    
    return {
        'model': model,
        'val_probs': val_probs,
        'test_probs': test_probs,
        'val_targets': val_targets,
        'test_targets': test_targets
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Dummy test block
    np.random.seed(42)
    cfg = {
        'random_seed': 42,
        'sequence_length': 5,
        'etase': {
            'market_hidden': 16,
            'temporal_hidden': 8,
            'context_hidden': 4,
            'fusion_hidden': 16,
            'epochs': 2,
            'batch_size': 32,
            'lr': 1e-3,
            'patience': 2,
            'dropout': 0.1
        }
    }
    
    global FEATURE_REGISTRY
    FEATURE_REGISTRY = {
        'price': ['close', 'high', 'low'],
        'time': ['hour', 'day'],
        'momentum': ['rsi', 'macd']
    }
    
    dates = pd.date_range('2023-01-01', periods=100, freq='h')
    df = pd.DataFrame(np.random.randn(100, 7), columns=['close', 'high', 'low', 'hour', 'day', 'rsi', 'macd'], index=dates)
    y = pd.Series(np.random.randint(0, 2, 100), index=dates)
    
    X_train, y_train = df.iloc[:60], y.iloc[:60]
    X_val, y_val = df.iloc[60:80], y.iloc[60:80]
    X_test, y_test = df.iloc[80:], y.iloc[80:]
    
    feature_columns = df.columns.tolist()
    
    res = run_etase(X_train, y_train, X_val, y_val, X_test, y_test, feature_columns, cfg)
    print("Test probs shape:", res['test_probs'].shape)
    print("Test targets shape:", res['test_targets'].shape)

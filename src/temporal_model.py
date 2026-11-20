"""
Temporal sequence models for cryptocurrency pump-and-dump detection.
Implements PyTorch-based LSTM and GRU models for sequence classification.
"""
import logging
import os
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """
    Dataset for creating sequences from time-series data.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 30):
        """
        Initialize the dataset.

        Args:
            X (np.ndarray): Feature matrix of shape (num_samples, num_features).
            y (np.ndarray): Target array of shape (num_samples,).
            sequence_length (int): Number of timesteps in each sequence.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sequence_length = sequence_length
        self.num_samples = len(X)
        
        if self.sequence_length > self.num_samples:
            logger.warning(f"Sequence length ({self.sequence_length}) is greater than number of samples ({self.num_samples}).")

    def __len__(self) -> int:
        """Returns the number of valid sequences in the dataset."""
        if self.sequence_length > self.num_samples:
            return 0
        return self.num_samples - self.sequence_length + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its corresponding target.

        Args:
            idx (int): Index of the sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Sequence tensor of shape (sequence_length, num_features)
                - Target tensor for the last timestep of the sequence
        """
        sequence = self.X[idx:idx + self.sequence_length]
        target = self.y[idx + self.sequence_length - 1]
        return sequence, target


class TemporalModel(nn.Module):
    """
    Recurrent Neural Network model (LSTM or GRU) for sequence classification.
    """
    def __init__(self, input_size: int, config: Dict[str, Any]):
        """
        Initialize the temporal model.

        Args:
            input_size (int): Number of features in the input sequence.
            config (Dict[str, Any]): Configuration dictionary containing model hyperparameters.
        """
        super(TemporalModel, self).__init__()
        
        # Get hyperparameters from config with defaults
        model_type = config.get('temporal_model', {}).get('model_type', 'LSTM').upper()
        hidden_size = config.get('temporal_model', {}).get('hidden_size', 64)
        num_layers = config.get('temporal_model', {}).get('num_layers', 2)
        dropout = config.get('temporal_model', {}).get('dropout', 0.2)
        
        self.model_type = model_type
        
        # RNN Layer
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif model_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Must be 'LSTM' or 'GRU'.")
            
        # Optional dropout on final hidden state
        self.dropout_layer = nn.Dropout(dropout)
            
        # Final classification layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, 1).
        """
        # Pass through RNN layer
        if self.model_type == 'LSTM':
            # LSTM returns (output, (h_n, c_n))
            rnn_out, (h_n, _) = self.rnn(x)
        else:
            # GRU returns (output, h_n)
            rnn_out, h_n = self.rnn(x)
            
        # Use the last hidden state for classification
        # h_n shape: (num_layers, batch, hidden_size) -> take the last layer
        last_hidden = h_n[-1]
        
        out = self.dropout_layer(last_hidden)
        logits = self.fc(out)
        return logits


def train_temporal_model(
    X_train: np.ndarray, y_train: np.ndarray, 
    X_val: np.ndarray, y_val: np.ndarray, 
    cfg: Dict[str, Any]
) -> nn.Module:
    """
    Train the temporal model with early stopping.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation targets.
        cfg (Dict[str, Any]): Configuration dictionary.

    Returns:
        nn.Module: Trained model.
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set seed
    seed = cfg.get('random_seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # Hyperparameters
    sequence_length = cfg.get('sequence_length', 30)
    batch_size = cfg.get('temporal_model', {}).get('batch_size', 64)
    learning_rate = cfg.get('temporal_model', {}).get('learning_rate', 1e-3)
    epochs = cfg.get('temporal_model', {}).get('epochs', 100)
    patience = cfg.get('temporal_model', {}).get('patience', 10)
    
    input_size = X_train.shape[1]
    
    # Create datasets and dataloaders
    train_dataset = SequenceDataset(X_train, y_train, sequence_length=sequence_length)
    val_dataset = SequenceDataset(X_val, y_val, sequence_length=sequence_length)
    
    # Check if there are valid sequences
    if len(train_dataset) == 0:
        raise ValueError("Training dataset sequence length is longer than the dataset itself.")
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset sequence length is longer than the dataset itself.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TemporalModel(input_size=input_size, config=cfg).to(device)
    
    # Handle class imbalance
    num_pos = np.sum(y_train == 1)
    num_neg = len(y_train) - num_pos
    pos_weight_val = num_neg / max(num_pos, 1)  # Avoid division by zero
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    logger.info(f"Using pos_weight for BCEWithLogitsLoss: {pos_weight_val:.4f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join('models', 'temporal_model.pt')
    
    os.makedirs('models', exist_ok=True)
    
    logger.info(f"Starting training for up to {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(-1)  # shape (batch_size,)
            
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x).squeeze(-1)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            logger.debug(f"Validation loss improved. Saved model to {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    # Load best model for returning
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    
    return model


def predict_temporal(
    model: nn.Module, X: np.ndarray, sequence_length: int, device: str = 'cpu'
) -> np.ndarray:
    """
    Generate probability predictions for time-series data.

    Args:
        model (nn.Module): Trained temporal model.
        X (np.ndarray): Feature matrix.
        sequence_length (int): Sequence length used for windowing.
        device (str, optional): Device to run predictions on. Defaults to 'cpu'.

    Returns:
        np.ndarray: Array of predicted probabilities of shape (num_valid_sequences,).
    """
    model.eval()
    dataset = SequenceDataset(X, np.zeros(len(X)), sequence_length=sequence_length)
    
    if len(dataset) == 0:
        logger.warning("Dataset too small for sequence prediction. Returning empty array.")
        return np.array([])
        
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    probs = []
    
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x).squeeze(-1)
            # Apply sigmoid to get probabilities
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            probs.extend(batch_probs)
            
    return np.array(probs)


def run_temporal_model(
    X_train: np.ndarray, y_train: np.ndarray, 
    X_val: np.ndarray, y_val: np.ndarray, 
    X_test: np.ndarray, y_test: np.ndarray, 
    cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    End-to-end wrapper for training and predicting with the temporal model.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        cfg: Configuration dict

    Returns:
        Dict[str, Any]: Dictionary containing model, predictions and aligned targets.
    """
    sequence_length = cfg.get('sequence_length', 30)
    
    logger.info("Starting temporal model training pipeline...")
    model = train_temporal_model(X_train, y_train, X_val, y_val, cfg)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info("Generating predictions for validation and test sets...")
    val_probs = predict_temporal(model, X_val, sequence_length, device)
    test_probs = predict_temporal(model, X_test, sequence_length, device)
    
    # Align targets with predictions by skipping the first (sequence_length - 1) elements
    if len(y_val) >= sequence_length:
        val_targets = y_val[sequence_length - 1:]
    else:
        val_targets = np.array([])
        
    if len(y_test) >= sequence_length:
        test_targets = y_test[sequence_length - 1:]
    else:
        test_targets = np.array([])
    
    logger.info(f"Pipeline completed. Val predictions shape: {val_probs.shape}, Test predictions shape: {test_probs.shape}")
    
    return {
        'model': model,
        'val_probs': val_probs,
        'test_probs': test_probs,
        'val_targets': val_targets,
        'test_targets': test_targets
    }


if __name__ == '__main__':
    # Simple test block
    logger.setLevel(logging.DEBUG)
    
    # Mock data
    np.random.seed(42)
    seq_len = 10
    features = 5
    
    X_tr = np.random.randn(100, features)
    y_tr = np.random.randint(0, 2, 100)
    
    X_v = np.random.randn(30, features)
    y_v = np.random.randint(0, 2, 30)
    
    X_te = np.random.randn(20, features)
    y_te = np.random.randint(0, 2, 20)
    
    config = {
        'random_seed': 42,
        'sequence_length': seq_len,
        'temporal_model': {
            'model_type': 'GRU',
            'hidden_size': 16,
            'num_layers': 1,
            'dropout': 0.1,
            'batch_size': 16,
            'learning_rate': 0.01,
            'epochs': 2,
            'patience': 2
        }
    }
    
    results = run_temporal_model(X_tr, y_tr, X_v, y_v, X_te, y_te, config)
    
    print(f"Validation targets shape: {results['val_targets'].shape}")
    print(f"Validation probs shape: {results['val_probs'].shape}")
    
    # Edge case: sequence_length > len(data)
    X_short = np.random.randn(5, features)
    y_short = np.random.randint(0, 2, 5)
    short_results = run_temporal_model(X_tr, y_tr, X_v, y_v, X_short, y_short, config)
    print(f"Short Test targets shape: {short_results['test_targets'].shape}")
    print(f"Short Test probs shape: {short_results['test_probs'].shape}")

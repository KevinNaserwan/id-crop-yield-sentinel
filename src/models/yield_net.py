import torch
import torch.nn as nn

class SpatioTemporalYieldNet(nn.Module):
    """
    Bi-directional LSTM with self-attention mechanism to ingest time-series
    vegetation index profiles (NDVI, NDRE, NDWI, SAR VV/VH) across the 120-day
    rice growing season to predict crop yield (Tons/Ha) and harvest failure risk.
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Yield output in Tons/Hectare
        )
        self.failure_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1), # Probability of crop failure / puso
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        # x shape: [batch_size, sequence_length (timesteps), input_features]
        lstm_out, _ = self.lstm(x) # [batch_size, seq_len, hidden_dim * 2]
        
        # Self-attention pooling
        weights = self.attention(lstm_out) # [batch_size, seq_len, 1]
        context = torch.sum(weights * lstm_out, dim=1) # [batch_size, hidden_dim * 2]

        predicted_yield = self.regressor(context)
        failure_prob = self.failure_classifier(context)

        return predicted_yield, failure_prob

import torch
import torch.nn as nn


class MusicRewardModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, token_ids):
        """
        token_ids: [B, T]
        returns normalized reward in [0, 1]
        """
        embedded = self.embedding(token_ids)
        _, (hidden, _) = self.encoder(embedded)

        h = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        reward = self.regressor(h).squeeze(-1)
        return reward
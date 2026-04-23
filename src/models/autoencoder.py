import torch
import torch.nn as nn


class MusicAutoencoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Map encoder hidden state to latent vector z
        self.hidden_to_z = nn.Linear(hidden_dim, latent_dim)

        # Map latent vector z into decoder initial hidden/cell states
        self.z_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.z_to_cell = nn.Linear(latent_dim, hidden_dim * num_layers)

        # Decoder gets token embedding + latent vector at every step
        self.decoder = nn.LSTM(
            input_size=embed_dim + latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def encode(self, encoder_input: torch.Tensor) -> torch.Tensor:
        """
        encoder_input: [B, T]
        returns z: [B, latent_dim]
        """
        embedded = self.embedding(encoder_input)               # [B, T, E]
        _, (hidden, _) = self.encoder(embedded)               # hidden: [L, B, H]

        h_last = hidden[-1]                                   # [B, H]
        z = self.hidden_to_z(h_last)                          # [B, Z]
        return z

    def init_decoder_state(self, z: torch.Tensor):
        """
        z: [B, latent_dim]
        returns hidden, cell each of shape [L, B, H]
        """
        batch_size = z.size(0)

        hidden = self.z_to_hidden(z).view(batch_size, self.num_layers, self.hidden_dim)
        cell = self.z_to_cell(z).view(batch_size, self.num_layers, self.hidden_dim)

        hidden = hidden.transpose(0, 1).contiguous()          # [L, B, H]
        cell = cell.transpose(0, 1).contiguous()              # [L, B, H]
        return hidden, cell

    def decode(self, decoder_input: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        decoder_input: [B, T]
        z: [B, latent_dim]
        returns logits: [B, T, vocab_size]
        """
        embedded = self.embedding(decoder_input)              # [B, T, E]
        z_seq = z.unsqueeze(1).expand(-1, embedded.size(1), -1)  # [B, T, Z]
        decoder_in = torch.cat([embedded, z_seq], dim=-1)     # [B, T, E+Z]

        hidden_init, cell_init = self.init_decoder_state(z)
        decoded, _ = self.decoder(decoder_in, (hidden_init, cell_init))
        logits = self.output_layer(decoded)                   # [B, T, V]
        return logits

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor):
        """
        encoder_input: [B, T]
        decoder_input: [B, T]
        returns logits, z
        """
        z = self.encode(encoder_input)
        logits = self.decode(decoder_input, z)
        return logits, z

    def decode_step(self, input_token, hidden, cell, z):
        """
        Single-step decoder for generation.
        input_token: [B]
        hidden/cell: [L, B, H]
        z: [B, Z]
        """
        embedded = self.embedding(input_token)                # [B, E]
        step_in = torch.cat([embedded, z], dim=-1).unsqueeze(1)  # [B, 1, E+Z]

        output, (hidden, cell) = self.decoder(step_in, (hidden, cell))
        logits = self.output_layer(output.squeeze(1))         # [B, V]
        return logits, hidden, cell
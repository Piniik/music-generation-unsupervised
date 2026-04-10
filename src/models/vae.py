import torch
import torch.nn as nn


class MusicVAE(nn.Module):
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
            bidirectional=True,
        )

        encoder_out_dim = hidden_dim * 2
        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_out_dim, latent_dim)

        self.z_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.z_to_cell = nn.Linear(latent_dim, hidden_dim * num_layers)

        self.decoder = nn.LSTM(
            input_size=embed_dim + latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def encode(self, encoder_input):
        embedded = self.embedding(encoder_input)
        _, (hidden, _) = self.encoder(embedded)

        # last forward + last backward
        h_last = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def init_decoder_state(self, z):
        batch_size = z.size(0)

        hidden = self.z_to_hidden(z).view(batch_size, self.num_layers, self.hidden_dim)
        cell = self.z_to_cell(z).view(batch_size, self.num_layers, self.hidden_dim)

        hidden = hidden.transpose(0, 1).contiguous()
        cell = cell.transpose(0, 1).contiguous()
        return hidden, cell

    def decode(self, decoder_input, z):
        embedded = self.embedding(decoder_input)                            # [B, T, E]
        z_seq = z.unsqueeze(1).expand(-1, embedded.size(1), -1)            # [B, T, Z]
        decoder_in = torch.cat([embedded, z_seq], dim=-1)                  # [B, T, E+Z]

        hidden_init, cell_init = self.init_decoder_state(z)
        decoded, _ = self.decoder(decoder_in, (hidden_init, cell_init))
        logits = self.output_layer(decoded)
        return logits

    def forward(self, encoder_input, decoder_input):
        mu, logvar = self.encode(encoder_input)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(decoder_input, z)
        return logits, mu, logvar

    def decode_step(self, input_token, hidden, cell, z):
        embedded = self.embedding(input_token)                              # [B, E]
        step_in = torch.cat([embedded, z], dim=-1).unsqueeze(1)            # [B, 1, E+Z]
        output, (hidden, cell) = self.decoder(step_in, (hidden, cell))
        logits = self.output_layer(output.squeeze(1))
        return logits, hidden, cell
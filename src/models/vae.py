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
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def encode(self, encoder_input):
        embedded = self.embedding(encoder_input)
        _, (hidden, _) = self.encoder(embedded)

        h_last = hidden[-1]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, decoder_input, z):
        embedded = self.embedding(decoder_input)

        hidden_init = self.latent_to_hidden(z)
        hidden_init = hidden_init.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell_init = torch.zeros_like(hidden_init)

        decoded, _ = self.decoder(embedded, (hidden_init, cell_init))
        logits = self.output_layer(decoded)
        return logits

    def forward(self, encoder_input, decoder_input):
        mu, logvar = self.encode(encoder_input)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(decoder_input, z)
        return logits, mu, logvar
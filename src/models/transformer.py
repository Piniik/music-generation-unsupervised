import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        batch_size, seq_len, d_model = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.position_embedding(positions)  # [B, T, D]
        return x + pos_emb


class MusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 512,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(max_seq_len=max_seq_len, d_model=d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns a causal mask of shape [T, T], where True means "do not attend".
        This is the mask format expected by nn.TransformerEncoder when batch_first=True.
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T]
        returns logits: [B, T, vocab_size]
        """
        batch_size, seq_len = input_ids.size()

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model max_seq_len {self.max_seq_len}. "
                "Increase max_seq_len or reduce input length."
            )

        x = self.token_embedding(input_ids) * (self.d_model ** 0.5)   # [B, T, D]
        x = self.positional_encoding(x)                               # [B, T, D]
        x = self.dropout(x)

        causal_mask = self._generate_causal_mask(seq_len, input_ids.device)  # [T, T]

        x = self.transformer(
            src=x,
            mask=causal_mask,
        )                                                             # [B, T, D]

        x = self.final_norm(x)
        logits = self.output_layer(x)                                 # [B, T, V]
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        input_ids: [B, T_start]
        returns: [B, T_start + max_new_tokens]
        """
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            if generated.size(1) > self.max_seq_len:
                context = generated[:, -self.max_seq_len :]
            else:
                context = generated

            logits = self(context)                        # [B, T, V]
            next_token_logits = logits[:, -1, :]         # [B, V]

            if temperature <= 0:
                raise ValueError("temperature must be > 0.")
            next_token_logits = next_token_logits / temperature

            if top_k is not None and top_k > 0:
                top_values, _ = torch.topk(
                    next_token_logits,
                    k=min(top_k, next_token_logits.size(-1)),
                    dim=-1,
                )
                kth = top_values[:, -1].unsqueeze(1)
                next_token_logits = torch.where(
                    next_token_logits < kth,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)      # [B, 1]

            generated = torch.cat([generated, next_token], dim=1)     # [B, T+1]

        return generated
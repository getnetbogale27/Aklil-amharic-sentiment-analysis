"""
Bidirectional LSTM model for Amharic sentiment classification.

Architecture:
  Embedding → Bi-LSTM (stacked) → Attention → Dropout → Linear

The attention layer helps the model focus on sentiment-bearing words in
Amharic, which is a morphologically rich language where a single word can
carry the sentiment load of an entire phrase.

Supports 3-class classification: positive / negative / neutral.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Additive self-attention over Bi-LSTM hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden*2)
        scores = self.attn(lstm_out).squeeze(-1)          # (batch, seq_len)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1) # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)         # (batch, hidden*2)
        return context


class AmharicBiLSTM(nn.Module):
    """
    Stacked Bidirectional LSTM for Amharic sentiment analysis.

    Args:
        vocab_size:    Size of the Amharic vocabulary (from tokenizer).
        embed_dim:     Embedding dimension (default 128).
        hidden_dim:    LSTM hidden state size per direction (default 256).
        num_layers:    Number of stacked LSTM layers (default 2).
        num_classes:   Output classes — 3 for pos/neg/neu (default 3).
        dropout:       Dropout probability (default 0.5).
        pad_idx:       Padding token index for the embedding layer.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.5,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = SelfAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # hidden_dim * 2 because bidirectional
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)
        embedded = self.dropout(self.embedding(input_ids))   # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)                    # (batch, seq_len, hidden*2)
        context = self.attention(lstm_out)                   # (batch, hidden*2)
        context = self.dropout(context)
        logits = self.classifier(context)                    # (batch, num_classes)
        return logits


def build_bilstm(config: dict) -> AmharicBiLSTM:
    """Instantiate AmharicBiLSTM from a config dictionary (loaded from YAML)."""
    return AmharicBiLSTM(
        vocab_size=config["vocab_size"],
        embed_dim=config.get("embed_dim", 128),
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 2),
        num_classes=config.get("num_classes", 3),
        dropout=config.get("dropout", 0.5),
        pad_idx=config.get("pad_idx", 0),
    )


if __name__ == "__main__":
    # Quick smoke-test with random tensors
    model = AmharicBiLSTM(vocab_size=5000)
    dummy = torch.randint(0, 5000, (8, 64))  # batch=8, seq_len=64
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # expected: (8, 3)
    print("AmharicBiLSTM smoke-test passed.")

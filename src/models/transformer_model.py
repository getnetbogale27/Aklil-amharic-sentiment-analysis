"""
XLM-RoBERTa fine-tuning wrapper for Amharic sentiment analysis.

XLM-RoBERTa (xlm-roberta-base / xlm-roberta-large) was pre-trained on 100
languages including Amharic using the CC-100 corpus.  Fine-tuning it on the
AfriSenti Amharic dataset and our collected policy tweets typically yields
state-of-the-art performance for low-resource Amharic NLP.

References:
  Conneau et al. (2020) "Unsupervised Cross-lingual Representation Learning
  at Scale". https://arxiv.org/abs/1911.02116
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional


# Default checkpoint — swap for 'xlm-roberta-large' for best accuracy
DEFAULT_CHECKPOINT = "xlm-roberta-base"

# AfriSenti / our dataset use these three sentiment labels
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class XLMRobertaSentiment(nn.Module):
    """
    Classification head on top of XLM-RoBERTa.

    Uses the [CLS] token representation followed by a linear projection.
    A light dropout is added before the classifier to regularise fine-tuning
    on the small Amharic training set (~10k examples in AfriSenti).
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        num_labels: int = 3,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)
        hidden_size = self.encoder.config.hidden_size  # 768 for base, 1024 for large

        if freeze_base:
            # Freeze all encoder weights — useful for probing experiments
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)             # (batch, num_labels)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


def load_tokenizer(checkpoint: str = DEFAULT_CHECKPOINT) -> AutoTokenizer:
    """Return the matching XLM-R tokenizer (handles Amharic subword units)."""
    return AutoTokenizer.from_pretrained(checkpoint)


def tokenize_batch(
    texts: list[str],
    tokenizer: AutoTokenizer,
    max_length: int = 128,
) -> dict:
    """
    Tokenize a list of Amharic strings for XLM-RoBERTa.

    max_length=128 covers ~99% of AfriSenti Amharic tweets.
    """
    return tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )


def build_transformer(config: dict) -> XLMRobertaSentiment:
    """Instantiate model from a config dict (loaded from configs/model_config.yaml)."""
    return XLMRobertaSentiment(
        checkpoint=config.get("transformer_checkpoint", DEFAULT_CHECKPOINT),
        num_labels=config.get("num_classes", 3),
        dropout=config.get("transformer_dropout", 0.1),
        freeze_base=config.get("freeze_base", False),
    )


if __name__ == "__main__":
    # Smoke-test: tokenise two Amharic sentences and run a forward pass
    sample_texts = [
        "መንግስቱ ለህዝቡ ጥሩ አገልግሎት ሰጥቷል",   # "The government served the people well"
        "ፖሊሲው ለድሃ ህዝብ ጠቃሚ አይደለም",          # "The policy is not beneficial for the poor"
    ]
    tokenizer = load_tokenizer()
    inputs = tokenize_batch(sample_texts, tokenizer)
    model = XLMRobertaSentiment()
    model.eval()
    with torch.no_grad():
        out = model(**inputs)
    print(f"Logits shape: {out['logits'].shape}")  # (2, 3)
    print("XLMRobertaSentiment smoke-test passed.")

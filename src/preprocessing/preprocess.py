"""
Amharic text preprocessing pipeline.

Handles the unique challenges of Amharic (Ethiopic script / Ge'ez):
  - Unicode normalization of visually identical Ethiopic characters
  - Removal of social-media noise (URLs, hashtags, mentions, emoji)
  - Amharic-specific stopword filtering
  - Whitespace-based tokenization (Amharic uses the Ethiopic word separator ፡)
"""

import re
import unicodedata
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Amharic character normalisation map
# Ethiopic has several groups of characters that are visually/phonetically
# equivalent in modern Amharic usage but encoded differently in Unicode.
# Mapping the rare variants to the dominant one improves vocabulary coverage.
# ---------------------------------------------------------------------------
AMHARIC_NORMALIZATION_MAP: dict[str, str] = {
    # ሀ group → ሃ (ha)
    "ሀ": "ሃ", "ሁ": "ሁ", "ሂ": "ሂ", "ሄ": "ሄ", "ህ": "ህ", "ሆ": "ሆ",
    # ሐ group → ሃ
    "ሐ": "ሃ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    # ኀ group → ሃ
    "ኀ": "ሃ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    # ዐ group → አ
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "አ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    # ጸ / ፀ (both represent ts')
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
}

# Common Amharic stopwords relevant to social-media / policy text
AMHARIC_STOPWORDS: set[str] = {
    "እና", "ና", "ወይ", "ወይም", "ግን", "ስለ", "ስለዚህ", "ምክንያቱም",
    "ነው", "ናቸው", "ነበር", "ነበሩ", "ይሆናል", "ሆኖ", "ሆናለች",
    "አይደለም", "አይደሉም", "አልነበረም",
    "ይህ", "ያ", "እነዚህ", "እነዚያ", "እዚህ", "እዚያ",
    "የ", "ው", "ት", "ም", "ን", "ና",
    "ሁሉ", "ሁሉም", "አንድ", "አንዲት",
    "ሰው", "ሰዎች", "ሃገር", "ሀገር",
    "ነው።", "ናቸው።", "ወዘተ",
    "እንደ", "እንዲሁ", "እንዲሁም", "ብቻ", "ደግሞ",
    "ጋር", "ላይ", "ውስጥ", "ወደ", "ከ", "እስከ", "በ", "ለ",
    "ቢሆን", "ቢሆንም", "እንኳ", "እንኳን",
    "አሁን", "ዛሬ", "ትናንት", "ነገ",
    "RT",  # retweet marker common in Twitter data
}


def normalize_amharic_chars(text: str) -> str:
    """Replace visually/phonetically equivalent Ethiopic characters with canonical forms."""
    for variant, canonical in AMHARIC_NORMALIZATION_MAP.items():
        text = text.replace(variant, canonical)
    return text


def remove_noise(text: str) -> str:
    """Strip URLs, @mentions, #hashtags, numbers, punctuation and extra whitespace."""
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                       # mentions
    text = re.sub(r"#\w+", " ", text)                       # hashtags
    text = re.sub(r"\d+", " ", text)                        # digits
    # Keep Ethiopic chars, Latin letters, Ethiopic word-separator (፡ U+1361) and space
    text = re.sub(r"[^ሀ-፿ᎀ-᎟ⶀ-⷟\w\s፡]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(tokens: List[str], extra: Optional[set] = None) -> List[str]:
    """Filter Amharic (and optional extra) stopwords from a token list."""
    stop = AMHARIC_STOPWORDS | (extra or set())
    return [t for t in tokens if t not in stop]


def tokenize(text: str) -> List[str]:
    """
    Whitespace tokenizer for Amharic.

    Amharic sentences use the Ethiopic full stop (።) and word separator (፡).
    We split on whitespace after cleaning so these act as natural boundaries.
    """
    return text.split()


def preprocess_text(
    text: str,
    remove_sw: bool = True,
    extra_stopwords: Optional[set] = None,
) -> List[str]:
    """Full preprocessing pipeline: normalize → clean → tokenize → stopword removal."""
    if not isinstance(text, str) or not text.strip():
        return []
    text = unicodedata.normalize("NFC", text)
    text = normalize_amharic_chars(text)
    text = remove_noise(text)
    tokens = tokenize(text)
    if remove_sw:
        tokens = remove_stopwords(tokens, extra_stopwords)
    return tokens


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "tweet",
    label_col: str = "label",
    remove_sw: bool = True,
) -> pd.DataFrame:
    """
    Apply the full pipeline to a DataFrame column and return a clean copy.

    Expects AfriSenti-style TSV columns: tweet (text) and label (sentiment).
    Labels: positive, negative, neutral.
    """
    df = df.copy()
    df["tokens"] = df[text_col].map(
        lambda t: preprocess_text(t, remove_sw=remove_sw)
    )
    df["cleaned_text"] = df["tokens"].map(lambda toks: " ".join(toks))
    df = df[df["cleaned_text"].str.strip() != ""]  # drop empty rows after cleaning
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CLI entry point: run on AfriSenti AMH splits
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Preprocess Amharic AfriSenti splits")
    parser.add_argument("--input_dir", default="data/raw/amh", help="Directory with train/dev/test.tsv")
    parser.add_argument("--output_dir", default="data/processed/amh", help="Output directory")
    parser.add_argument("--no_stopwords", action="store_true", help="Skip stopword removal")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "test"):
        src = input_dir / f"{split}.tsv"
        if not src.exists():
            print(f"  [skip] {src} not found")
            continue
        df = pd.read_csv(src, sep="\t", header=None, names=["tweet", "label"])
        df_clean = preprocess_dataframe(df, remove_sw=not args.no_stopwords)
        dst = output_dir / f"{split}_processed.tsv"
        df_clean.to_csv(dst, sep="\t", index=False)
        print(f"  {split}: {len(df)} → {len(df_clean)} rows → {dst}")

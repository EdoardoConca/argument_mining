
import torch
from torch import nn
from utils.utils_notebooks import convert_ids_to_tags
from collections import Counter
import torch
from transformers import AutoTokenizer
import pandas as pd
from typing import *
from tqdm import tqdm

def sequence_tagging_with_positions(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    id_to_tag: Dict[int, str],
    input_text: str,
    device: str
) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
    """
    Performs sequence tagging on input text and retrieves start and end positions.

    Args:
        model: The sequence tagging model
        tokenizer: Tokenizer to use
        id_to_tag: Mapping of IDs to tags
        input_text: Text to analyze
        device: Device to run model on

    Returns:
        Tuple of tokens, their predicted tags, and start-end positions
    """
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        _, tag_seq = model(input_ids=input_ids, attention_mask=attention_mask)

    # Convert token IDs to tokens and get predicted tags
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    tags = convert_ids_to_tags(tag_seq, id_to_tag)

    # Get start and end positions from predictions
    start_end_positions = []
    current_start = None

    for idx, token in enumerate(tokens):
        # Get the corresponding prediction for this token
        pred = tags[0][idx]

        if pred.startswith("B-"):
            if current_start is not None:
                start_end_positions.append((current_start, idx - 1))
            current_start = idx

        elif pred.startswith("O"):
            if current_start is not None:
                start_end_positions.append((current_start, idx - 1))
                current_start = None

    if current_start is not None:
        start_end_positions.append((current_start, len(tokens) - 1))

    return tokens, tags, start_end_positions



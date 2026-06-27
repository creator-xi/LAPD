"""
Text processing utilities for ablation studies.
"""
from typing import List
import re


def truncate_text_to_word_count(text: str, max_words: int) -> str:
    """
    Truncate text from the beginning to the specified word count.

    Args:
        text: Input text string
        max_words: Maximum number of words

    Returns:
        Truncated text

    Examples:
        >>> text = "This is a sample text for testing."
        >>> truncate_text_to_word_count(text, 5)
        'This is a sample text'
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])


def truncate_batch_texts(texts: List[str], max_words: int) -> List[str]:
    """
    Batch truncate a list of texts.

    Args:
        texts: List of texts
        max_words: Maximum number of words

    Returns:
        List of truncated texts

    Examples:
        >>> texts = ["Short text", "This is a much longer text"]
        >>> truncate_batch_texts(texts, 3)
        ['Short text', 'This is a much']
    """
    return [truncate_text_to_word_count(text, max_words) for text in texts]


def truncate_data_dict(data: dict, max_words: int, keys: List[str] = None) -> dict:
    """
    Truncate texts in a standard data dict (for ablation studies).

    Args:
        data: Standard format data dict {'original': [...], 'sampled': [...]}
        max_words: Maximum number of words
        keys: Keys to truncate, defaults to ['original', 'sampled']

    Returns:
        Truncated data dict

    Examples:
        >>> data = {'original': ['Long human text here...'], 'sampled': ['Long AI text here...']}
        >>> truncated_data = truncate_data_dict(data, 10)
    """
    if keys is None:
        keys = ['original', 'sampled']

    truncated_data = {}
    for key in keys:
        if key in data and isinstance(data[key], list):
            truncated_data[key] = truncate_batch_texts(data[key], max_words)

    for key in data:
        if key not in keys:
            truncated_data[key] = data[key]

    return truncated_data

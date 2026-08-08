"""
Metrics for evaluating generation quality.

Three axes of measurement:
1. Repetition — n-gram overlap (are tokens repeating?)
2. Self-similarity — sliding window Jaccard (are passages repeating semantically?)
3. Degeneration detection — sustained low-entropy collapse
4. Semantic coherence — embedding-based topic continuity (NEW)
"""

from typing import Optional


def ngram_repetition_rate(token_ids: list[int], n: int = 4) -> float:
    """
    Fraction of n-grams that are repeated in the token sequence.

    Returns 0.0 for fully unique sequences, approaches 1.0 for
    degenerate repetition loops.

    Args:
        token_ids: List of token IDs from generation
        n: N-gram size (4 catches phrase-level, 8 catches sentence-level)

    Returns:
        Repetition rate in [0, 1]
    """
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    if not ngrams:
        return 0.0
    unique = len(set(ngrams))
    total = len(ngrams)
    return 1.0 - (unique / total)


def sliding_self_similarity(
    text: str, window: int = 200, stride: int = 100
) -> list[float]:
    """
    Compute word-level Jaccard similarity between consecutive sliding windows.

    Cheap proxy for semantic repetition without needing an embedding model.
    High values indicate the model is cycling through the same concepts.

    Args:
        text: Generated text
        window: Window size in words
        stride: Step size between windows

    Returns:
        List of Jaccard similarities between consecutive windows.
    """
    similarities = []
    words = text.split()
    for i in range(0, max(1, len(words) - window), stride):
        w1 = set(words[i : i + window])
        w2 = set(words[i + stride : i + stride + window])
        if not w1 or not w2:
            continue
        jaccard = len(w1 & w2) / len(w1 | w2)
        similarities.append(jaccard)
    return similarities


def find_degeneration_point(
    entropies: list[float], threshold: float = 0.3, window: int = 20
) -> Optional[int]:
    """
    Find the generation step where entropy collapses.

    Collapse = entropy drops below `threshold` for `window` consecutive steps.
    This indicates the model has entered a repetition loop.

    Args:
        entropies: Entropy values at each generation step
        threshold: Entropy below this is considered collapsed
        window: Must sustain for this many steps to count

    Returns:
        Step index where collapse began, or None if it never collapsed.
    """
    count = 0
    for i, h in enumerate(entropies):
        if h < threshold:
            count += 1
            if count >= window:
                return i - window + 1
        else:
            count = 0
    return None


# ═══════════════════════════════════════════════════════════════════
# SEMANTIC COHERENCE METRIC (NEW - requires sentence-transformers)
# ═══════════════════════════════════════════════════════════════════

def semantic_coherence_score(
    text: str, 
    window: int = 200, 
    stride: int = 150,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Optional[float]:
    """
    Compute semantic coherence using sentence embeddings.
    
    Measures how semantically related consecutive passages are.
    Higher scores indicate more coherent transitions between topics.
    
    Args:
        text: Generated text to analyze
        window: Window size in words for passage segmentation
        stride: Overlap between windows (higher = more overlap)
        model_name: Pretrained sentence transformer model
    
    Returns:
        Average cosine similarity between consecutive passages, or None if unavailable.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        model = SentenceTransformer(model_name)
        
        # Split text into overlapping windows
        words = text.split()
        if len(words) < window:
            return None
            
        passages = []
        for i in range(0, max(window, len(words) - window), stride):
            passage_words = words[i:i+window]
            if len(passage_words) >= 10:  # Minimum meaningful passage
                passages.append(" ".join(passage_words))
        
        if len(passages) < 2:
            return None
            
        # Compute embeddings and similarities
        embeddings = model.encode(passages, convert_to_tensor=True)
        similarities = []
        
        for i in range(len(embeddings) - 1):
            sim = torch.cosine_similarity(
                embeddings[i].unsqueeze(0), 
                embeddings[i+1].unsqueeze(0)
            ).item()
            similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else None
        
    except ImportError:
        print("  [semantic coherence requires sentence-transformers]")
        return None


# ═══════════════════════════════════════════════════════════════════
# EXTENDED METRIC SETTER (for longer generation experiments)
# ═══════════════════════════════════════════════════════════════════

def find_degeneration_point_extended(
    entropies: list[float], 
    threshold: float = 0.3, 
    window: int = 30,
    early_threshold: Optional[float] = None,
    early_window: int = 15
) -> dict:
    """
    Extended degeneration detection with multiple thresholds.
    
    Returns a dict with both early and late collapse points.
    Useful for longer generations where different failure modes may occur.
    
    Args:
        entropies: Entropy values at each generation step
        threshold: Primary entropy threshold for collapse
        window: Steps required to confirm collapse (primary)
        early_threshold: Optional lower threshold for early detection
        early_window: Steps required for early collapse confirmation
    
    Returns:
        Dict with 'early' and 'late' keys, values are step indices or None.
    """
    def _find_collapse(entropies, threshold, window):
        count = 0
        for i, h in enumerate(entropies):
            if h < threshold:
                count += 1
                if count >= window:
                    return i - window + 1
            else:
                count = 0
        return None
    
    result = {"early": None, "late": None}
    
    # Check for early collapse (more lenient threshold)
    if early_threshold is not None:
        result["early"] = _find_collapse(entropies, early_threshold, early_window)
    
    # Check for late collapse (primary threshold)
    result["late"] = _find_collapse(entropies, threshold, window)
    
    return result
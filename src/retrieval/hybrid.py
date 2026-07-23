def reciprocal_rank_fusion(rankings_list, k=60):
    """Combines multiple rank lists using Reciprocal Rank Fusion (RRF)."""
    rrf_scores = {}
    for rankings in rankings_list:
        for rank, doc_idx in enumerate(rankings):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank + 1.0))
            
    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return sorted_indices, rrf_scores
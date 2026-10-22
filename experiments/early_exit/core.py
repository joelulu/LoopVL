"""Dependency-free greedy draft/verify protocol and diagnostic summaries."""
from __future__ import annotations

from collections.abc import Callable, Sequence


def draft_verify(
    propose: Callable[[list[int]], int],
    verify: Callable[[list[int], list[int]], Sequence[int]],
    *,
    block_size: int,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> tuple[list[int], list[dict]]:
    """Verify K proposals against K+1 causal target predictions.

    verify returns predictions at [last-prefix, draft[0], ..., draft[K-1]].
    Only the matching prefix and the first correction/bonus can be committed.
    Callbacks receive copies, so a rejected branch cannot mutate committed state.
    """
    if block_size < 1 or max_new_tokens < 1:
        raise ValueError("block_size and max_new_tokens must be positive")
    committed: list[int] = []
    rounds: list[dict] = []
    while len(committed) < max_new_tokens:
        prefix = committed.copy()
        drafts: list[int] = []
        for _ in range(min(block_size, max_new_tokens - len(prefix))):
            token = int(propose(prefix + drafts))
            drafts.append(token)
            if token == eos_token_id:
                break
        target = list(verify(prefix.copy(), drafts.copy()))
        if len(target) != len(drafts) + 1:
            raise ValueError("verifier must return exactly K+1 predictions")
        accepted = 0
        while accepted < len(drafts) and drafts[accepted] == target[accepted]:
            accepted += 1
        advance = drafts[:accepted] + [int(target[accepted])]
        advance = advance[: max_new_tokens - len(prefix)]
        if eos_token_id is not None and eos_token_id in advance:
            advance = advance[: advance.index(eos_token_id) + 1]
        committed.extend(advance)
        rounds.append({
            "prefix_length": len(prefix), "draft_ids": drafts,
            "target_ids": target, "matched_prefix": accepted,
            "accepted_draft_tokens": min(accepted, len(advance)),
            "committed_ids": advance,
        })
        if committed[-1] == eos_token_id:
            break
    return committed, rounds


def summarize(rows: list[dict], generations: list[dict]) -> dict:
    """Token-weighted agreement; actual sample/token counts, never nominal counts."""
    groups = sorted({str(g["category"]) for g in generations})
    output = {}
    for group in ["__all__", *groups]:
        samples = generations if group == "__all__" else [g for g in generations if str(g["category"]) == group]
        ids = {g["id"] for g in samples}
        tokens = [r for r in rows if r["id"] in ids]
        n = len(tokens)
        confident = [r for r in tokens if r["h1_confidence"] >= r["confidence_threshold"]]
        output[group] = {
            "samples": len(samples), "generated_tokens": sum(len(g["token_ids"]) for g in samples),
            "compared_tokens": n,
            "h1_h2_agreement": sum(r["h1_id"] == r["h2_id"] for r in tokens) / n if n else None,
            "h2_replay_mismatches": sum(r["h2_id"] != r["reference_id"] for r in tokens),
            "high_confidence_tokens": len(confident),
            "high_confidence_disagreement_rate": sum(r["h1_id"] != r["h2_id"] for r in confident) / len(confident) if confident else None,
        }
    return output

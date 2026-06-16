"""Helpers for normalizing image captions into one consistent form.

BLIP captions (from ``scripts/caption_dataset.py``) are stylistically inconsistent and
carry artifacts -- filler like ``there is a`` / ``they are``, missing leading articles,
and the BLIP ``arafed`` hallucination. ``scripts/normalize_captions.py`` rewrites each one
into a single clean natural phrase with a small instruction-tuned LLM.

This module holds the deterministic, model-free pieces (the prompt and the post-processing
cleanup) so they can be unit-tested without downloading the model.
"""

from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You rewrite messy image captions into one clean, consistent caption. Rules:\n"
    "- Output a single short descriptive phrase about the person and scene.\n"
    "- Start with the subject; use an article ('a'/'an') when it is singular.\n"
    "- Remove filler such as 'there is', 'there are', 'they are'.\n"
    "- Remove nonsense or non-words such as 'arafed'.\n"
    "- Fix grammar, but stay faithful: do not invent details that are not in the input.\n"
    "- Use all lowercase, no trailing period, and no surrounding quotes.\n"
    "Output only the rewritten caption, nothing else."
)

# Few-shot pairs covering the recurring BLIP patterns. Kept here (not in the script) so the
# prompt is versioned with the cleanup rules it mirrors.
_FEWSHOT: list[tuple[str, str]] = [
    (
        "there is a woman sitting at a table with a plate of food",
        "a woman sitting at a table with a plate of food",
    ),
    (
        "smiling woman with blue earrings and green top sitting at a table",
        "a smiling woman with blue earrings and a green top sitting at a table",
    ),
    (
        "arafed man with a black shirt and a tie looking at the camera",
        "a man with a black shirt and a tie looking at the camera",
    ),
    (
        "they are two men talking to each other while wearing glasses",
        "two men talking to each other while wearing glasses",
    ),
]


def build_messages(raw: str) -> list[dict]:
    """Chat messages (system + few-shot + the new caption) for ``apply_chat_template``."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for src, dst in _FEWSHOT:
        messages.append({"role": "user", "content": src})
        messages.append({"role": "assistant", "content": dst})
    messages.append({"role": "user", "content": raw})
    return messages


# Leading filler the model is told to drop; also stripped here as a safety net. We strip the
# filler verb but keep any following article ("there is a woman" -> "a woman"); the model is
# what restores a missing article, so the deterministic pass never invents one.
_LEADING_FILLER = re.compile(
    r"^(?:there\s+(?:is|are)\s+|they\s+are\s+|arafed\s+)",
    flags=re.IGNORECASE,
)
_ARAFED = re.compile(r"\barafed\b\s*", flags=re.IGNORECASE)


def clean_caption(text: str) -> str:
    """Deterministic post-processing for a (model or raw) caption.

    Lowercases, strips surrounding quotes/whitespace, collapses inner whitespace, removes a
    trailing period, and strips residual leading filler / ``arafed`` the LLM may have left.
    Idempotent: a caption already in canonical form is returned unchanged.
    """
    text = text.strip().strip("\"'").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    # Strip residual artifacts the model was supposed to remove.
    text = _LEADING_FILLER.sub("", text)
    text = _ARAFED.sub("", text)
    text = text.strip().rstrip(".").strip()
    return text

from imagegen.captions import SYSTEM_PROMPT, build_messages, clean_caption


def test_strips_leading_filler():
    assert clean_caption("there is a woman sitting at a table") == "a woman sitting at a table"
    assert clean_caption("there are two women posing") == "two women posing"
    assert clean_caption("they are two men talking") == "two men talking"


def test_removes_arafed_artifact():
    # The deterministic pass strips the artifact but never invents a leading article
    # (that is the model's job); it just must not leave "arafed" behind.
    assert clean_caption("arafed man with a black shirt and a tie") == (
        "man with a black shirt and a tie"
    )
    assert clean_caption("a man arafed wearing a hat") == "a man wearing a hat"


def test_lowercases_and_strips_trailing_period_and_quotes():
    assert clean_caption('"A Woman Wearing A Pink Hat."') == "a woman wearing a pink hat"


def test_collapses_whitespace():
    assert clean_caption("a   man   with\n a  hat") == "a man with a hat"


def test_already_clean_is_unchanged():
    clean = "a smiling woman with blue earrings and a green top"
    assert clean_caption(clean) == clean


def test_build_messages_has_system_and_ends_with_raw():
    raw = "there is a young girl with a toothbrush in her hand"
    messages = build_messages(raw)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[-1] == {"role": "user", "content": raw}
    # few-shot pairs sit between system and the final user turn
    assert len(messages) >= 4 and messages[-1]["role"] == "user"

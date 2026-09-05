"""Speech normalisation.

Regression cover for a live failure: asked to compare an LLM and a VLM, the
model answered in markdown and Piper read the syntax aloud -- "star star",
"hash hash" -- through an otherwise correct answer.
"""

from __future__ import annotations

import pytest

from aura.voice.tts.normalize import for_speech


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("**LLM (Large Language Model)**", "LLM (Large Language Model)"),
        ("*   **Input:** Primarily text.", "Input: Primarily text."),
        ("### Key Difference", "Key Difference"),
        ("# Heading", "Heading"),
        ("- a bullet", "a bullet"),
        ("+ another bullet", "another bullet"),
        ("• unicode bullet", "unicode bullet"),
        ("`model.py`", "model.py"),
        ("~~struck~~", "struck"),
        ("_emphasis_", "emphasis"),
        ("> quoted line", "quoted line"),
        ("[the docs](https://example.com)", "the docs"),
        ("![alt text](img.png)", "alt text"),
    ],
)
def test_markup_is_removed(written: str, spoken: str) -> None:
    assert for_speech(written) == spoken


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("text & images", "text and images"),
        ("input -> output", "input to output"),
        ("a => b", "a to b"),
        ("b <- a", "b from a"),
        ("text/images", "text or images"),
    ],
)
def test_symbols_become_words(written: str, spoken: str) -> None:
    assert for_speech(written) == spoken


def test_no_markup_characters_survive() -> None:
    """The point of the exercise: nothing a synthesiser would read as a symbol."""
    messy = "**bold** ### head *item* `code` ~~old~~ | table | and *stray"
    result = for_speech(messy)
    for char in ("*", "#", "`", "|", "~~"):
        assert char not in result, f"{char!r} survived in {result!r}"


def test_code_fences_are_dropped_entirely() -> None:
    """A spoken answer that needs a code block is the wrong answer."""
    text = "Try this:\n```python\nfor x in range(10):\n    print(x)\n```\nThat works."
    result = for_speech(text)
    assert "print" not in result
    assert "That works." in result


def test_bare_urls_are_dropped() -> None:
    """Read character by character a URL is noise."""
    assert "example.com" not in for_speech("See https://example.com/a/b?c=d for more")


def test_nothing_speakable_returns_empty() -> None:
    """The caller treats this as 'skip', not as an error."""
    for junk in ("***", "---", "###", "|  |  |", "   ", ""):
        assert for_speech(junk) == "", junk


def test_ordinary_prose_is_untouched() -> None:
    plain = "Paris is the capital of France. It has about 2.1 million people."
    assert for_speech(plain) == plain


def test_the_reply_that_prompted_this() -> None:
    """The actual answer that was read aloud with its syntax intact."""
    written = (
        "**LLM (Large Language Model)**\n"
        "*   **Input:** Primarily text (tokens).\n"
        "*   **Examples:** GPT-4, Llama 3.\n\n"
        "**Key Difference:** LLMs understand text; VLMs understand both."
    )
    result = for_speech(written)
    assert "*" not in result
    assert "LLM (Large Language Model)" in result
    assert "Key Difference: LLMs understand text; VLMs understand both." in result

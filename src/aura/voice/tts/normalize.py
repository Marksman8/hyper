"""Turn written text into something worth saying out loud.

Models write for the eye. Asked to compare two things they produce headings,
bold runs and bullet lists, and a speech synthesiser reads that literally --
observed live as AURA saying "hash hash" and "star star" through an otherwise
correct answer.

Two defences, and both are needed:

* The system prompt asks for spoken prose when the reply will be spoken
  (`persona.voice_system_prompt`). That fixes the *shape* of the answer, so it
  does not arrive as a spec sheet.
* This module strips whatever markup arrives anyway. Prompting is a request,
  not a guarantee, and a 4B model will forget under pressure.

It also strips LaTeX delimiters, which a model reaches for unprompted the
moment the subject is complexity or maths: `$O(1)$` was read out as "dollar O
of one dollar". The contents survive -- "O(1)" is said perfectly well; only the
fence is noise.

Deliberately conservative: it removes markup and expands symbols that have an
unambiguous spoken form. It does not try to be a text normaliser for numbers,
dates or units -- Piper already handles those, and guessing badly is worse
than leaving them alone.
"""

from __future__ import annotations

import re

# Fenced code blocks. Read aloud these are unbearable, and a spoken answer that
# needs one is already the wrong answer -- say that instead.
_CODE_FENCE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")

# Inline markup, in the order that matters: links before emphasis, so the label
# survives; bold before italic, so `**x**` does not leave a stray star.
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BOLD_ITALIC = re.compile(r"\*{1,3}([^*\n]+?)\*{1,3}")
_UNDERSCORE = re.compile(r"(?<![A-Za-z0-9])_{1,3}([^_\n]+?)_{1,3}(?![A-Za-z0-9])")
_INLINE_CODE = re.compile(r"`+([^`\n]+?)`+")
_STRIKE = re.compile(r"~~([^~\n]+?)~~")

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_HRULE = re.compile(r"^\s{0,3}(?:[-*_]\s*){3,}$", re.MULTILINE)
_BULLET = re.compile("^\\s*[-*+\u2022]\\s+", re.MULTILINE)
# Table pipes and separator rows.
_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]{5,}\|?\s*$", re.MULTILINE)
_PIPES = re.compile(r"\s*\|\s*")

_URL = re.compile(r"https?://\S+|www\.\S+")

# LaTeX math delimiters. A model asked about complexity writes $O(1)$ and the
# dollars are read aloud -- "dollar O of one dollar". Display math ($$..$$)
# first, so its pair is not mistaken for two inline spans. The contents are
# kept: "O(1)" is said perfectly well, it is only the fence that is noise.
_MATH_DISPLAY = re.compile(r"\$\$([^$]+?)\$\$", re.S)
_MATH_INLINE = re.compile(r"\$([^$\n]+?)\$")
# \(..\) and \[..\], the other spelling.
_MATH_PAREN = re.compile(r"\\[([]\s*(.+?)\s*\\[)\]]", re.S)

# Anything outside the Basic Multilingual Plane is almost always an emoji here.
_EMOJI = re.compile("[\U0001f000-\U0001ffff\U00002600-\U000027bf\U0001f900-\U0001f9ff️]+")

# Symbols with one obvious spoken form. Kept short on purpose: every entry is a
# chance to be wrong, and Piper handles most punctuation sensibly already.
_SYMBOLS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\s*&\s*"), " and "),
    (re.compile("\\s*->\\s*|\\s*=>\\s*|\\s*\u2192\\s*"), " to "),
    (re.compile("\\s*<-\\s*|\\s*\u2190\\s*"), " from "),
    (re.compile(r"(?<=\w)\s*/\s*(?=\w)"), " or "),
    (re.compile("\\s*\\.\\.\\.\\s*|\u2026"), ", "),
    (re.compile("[\u2018\u2019]"), "'"),  # curly single quotes
    (re.compile("[\u201c\u201d]"), ""),  # curly double quotes
    (re.compile("[\u2013\u2014]"), ", "),  # en/em dash
)

_MULTISPACE = re.compile(r"[ \t]{2,}")
_BLANKLINES = re.compile(r"\n{2,}")
# A run of leftover punctuation reads as a stutter of clicks.
_PUNCT_RUN = re.compile(r"([,.;:!?])\1+")
_ORPHAN_PUNCT = re.compile(r"^\s*[,.;:]\s*", re.MULTILINE)


def for_speech(text: str) -> str:
    """Strip markup and expand symbols, leaving words a voice can read.

    Returns an empty string when nothing speakable is left, which the caller
    treats as "skip this chunk" rather than an error -- a bullet marker on its
    own line is legitimately nothing to say.
    """
    if not text:
        return ""

    out = _CODE_FENCE.sub(" ", text)

    out = _IMAGE.sub(r"\1", out)
    out = _LINK.sub(r"\1", out)
    out = _INLINE_CODE.sub(r"\1", out)
    out = _STRIKE.sub(r"\1", out)
    out = _BOLD_ITALIC.sub(r"\1", out)
    out = _UNDERSCORE.sub(r"\1", out)

    out = _HRULE.sub(" ", out)
    out = _HEADING.sub("", out)
    out = _BLOCKQUOTE.sub("", out)
    out = _TABLE_SEP.sub(" ", out)
    out = _BULLET.sub("", out)
    out = _PIPES.sub(", ", out)

    # A bare URL spoken character by character is noise; the label from a
    # markdown link has already been kept above.
    out = _URL.sub("", out)
    out = _EMOJI.sub("", out)

    out = _MATH_DISPLAY.sub(r" \1 ", out)
    out = _MATH_PAREN.sub(r" \1 ", out)
    out = _MATH_INLINE.sub(r" \1 ", out)

    for pattern, replacement in _SYMBOLS:
        out = pattern.sub(replacement, out)

    # Any asterisk, hash or stray delimiter still standing was not part of
    # valid markup.
    out = out.replace("*", "").replace("#", "").replace("`", "")
    out = out.replace("$", "")

    out = _BLANKLINES.sub("\n", out)
    out = _ORPHAN_PUNCT.sub("", out)
    out = _PUNCT_RUN.sub(r"\1", out)
    out = _MULTISPACE.sub(" ", out)
    out = "\n".join(line.strip() for line in out.splitlines())
    out = out.strip()

    # Nothing with a letter or digit in it is worth synthesising.
    return out if re.search(r"[A-Za-z0-9]", out) else ""

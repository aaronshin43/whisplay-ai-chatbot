import re

_SENTENCE_RE = re.compile(r'.*?([。！？!?，,]|\.)(?=\s|$)', re.DOTALL)
_EMOJI_SPECIAL_RE = re.compile(r'[*#~]|[\U0001F300-\U0001FAFF\u200d\ufe0f]')


def split_sentences(text: str) -> tuple[list[str], str]:
    """Split text at sentence boundaries. Returns (complete_sentences, remaining_buffer).

    Ported from src/utils/index.ts splitSentences().
    - Boundaries: . ! ? , followed by whitespace or end of string
    - Merges short sentences (<=60 chars combined) for natural TTS pacing
    """
    sentences = []
    last_index = 0
    for match in _SENTENCE_RE.finditer(text):
        sentence = match.group(0).strip()
        if sentence:
            sentences.append(sentence)
            last_index = match.end()

    remaining = text[last_index:].strip()

    # Merge short sentences (<=60 chars) — matches TS implementation
    merged = []
    buf = ""
    for s in sentences:
        candidate = f"{buf}{s} "
        if len(candidate) <= 60:
            buf = candidate
        else:
            if buf:
                merged.append(buf.rstrip())
            buf = f"{s} "
    if buf:
        merged.append(buf.rstrip())

    return merged, remaining


def purify_for_tts(text: str) -> str:
    """Remove characters unsuitable for TTS (emojis, markdown chars).

    Ported from src/utils/index.ts purifyTextForTTS().
    """
    return _EMOJI_SPECIAL_RE.sub("", text).strip()

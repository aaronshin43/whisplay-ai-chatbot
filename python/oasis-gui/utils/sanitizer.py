import re


def sanitize_chunk(chunk: str) -> str:
    """Strip markdown formatting from LLM token chunks.

    Ported from OasisAdapter.ts sanitizeOasisChunk().
    Applied to every token before display and before TTS sentence buffer.
    """
    chunk = re.sub(r'\*+', '', chunk)                      # **bold** / *italic*
    chunk = re.sub(r'`+', '', chunk)                        # `code` backticks
    chunk = re.sub(r'^#+\s*', '', chunk, flags=re.MULTILINE)  # ### headings
    return chunk

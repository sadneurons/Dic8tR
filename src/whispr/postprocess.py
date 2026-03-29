"""Post-processing pipeline: punctuation commands, vocab corrections, capitalisation.

Transforms raw Whisper transcript into properly formatted, punctuated text
with domain-specific vocabulary corrections. Operates as a sequential pipeline
of transformations on the raw string.
"""

import logging
import re

logger = logging.getLogger(__name__)


# --- Editing commands (checked first, may short-circuit) ---

EDITING_COMMANDS = {
    "scratch that": "SCRATCH_THAT",
    "scratch word": "SCRATCH_WORD",
}

# --- Punctuation / formatting commands ---
# Order matters: longer phrases must match before shorter ones.
# Each entry: (pattern, replacement, capitalise_next)

# Trailing [.,!?]? on each pattern absorbs punctuation Whisper may have added
# after the spoken command (e.g. "full stop." → just ".")
PUNCTUATION_COMMANDS: list[tuple[str, str, bool]] = [
    (r"\bnew paragraph[.,]?\b", "\n\n", True),
    (r"\bnew line[.,]?\b", "\n", True),
    (r"\bfull stop[.,]?", ".", True),
    (r"\bperiod[.,]?", ".", True),
    (r"\bquestion mark[.?]?", "?", True),
    (r"\bexclamation mark[.!]?", "!", True),
    (r"\bsemicolon[.,;]?", ";", False),
    (r"\bcolon[.,:]?", ":", False),
    (r"\bcomma[.,]?", ",", False),
    (r"\bopen bracket[.,]?", "(", False),
    (r"\bclose bracket[.,]?", ")", False),
    (r"\bhyphen[.,-]?", "-", False),
    (r"\bopen quotes?[.,]?", '"', False),
    (r"\bclose quotes?[.,]?", '"', False),
    (r"\bquote[- ]unquote[.,]?", '"', False),  # wraps next utterance — see note
]

# NOTE: "quote unquote" currently inserts a bare " mark. A future enhancement
# (stateful mode) will open quotes that stay open until key release, then
# auto-close. For now the user can say "open quote ... close quote" for
# explicit control.


def postprocess(
    text: str,
    corrections: dict[str, str] | None = None,
    expansions: dict[str, str] | None = None,
    enable_punctuation: bool = True,
    enable_editing: bool = True,
    enable_corrections: bool = True,
    enable_expansions: bool = True,
    enable_capitalisation: bool = True,
) -> str:
    """Run the full post-processing pipeline on raw transcript text.

    Returns the processed string, or a special command string
    ("SCRATCH_THAT" / "SCRATCH_WORD") if an editing command was detected.
    """
    if not text:
        return text

    # Step 0: Fix Whisper spacing — ensure space after sentence-ending punctuation
    text = _fix_whisper_spacing(text)

    # Step 1: Check for editing commands (must be first — may discard the buffer)
    if enable_editing:
        cmd = _check_editing_commands(text)
        if cmd:
            return cmd

    # Step 2: Vocabulary expansions (before punctuation, so "standard intro" expands whole)
    if enable_expansions and expansions:
        text = _apply_expansions(text, expansions)

    # Step 3: Punctuation and formatting commands
    if enable_punctuation:
        text = _apply_punctuation_commands(text)

    # Step 4: Vocabulary corrections
    if enable_corrections and corrections:
        text = _apply_corrections(text, corrections)

    # Step 5: Capitalisation
    if enable_capitalisation:
        text = _apply_capitalisation(text)

    return text.strip()


def _fix_whisper_spacing(text: str) -> str:
    """Fix missing spaces after punctuation in Whisper output.

    Whisper often outputs "word.Word" or "word,word" without spaces.
    Insert a space after sentence/clause punctuation when followed by a letter.
    """
    # Space after . ? ! when followed by a letter (but not inside numbers like 3.14
    # or abbreviations like pTau217)
    text = re.sub(r'([.?!])([A-Z])', r'\1 \2', text)
    # Space after , ; : ) when followed by a letter
    text = re.sub(r'([,;:)])([A-Za-z])', r'\1 \2', text)
    return text


def _check_editing_commands(text: str) -> str | None:
    """Check if the entire utterance is an editing command.

    Returns the command string if matched, None otherwise.
    """
    normalized = text.strip().lower().rstrip(".")
    for phrase, command in EDITING_COMMANDS.items():
        if normalized == phrase:
            logger.info("Editing command detected: %s", command)
            return command
    return None


def _apply_expansions(text: str, expansions: dict[str, str]) -> str:
    """Replace spoken shorthand with full expansion text."""
    for phrase, expansion in expansions.items():
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        text = pattern.sub(expansion, text)
    return text


def _apply_punctuation_commands(text: str) -> str:
    """Replace spoken punctuation commands with their symbols.

    Handles spacing: removes the space before punctuation that attaches
    to the previous word (.,!?;:) and the space after opening brackets/quotes.
    """
    for pattern, replacement, _ in PUNCTUATION_COMMANDS:
        if replacement in ("\n", "\n\n"):
            # Newlines: strip surrounding spaces
            text = re.sub(
                r'\s*' + pattern + r'\s*',
                replacement,
                text,
                flags=re.IGNORECASE,
            )
        elif replacement == "(":
            # Opening bracket: space before, no space after
            text = re.sub(
                r'\s*' + pattern + r'\s*',
                " (",
                text,
                flags=re.IGNORECASE,
            )
        elif replacement == ")":
            # Closing bracket: no space before, space after
            text = re.sub(
                r'\s*' + pattern + r'\s*',
                ") ",
                text,
                flags=re.IGNORECASE,
            )
        elif replacement == '"':
            # Quotes: detect open vs close by keyword in pattern
            if "open" in pattern:
                # Space before, no space after
                text = re.sub(
                    r'\s*' + pattern + r'\s*',
                    ' "',
                    text,
                    flags=re.IGNORECASE,
                )
            else:
                # No space before, space after
                text = re.sub(
                    r'\s*' + pattern + r'\s*',
                    '" ',
                    text,
                    flags=re.IGNORECASE,
                )
        else:
            # Most punctuation: attach to previous word, space after
            text = re.sub(
                r'\s*' + pattern + r'\s*',
                replacement + " ",
                text,
                flags=re.IGNORECASE,
            )

    # Clean up doubled punctuation (e.g. ".." -> ".", ",," -> ",")
    text = re.sub(r'([.?!,;:])\1+', r'\1', text)
    # Clean up double spaces
    text = re.sub(r'  +', ' ', text)
    return text


def _apply_corrections(text: str, corrections: dict[str, str]) -> str:
    """Apply vocabulary corrections (case-insensitive match, exact replacement)."""
    for wrong, right in corrections.items():
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        text = pattern.sub(right, text)
    return text


def _apply_capitalisation(text: str) -> str:
    """Capitalise first character and characters after sentence-ending punctuation."""
    if not text:
        return text

    result = list(text)
    capitalise_next = True

    for i, ch in enumerate(result):
        if capitalise_next and ch.isalpha():
            result[i] = ch.upper()
            capitalise_next = False
        elif ch in ".?!\n":
            capitalise_next = True

    return "".join(result)

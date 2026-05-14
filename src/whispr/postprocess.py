"""Post-processing pipeline: punctuation commands, vocab corrections, capitalisation.

Transforms raw Whisper transcript into properly formatted, punctuated text
with domain-specific vocabulary corrections. Also detects editing commands
(scratch that/word, undo, redo) and editor interaction commands (bold,
italic, save, select all, etc.) that return action strings rather than text.
"""

import logging
import re

logger = logging.getLogger(__name__)


# --- Action commands ---
# These return a special string that main.py interprets as an action
# rather than text to inject. Checked against the full normalised utterance.

ACTION_COMMANDS = {
    # Editing
    "scratch that":     "ACTION:SCRATCH_THAT",
    "scratch word":     "ACTION:SCRATCH_WORD",
    "undo":             "ACTION:UNDO",
    "undo that":        "ACTION:UNDO",
    "redo":             "ACTION:REDO",
    "redo that":        "ACTION:REDO",
    # Editor interaction
    "select all":       "ACTION:KEY:ctrl+a",
    "copy that":        "ACTION:KEY:ctrl+c",
    "cut that":         "ACTION:KEY:ctrl+x",
    "paste":            "ACTION:KEY:ctrl+v",
    "paste that":       "ACTION:KEY:ctrl+v",
    "save":             "ACTION:KEY:ctrl+s",
    "save file":        "ACTION:KEY:ctrl+s",
    "bold":             "ACTION:KEY:ctrl+b",
    "bold that":        "ACTION:KEY:ctrl+b",
    "italic":           "ACTION:KEY:ctrl+i",
    "italics":          "ACTION:KEY:ctrl+i",
    "underline":        "ACTION:KEY:ctrl+u",
    "underline that":   "ACTION:KEY:ctrl+u",
    # Navigation
    "go to top":        "ACTION:KEY:ctrl+Home",
    "go to bottom":     "ACTION:KEY:ctrl+End",
    "go to start":      "ACTION:KEY:Home",
    "go to end":        "ACTION:KEY:End",
    "page up":          "ACTION:KEY:Prior",
    "page down":        "ACTION:KEY:Next",
    # Deletion
    "delete line":      "ACTION:KEY:Home,shift+End,BackSpace",
    "delete word":      "ACTION:KEY:ctrl+BackSpace",
}


# --- Punctuation / formatting commands ---
# Each entry: (pattern, replacement, capitalise_next)
# Trailing [.,!?]? absorbs Whisper's auto-punctuation

PUNCTUATION_COMMANDS: list[tuple[str, str, bool]] = [
    (r"\bnew paragraph[.,]?\b",     "\n\n", True),
    (r"\bnew line[.,]?\b",          "\n", True),
    (r"\bnext line[.,]?\b",         "\n", True),
    (r"\bline break[.,]?\b",        "\n", True),
    (r"\bbullet point[.,]?\b",      "\n  \u2022 ", True),
    (r"\bbullet[.,]?\b",            "\n  \u2022 ", True),
    (r"\bnumbered list[.,]?\b",     "\n  1. ", True),
    (r"\bnext item[.,]?\b",         "\n  \u2022 ", True),
    (r"\btab[.,]?\b",               "\t", False),
    (r"\bindent[.,]?\b",            "\t", False),
    (r"\bfull stop[.,]?",           ".", True),
    (r"\bperiod[.,]?",             ".", True),
    (r"\bquestion mark[.?]?",      "?", True),
    (r"\bexclamation mark[.!]?",   "!", True),
    (r"\bsemicolon[.,;]?",         ";", False),
    (r"\bcolon[.,:]?",             ":", False),
    (r"\bcomma[.,]?",              ",", False),
    (r"\bopen bracket[.,]?",       "(", False),
    (r"\bclose bracket[.,]?",      ")", False),
    (r"\bhyphen[.,-]?",            "-", False),
    (r"\bdash[.,-]?",              " \u2014 ", False),
    (r"\bem dash[.,-]?",           " \u2014 ", False),
    (r"\bellipsis[.,]?",           "\u2026", False),
    (r"\bopen quotes?[.,]?",      '"', False),
    (r"\bclose quotes?[.,]?",     '"', False),
    (r"\bquote[- ]unquote[.,]?",  '"', False),
    (r"\bapostrophe[.,]?",        "'", False),
]

# --- Numeral conversion ---
_NUMERALS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
}


def postprocess(
    text: str,
    corrections: dict[str, str] | None = None,
    expansions: dict[str, str] | None = None,
    custom_commands: dict[str, str] | None = None,
    enable_punctuation: bool = True,
    enable_editing: bool = True,
    enable_corrections: bool = True,
    enable_expansions: bool = True,
    enable_capitalisation: bool = True,
) -> str:
    """Run the full post-processing pipeline on raw transcript text.

    Returns either:
    - Processed text string for injection
    - "ACTION:..." string for editor commands (handled by main.py)
    """
    if not text:
        return text

    # Step 0: Fix Whisper spacing
    text = _fix_whisper_spacing(text)

    # Step 1: Check for action commands (editing, editor interaction)
    if enable_editing:
        cmd = _check_action_commands(text, custom_commands)
        if cmd:
            return cmd

    # Step 2: Vocabulary expansions
    if enable_expansions and expansions:
        text = _apply_expansions(text, expansions)

    # Step 3: Numeral conversion ("numeral seven" → "7")
    if enable_punctuation:
        text = _apply_numerals(text)

    # Step 4: Punctuation and formatting commands
    if enable_punctuation:
        text = _apply_punctuation_commands(text)

    # Step 5: Vocabulary corrections
    if enable_corrections and corrections:
        text = _apply_corrections(text, corrections)

    # Step 6: Capitalisation
    if enable_capitalisation:
        text = _apply_capitalisation(text)

    return text.strip()


def _fix_whisper_spacing(text: str) -> str:
    """Fix missing spaces after punctuation in Whisper output."""
    text = re.sub(r'([.?!])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([,;:)])([A-Za-z])', r'\1 \2', text)
    return text


# Filler / politeness tokens that may surround an action command in natural
# speech ("scratch that please", "uh, scratch that"). Stripped during command
# matching only — they don't affect the dictated text downstream.
_LEADING_FILLERS = frozenset({"um", "uh", "well", "so", "okay", "ok", "please"})
_TRAILING_FILLERS = frozenset({"please", "thanks"})
# Multi-word trailing fillers handled separately.
_TRAILING_MULTIWORD = ("thank you",)


def _strip_filler(text: str) -> str:
    """Remove leading/trailing filler tokens for command matching.

    Operates on the lower-cased, comma-stripped form so the caller can pass
    in already-normalised text. Returns the inner phrase only — does not
    modify text injected downstream.
    """
    tokens = text.replace(",", " ").split()
    while tokens and tokens[0] in _LEADING_FILLERS:
        tokens.pop(0)
    while tokens:
        # Multi-word trailing match first ("thank you")
        joined = " ".join(tokens[-2:]) if len(tokens) >= 2 else ""
        if joined in _TRAILING_MULTIWORD:
            tokens.pop()
            tokens.pop()
        elif tokens[-1] in _TRAILING_FILLERS:
            tokens.pop()
        else:
            break
    return " ".join(tokens)


def _check_action_commands(text: str, custom_commands: dict[str, str] | None = None) -> str | None:
    """Check if the utterance is (or wraps) an action command.

    Strips common leading/trailing fillers ("uh, scratch that please" →
    "scratch that") so natural speech still matches. Custom commands take
    priority over built-ins.
    """
    normalized = _strip_filler(text.strip().lower().rstrip("."))

    # Custom voice commands
    if custom_commands:
        for phrase, action in custom_commands.items():
            if normalized == phrase.lower():
                # Phrase is user-defined; may contain identifiers. DEBUG only.
                logger.debug("Custom command: '%s' -> %s", phrase, action)
                return action

    # Built-in action commands
    for phrase, action in ACTION_COMMANDS.items():
        if normalized == phrase:
            logger.info("Action command: %s", action)
            return action

    return None


def _apply_expansions(text: str, expansions: dict[str, str]) -> str:
    """Replace spoken shorthand with full expansion text — whole-phrase only.

    Uses non-word lookarounds (rather than \\b) so phrases that begin or end
    with non-word characters (e.g. "e.g.") still match. Without this guard
    a shorthand like "ad" would rewrite "add", "advance", etc.
    """
    for phrase, expansion in expansions.items():
        pattern = re.compile(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', re.IGNORECASE)
        text = pattern.sub(expansion, text)
    return text


def _apply_numerals(text: str) -> str:
    """Convert 'numeral <word>' to digit form."""
    def _replace_numeral(match):
        word = match.group(1).lower()
        return _NUMERALS.get(word, match.group(0))

    text = re.sub(
        r'\bnumeral\s+(\w+)',
        _replace_numeral,
        text,
        flags=re.IGNORECASE,
    )
    return text


def _apply_punctuation_commands(text: str) -> str:
    """Replace spoken punctuation/formatting commands with their symbols."""
    for pattern, replacement, _ in PUNCTUATION_COMMANDS:
        if replacement in ("\n", "\n\n") or replacement.startswith("\n"):
            text = re.sub(
                r'\s*' + pattern + r'\s*',
                replacement,
                text,
                flags=re.IGNORECASE,
            )
        elif replacement == "(":
            text = re.sub(
                r'\s*' + pattern + r'\s*',
                " (",
                text,
                flags=re.IGNORECASE,
            )
        elif replacement == ")":
            text = re.sub(
                r'\s*' + pattern + r'\s*',
                ") ",
                text,
                flags=re.IGNORECASE,
            )
        elif replacement == '"':
            if "open" in pattern:
                text = re.sub(
                    r'\s*' + pattern + r'\s*',
                    ' "',
                    text,
                    flags=re.IGNORECASE,
                )
            else:
                text = re.sub(
                    r'\s*' + pattern + r'\s*',
                    '" ',
                    text,
                    flags=re.IGNORECASE,
                )
        elif replacement == "\t":
            text = re.sub(
                r'\s*' + pattern + r'\s*',
                replacement,
                text,
                flags=re.IGNORECASE,
            )
        else:
            text = re.sub(
                r'\s*' + pattern + r'\s*',
                replacement + " ",
                text,
                flags=re.IGNORECASE,
            )

    text = re.sub(r'([.?!,;:])\1+', r'\1', text)
    text = re.sub(r'  +', ' ', text)
    return text


def _apply_corrections(text: str, corrections: dict[str, str]) -> str:
    """Apply vocabulary corrections — whole-phrase, case-insensitive.

    Uses non-word lookarounds (rather than \\b) so corrections that begin or
    end with non-word characters still match. Prevents a correction like
    "ad" → "Alzheimer’s disease" from rewriting "add" or "advance".
    """
    for wrong, right in corrections.items():
        pattern = re.compile(r'(?<!\w)' + re.escape(wrong) + r'(?!\w)', re.IGNORECASE)
        text = pattern.sub(right, text)
    return text


# Words that legitimately end with a period without terminating a sentence.
# Capitalisation walker stays in "mid-sentence" mode when it sees one before
# a period. Single-letter words ("e", "i") are handled separately via length,
# so they don't need to be listed here.
_ABBREVIATIONS = frozenset({
    "etc", "vs", "cf", "ca",
    "dr", "mr", "mrs", "ms", "prof", "rev", "st",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "mg", "kg", "lb", "oz", "ml", "fig", "vol", "ed", "pp",
})


def _apply_capitalisation(text: str) -> str:
    """Capitalise the first letter and the letter after each sentence boundary.

    A period only counts as a sentence boundary if the preceding word is
    *not* a likely abbreviation. Heuristic: words ≤ 2 chars (covers "e", "i",
    "Dr", "Mr") and an explicit list of common abbreviations are treated as
    non-terminating. Without this, "e.g. lithium" becomes "E.G. Lithium" and
    "Dr. Smith said" becomes "Dr. Smith Said".
    """
    if not text:
        return text

    result = list(text)
    capitalise_next = True
    # Letters since the last word boundary — used to test for abbreviation
    # when we hit a '.'
    word: list[str] = []

    for i, ch in enumerate(result):
        if capitalise_next and ch.isalpha():
            result[i] = ch.upper()
            capitalise_next = False
            word = [ch.lower()]
        elif ch.isalpha():
            word.append(ch.lower())
        elif ch == ".":
            recent = "".join(word)
            # Treat ≤2-char words and known abbreviations as non-terminating.
            is_abbreviation = len(recent) <= 2 or recent in _ABBREVIATIONS
            capitalise_next = not is_abbreviation
            word = []
        elif ch in "?!\n":
            capitalise_next = True
            word = []
        elif ch.isspace():
            word = []
        # Other punctuation (commas, parens, etc.) preserves the running word
        # state — "e.g.," shouldn't reset word to empty mid-abbreviation.

    return "".join(result)

"""Tests for postprocess module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from whispr.postprocess import postprocess, _fix_whisper_spacing


# --- Whisper spacing fixes ---

class TestWhisperSpacing:
    def test_missing_space_after_period(self):
        assert _fix_whisper_spacing("word.Word") == "word. Word"

    def test_missing_space_after_question_mark(self):
        assert _fix_whisper_spacing("really?Yes") == "really? Yes"

    def test_missing_space_after_exclamation(self):
        assert _fix_whisper_spacing("wow!Great") == "wow! Great"

    def test_missing_space_after_comma(self):
        assert _fix_whisper_spacing("hello,world") == "hello, world"

    def test_preserves_numbers(self):
        # Should NOT insert space in "3.14" — only triggers on uppercase after period
        assert _fix_whisper_spacing("3.14") == "3.14"

    def test_preserves_existing_spaces(self):
        assert _fix_whisper_spacing("word. Next") == "word. Next"

    def test_preserves_lowercase_after_period(self):
        # e.g. abbreviations or domain terms — only uppercase triggers
        assert _fix_whisper_spacing("pTau217.positivity") == "pTau217.positivity"

    def test_mixed(self):
        result = _fix_whisper_spacing("The patient had pTau217.Positivity as described.There are issues.")
        assert result == "The patient had pTau217. Positivity as described. There are issues."


# --- Punctuation commands ---

class TestPunctuationCommands:
    def test_full_stop(self):
        result = postprocess("the patient is well full stop", enable_capitalisation=False)
        assert result == "the patient is well."

    def test_period(self):
        result = postprocess("end of sentence period", enable_capitalisation=False)
        assert result == "end of sentence."

    def test_comma(self):
        result = postprocess("first comma second", enable_capitalisation=False)
        assert result == "first, second"

    def test_question_mark(self):
        result = postprocess("is that right question mark", enable_capitalisation=False)
        assert result == "is that right?"

    def test_exclamation_mark(self):
        result = postprocess("wow exclamation mark", enable_capitalisation=False)
        assert result == "wow!"

    def test_colon(self):
        result = postprocess("findings colon normal", enable_capitalisation=False)
        assert result == "findings: normal"

    def test_semicolon(self):
        result = postprocess("first semicolon second", enable_capitalisation=False)
        assert result == "first; second"

    def test_new_line(self):
        result = postprocess("line one new line line two", enable_capitalisation=False)
        assert result == "line one\nline two"

    def test_new_paragraph(self):
        result = postprocess("para one new paragraph para two", enable_capitalisation=False)
        assert result == "para one\n\npara two"

    def test_brackets(self):
        result = postprocess("see open bracket figure one close bracket", enable_capitalisation=False)
        assert result == "see (figure one)"

    def test_hyphen(self):
        result = postprocess("well hyphen known", enable_capitalisation=False)
        assert result == "well- known"

    def test_quotes(self):
        result = postprocess("he said open quote hello close quote", enable_capitalisation=False)
        assert result == 'he said "hello"'

    def test_multiple_commands(self):
        result = postprocess(
            "the patient is well full stop no issues comma all clear full stop",
            enable_capitalisation=False,
        )
        assert result == "the patient is well. no issues, all clear."


# --- Editing commands ---

class TestEditingCommands:
    def test_scratch_that(self):
        assert postprocess("scratch that") == "ACTION:SCRATCH_THAT"

    def test_scratch_that_with_period(self):
        assert postprocess("Scratch that.") == "ACTION:SCRATCH_THAT"

    def test_scratch_word(self):
        assert postprocess("scratch word") == "ACTION:SCRATCH_WORD"

    def test_scratch_not_triggered_in_sentence(self):
        result = postprocess("I want to scratch that idea", enable_capitalisation=False)
        assert "SCRATCH" not in result

    # --- Filler / politeness tolerance (#7) ---

    def test_action_match_strips_trailing_please(self):
        assert postprocess("scratch that please") == "ACTION:SCRATCH_THAT"

    def test_action_match_strips_leading_uh(self):
        assert postprocess("uh scratch that") == "ACTION:SCRATCH_THAT"

    def test_action_match_strips_leading_comma_uh(self):
        # Whisper sometimes emits a comma after a discourse marker.
        assert postprocess("uh, scratch that") == "ACTION:SCRATCH_THAT"

    def test_action_match_strips_thank_you(self):
        assert postprocess("undo thank you") == "ACTION:UNDO"

    def test_action_filler_doesnt_swallow_real_words(self):
        # "scratch that down" must NOT match the scratch-that action —
        # filler stripping should only remove the surrounding fillers, not
        # words in the middle that change the semantics.
        result = postprocess("scratch that down please", enable_capitalisation=False)
        assert "ACTION" not in result


# --- Vocabulary corrections ---

class TestVocabularyCorrections:
    CORRECTIONS = {
        "pee tau 217": "pTau217",
        "our bands": "RBANS",
        "fasecas": "Fazekas",
    }

    def test_correction_applied(self):
        result = postprocess("the fasecas score is high", corrections=self.CORRECTIONS, enable_capitalisation=False)
        assert result == "the Fazekas score is high"

    def test_correction_case_insensitive(self):
        result = postprocess("Our Bands total", corrections=self.CORRECTIONS, enable_capitalisation=False)
        assert result == "RBANS total"

    def test_correction_preserves_case_of_replacement(self):
        result = postprocess("pee tau 217 was positive", corrections=self.CORRECTIONS, enable_capitalisation=False)
        assert result == "pTau217 was positive"

    def test_short_correction_does_not_match_inside_word(self):
        # "ad" should not rewrite "add", "advance", "advised", "shall".
        corrections = {"ad": "Alzheimer's disease"}
        result = postprocess(
            "add the advance plan and advise the patient",
            corrections=corrections,
            enable_capitalisation=False,
        )
        assert result == "add the advance plan and advise the patient"

    def test_short_correction_matches_whole_word(self):
        corrections = {"ad": "Alzheimer's disease"}
        result = postprocess(
            "patient has ad and family history.",
            corrections=corrections,
            enable_capitalisation=False,
        )
        assert "Alzheimer's disease" in result
        assert "add" not in result.lower().split()  # no spurious matches

    def test_correction_matches_at_string_boundaries(self):
        # Word-boundary fix must still match at start/end of string and
        # adjacent to punctuation.
        corrections = {"ad": "Alzheimer's disease"}
        assert postprocess("ad confirmed", corrections=corrections, enable_capitalisation=False) \
            == "Alzheimer's disease confirmed"
        assert postprocess("confirmed ad", corrections=corrections, enable_capitalisation=False) \
            == "confirmed Alzheimer's disease"
        assert postprocess("(ad)", corrections=corrections, enable_capitalisation=False) \
            == "(Alzheimer's disease)"


# --- Vocabulary expansions ---

class TestVocabularyExpansions:
    EXPANSIONS = {
        "standard intro": "Thank you for referring this patient to the Brain Health Clinic.",
    }

    def test_expansion(self):
        result = postprocess("standard intro", expansions=self.EXPANSIONS, enable_capitalisation=False)
        assert result == "Thank you for referring this patient to the Brain Health Clinic."

    def test_expansion_case_insensitive(self):
        result = postprocess("Standard Intro", expansions=self.EXPANSIONS, enable_capitalisation=False)
        assert result == "Thank you for referring this patient to the Brain Health Clinic."

    def test_short_expansion_does_not_match_inside_word(self):
        # Shorthand 'dr' should not rewrite 'drop' / 'drum' / 'address'.
        expansions = {"dr": "Doctor"}
        result = postprocess(
            "drop the drum and address the issue",
            expansions=expansions,
            enable_capitalisation=False,
        )
        assert result == "drop the drum and address the issue"

    def test_short_expansion_matches_whole_word(self):
        expansions = {"dr": "Doctor"}
        result = postprocess(
            "dr smith reviewed dr jones",
            expansions=expansions,
            enable_capitalisation=False,
        )
        assert result == "Doctor smith reviewed Doctor jones"


# --- Capitalisation ---

class TestCapitalisation:
    def test_capitalise_first_char(self):
        assert postprocess("hello world") == "Hello world"

    def test_capitalise_after_period(self):
        result = postprocess("first full stop second")
        assert result == "First. Second"

    def test_capitalise_after_question_mark(self):
        result = postprocess("really question mark yes")
        assert result == "Really? Yes"

    def test_capitalise_after_newline(self):
        result = postprocess("line one new line line two")
        assert result == "Line one\nLine two"

    def test_preserves_existing_caps(self):
        result = postprocess("the RBANS score", corrections={"": ""})
        assert "RBANS" in result

    # --- Abbreviation handling (#5) ---

    def test_eg_does_not_get_uppercased(self):
        # "e.g. lithium" must stay lowercase after the periods (the dots in
        # the abbreviation are not sentence boundaries).
        result = postprocess("the patient was started on e.g. lithium")
        assert "E.G." not in result
        assert "e.g." in result

    def test_ie_does_not_get_uppercased(self):
        result = postprocess("comorbidities i.e. diabetes and hypertension")
        assert "I.E." not in result
        assert "i.e." in result

    def test_title_dr_does_not_terminate_sentence(self):
        # "Dr." should not cause the next word to be uppercased mid-sentence.
        # First-letter capitalisation still applies to the leading "the".
        result = postprocess("the patient saw dr. smith yesterday")
        # The first char is capitalised; "smith" after "dr." is NOT.
        assert result.startswith("The patient saw")
        assert "dr. smith" in result.lower()
        # No spurious mid-sentence capital after "dr."
        assert "Dr. Smith" not in result or result == "The patient saw Dr. Smith yesterday"
        # The specific bug: "dr. " must not trigger capitalise_next
        assert "dr. Smith yesterday" not in result

    def test_sentence_boundary_still_capitalises(self):
        # Normal sentence boundaries (period after a regular-length word)
        # must still cap the next word.
        result = postprocess("the patient is well full stop the family is supportive")
        assert "well. The family" in result

    def test_decimal_number_does_not_trigger(self):
        # "3.14" — period followed by a digit, not a letter. capitalise_next
        # cares about the next *alpha* char so this is mostly safe, but
        # verify no spurious uppercasing on a following word.
        result = postprocess("the value was 3.14 mm")
        assert "3.14" in result
        # "mm" after "3.14 " — only triggered if "14" counted as a word ≤ 2
        # chars. With digits filtered out of `word`, "14" doesn't count.
        # Either way the next alpha "m" shouldn't be capitalised mid-sentence.
        assert "3.14 mm" in result


# --- Full pipeline ---

class TestFullPipeline:
    CORRECTIONS = {
        "pee tau 217": "pTau217",
        "our bands": "RBANS",
    }
    EXPANSIONS = {
        "standard intro": "Thank you for referring this patient to the Brain Health Clinic.",
    }

    def test_full_clinical_sentence(self):
        raw = "the patient had pee tau 217 positivity full stop our bands was normal full stop"
        result = postprocess(raw, corrections=self.CORRECTIONS, expansions=self.EXPANSIONS)
        assert result == "The patient had pTau217 positivity. RBANS was normal."

    def test_expansion_in_context(self):
        raw = "standard intro new paragraph the patient is well full stop"
        result = postprocess(raw, corrections=self.CORRECTIONS, expansions=self.EXPANSIONS)
        assert result == "Thank you for referring this patient to the Brain Health Clinic.\n\nThe patient is well."

    def test_whisper_spacing_fix_in_pipeline(self):
        raw = "word.Another word.More text"
        result = postprocess(raw)
        assert result == "Word. Another word. More text"

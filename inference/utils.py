import re
import json
import os
from collections import Counter
from typing import Dict, Any, List
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class RepeatEvent:
    type: str                # "char" / "word" / "phrase"
    start: int               # start character index of the repeated block
    end: int                 # end character index of the repeated block
    repeat_times: int
    content: str             # the repeated pattern (character / word / phrase)
    extra: Dict[str, Any]


def _normalize(text: str) -> str:
    """simple text normalization for better repeat detection, can be extended as needed."""
    cleaned = re.sub(r"\[.*?\]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _detect_char_repeats(cleaned: str, min_char_repeat: int) -> List[RepeatEvent]:
    """detect continuous repeated characters (any non-whitespace characters)."""
    if min_char_repeat <= 1:
        return []
    pattern_char = re.compile(r"(\S)\1{" + str(min_char_repeat - 1) + r",}")
    events = []
    for m in pattern_char.finditer(cleaned):
        char = m.group(1)
        count = len(m.group(0))
        events.append(
            RepeatEvent(
                type="char",
                start=m.start(),
                end=m.end(),
                repeat_times=count,
                content=char,
                extra={},
            )
        )
    return events


def _tokenize(cleaned: str) -> Tuple[List[str], List[int]]:
    """
    simple English tokenization: split by non-alphanumeric characters.
    return tokens and their starting character indices in cleaned (for position tracking).
    """
    tokens: List[str] = []
    starts: List[int] = []
    for m in re.finditer(r"\w+", cleaned):
        tokens.append(m.group(0))
        starts.append(m.start())
    return tokens, starts


def _detect_word_repeats(
    cleaned: str,
    min_word_repeat: int,
) -> List[RepeatEvent]:
    """
    detect continuous repeated words (mainly for scenarios with spaces / English).
    for pure Chinese without spaces, it is recommended to rely on character-level / phrase-level detection.
    """
    if min_word_repeat <= 1:
        return []
    tokens, starts = _tokenize(cleaned)
    if not tokens:
        return []

    events: List[RepeatEvent] = []
    current_word = tokens[0]
    current_start_idx = 0      # token index
    current_count = 1

    for i in range(1, len(tokens)):
        if tokens[i] == current_word:
            current_count += 1
        else:
            if current_count >= min_word_repeat:
                start_char = starts[current_start_idx]
                end_token_idx = current_start_idx + current_count - 1
                end_char = starts[end_token_idx] + len(tokens[end_token_idx])
                events.append(
                    RepeatEvent(
                        type="word",
                        start=start_char,
                        end=end_char,
                        repeat_times=current_count,
                        content=current_word,
                        extra={"token_start_idx": current_start_idx},
                    )
                )
            current_word = tokens[i]
            current_start_idx = i
            current_count = 1

    # check the last run
    if current_count >= min_word_repeat:
        start_char = starts[current_start_idx]
        end_token_idx = current_start_idx + current_count - 1
        end_char = starts[end_token_idx] + len(tokens[end_token_idx])
        events.append(
            RepeatEvent(
                type="word",
                start=start_char,
                end=end_char,
                repeat_times=current_count,
                content=current_word,
                extra={"token_start_idx": current_start_idx},
            )
        )
    return events


def _detect_phrase_repeats(
    cleaned: str,
    min_phrase_repeat: int,
    phrase_min_len: int,
    phrase_max_len: int,
) -> List[RepeatEvent]:
    """
    detect continuous repeated phrases (only used in "with spaces" scenarios with English-style tokenization, to avoid noise from pure Chinese character segmentation).
    """
    if min_phrase_repeat <= 1:
        return []
    if " " not in cleaned:
        return []

    tokens, starts = _tokenize(cleaned)
    if not tokens:
        return []

    seen_phrases: set = set()
    events: List[RepeatEvent] = []

    for n in range(max(1, phrase_min_len), phrase_max_len + 1):
        if len(tokens) < n * min_phrase_repeat:
            continue
        i = 0
        while i <= len(tokens) - n:
            phrase = tuple(tokens[i: i + n])
            count = 1
            j = i + n
            while j + n <= len(tokens) and tokens[j: j + n] == list(phrase):
                count += 1
                j += n
            if count >= min_phrase_repeat:
                phrase_str = " ".join(phrase)
                if phrase_str not in seen_phrases:
                    seen_phrases.add(phrase_str)
                    start_char = starts[i]
                    end_token_idx = j - 1
                    end_char = starts[end_token_idx] + len(tokens[end_token_idx])
                    events.append(
                        RepeatEvent(
                            type="phrase",
                            start=start_char,
                            end=end_char,
                            repeat_times=count,
                            content=phrase_str,
                            extra={"token_start_idx": i, "n": n},
                        )
                    )
                i = j
            else:
                i += 1

    return events


def _compute_bigram_ratio(tokens: List[str]) -> float:
    """compute global bigram repeat rate"""
    if len(tokens) < 2:
        return 0.0
    bigrams = [tuple(tokens[i: i + 2]) for i in range(len(tokens) - 1)]
    counter = Counter(bigrams)
    total = len(bigrams)
    duplicated = sum(cnt - 1 for cnt in counter.values() if cnt > 1)
    return round(duplicated / total, 4)


def detect_and_fix_hallucination_repetition(
    text: str,
    min_char_repeat: int = 15,
    min_word_repeat: int = 10,
    min_phrase_repeat: int = 5,
    phrase_min_len: int = 2,
    phrase_max_len: int = 8,
    ngram_ratio_threshold: float = 0.99,
) -> Dict[str, Any]:
    """
    detect & fix "repeated hallucination" in large model output text.

    repair strategy:
      - for each triggered repeat event, only keep the first occurrence; subsequent repeated parts are marked for deletion.
      - other non-hallucinated content is retained.

    return a dict with:
      {
        "has_hallucination": bool,
        "original_text": str,
        "repaired_text": str,
        "global_ngram_ratio": float,
        "events": [...],   # each repeat event's detailed information (based on cleaned text)
      }
    """
    if not text or not text.strip():
        return {
            "has_hallucination": False,
            "original_text": text,
            "repaired_text": text,
            "global_ngram_ratio": 0.0,
            "events": [],
        }

    cleaned = _normalize(text)
    events: List[RepeatEvent] = []

    # 1) consecutive repeated chars
    events.extend(_detect_char_repeats(cleaned, min_char_repeat))

    # 2) consecutive repeated words & phrases
    han_count = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
    space_count = cleaned.count(" ")
    is_chinese_no_space = (han_count > len(cleaned) * 0.3) and (space_count < len(cleaned) * 0.1)

    tokens_for_ngram: List[str] = []
    if not is_chinese_no_space:
        tokens_for_ngram, _starts = _tokenize(cleaned)
        events.extend(_detect_word_repeats(cleaned, min_word_repeat))
        events.extend(
            _detect_phrase_repeats(
                cleaned,
                min_phrase_repeat=min_phrase_repeat,
                phrase_min_len=phrase_min_len,
                phrase_max_len=phrase_max_len,
            )
        )
    else:
        # pure Chinese without spaces: only use character-level + ngram, to avoid excessive triggering of phrase repeats
        tokens_for_ngram = list(cleaned.replace(" ", ""))

    # 3) global n-gram repeat rate (only for marking, not for direct trimming)
    global_ngram_ratio = _compute_bigram_ratio(tokens_for_ngram)

    # 4) construct event information
    event_dicts: List[Dict[str, Any]] = []
    for ev in events:
        event_dicts.append(
            {
                "type": {
                    "char": "consecutive repeated characters",
                    "word": "consecutive repeated words",
                    "phrase": "consecutive repeated phrases",
                }[ev.type],
                "content": ev.content,
                "repeat_times": ev.repeat_times,
                "position": (ev.start, ev.end),
                "extra": ev.extra,
            }
        )

    # 5) only judge based on specific events whether there is hallucination (n-gram is only for reference)
    has_by_detail = any(
        (ev.type == "char" and ev.repeat_times >= min_char_repeat)
        or (ev.type == "word" and ev.repeat_times >= min_word_repeat)
        or (ev.type == "phrase" and ev.repeat_times >= min_phrase_repeat)
        for ev in events
    )
    has_hallucination = has_by_detail or (global_ngram_ratio >= ngram_ratio_threshold)

    # 6) construct character-level retention mask: default all retained
    keep = [True] * len(cleaned)

    # 7) for each "triggered threshold" event, only keep the first repeated unit, delete the rest
    for ev in events:
        if ev.type == "char" and ev.repeat_times >= min_char_repeat:
            # repeated block as cleaned[ev.start:ev.end] = content * repeat_times
            unit_len = len(ev.content)  # for char, this is 1
            first_end = ev.start + unit_len
            for i in range(first_end, ev.end):
                keep[i] = False

        elif ev.type in ("word", "phrase"):
            threshold = min_word_repeat if ev.type == "word" else min_phrase_repeat
            if ev.repeat_times < threshold:
                continue
            # block like "X X X X...", we keep the first X，delete the rest
            block = cleaned[ev.start:ev.end]
            pos0 = block.find(ev.content)
            if pos0 == -1:
                # not found, be conservative and keep everything
                continue
            pos1 = pos0 + len(ev.content)
            first_keep_start = ev.start + pos0
            first_keep_end = ev.start + pos1
            # delete first_keep_end ~ ev.end
            for i in range(first_keep_end, ev.end):
                keep[i] = False

    # 8) reconstruct cleaned version of text
    repaired_cleaned = "".join(ch for i, ch in enumerate(cleaned) if i < len(keep) and keep[i])

    return {
        "has_hallucination": has_hallucination,
        "original_text": text,
        "repaired_text": repaired_cleaned,
        "global_ngram_ratio": global_ngram_ratio,
        "events": event_dicts,
    }


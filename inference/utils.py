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
    start: int               # 在 cleaned 文本中的起始字符位置
    end: int                 # 结束字符位置（不含）
    repeat_times: int
    content: str             # 触发的模式（字符 / 词 / 短语）
    extra: Dict[str, Any]


def _normalize(text: str) -> str:
    """简单清洗：去掉 [xx] 形式（如时间戳）、压缩空白。"""
    cleaned = re.sub(r"\[.*?\]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _detect_char_repeats(cleaned: str, min_char_repeat: int) -> List[RepeatEvent]:
    """连续重复单字（任意非空白字符）。"""
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
    简单英文 token：按非字母数字下划线拆分。
    返回 tokens 及每个 token 在 cleaned 中的起始字符索引（便于回溯位置）。
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
    连续重复单词（主要针对含空格 / 英文等场景）。
    对纯中文无空格场景，建议依赖字符级 / 短语级。
    """
    if min_word_repeat <= 1:
        return []
    tokens, starts = _tokenize(cleaned)
    if not tokens:
        return []

    events: List[RepeatEvent] = []
    current_word = tokens[0]
    current_start_idx = 0      # token 索引
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

    # 收尾
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
    连续重复短语（只在“有空格”的情况使用英文式分词，避免纯中文拆字噪声）。
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
    """计算全局 bigram 重复率。"""
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
    检测 & 修复大模型输出文本中的“重复型幻觉”。

    修复策略：
      - 对每个触发的重复事件，只保留第一次出现；后续重复部分标记为删除。
      - 其它非幻觉内容全部保留。

    返回:
      {
        "has_hallucination": bool,
        "original_text": str,
        "repaired_text": str,
        "global_ngram_ratio": float,
        "events": [...],   # 每个重复事件的详细信息（基于 cleaned 文本）
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

    # 1) 连续重复字
    events.extend(_detect_char_repeats(cleaned, min_char_repeat))

    # 2) 连续重复词 & 短语
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
        # 纯中文无空格：只用字符级 + ngram，避免过度触发短语重复
        tokens_for_ngram = list(cleaned.replace(" ", ""))

    # 3) 全局 n-gram 重复率（只用于标记，不直接裁剪）
    global_ngram_ratio = _compute_bigram_ratio(tokens_for_ngram)

    # 4) 构造事件信息
    event_dicts: List[Dict[str, Any]] = []
    for ev in events:
        event_dicts.append(
            {
                "type": {
                    "char": "连续重复字",
                    "word": "连续重复词",
                    "phrase": "连续重复短语",
                }[ev.type],
                "content": ev.content,
                "repeat_times": ev.repeat_times,
                "position": (ev.start, ev.end),
                "extra": ev.extra,
            }
        )

    # 5) 仅基于具体事件判断是否有幻觉（n-gram 只作为参考）
    has_by_detail = any(
        (ev.type == "char" and ev.repeat_times >= min_char_repeat)
        or (ev.type == "word" and ev.repeat_times >= min_word_repeat)
        or (ev.type == "phrase" and ev.repeat_times >= min_phrase_repeat)
        for ev in events
    )
    has_hallucination = has_by_detail or (global_ngram_ratio >= ngram_ratio_threshold)

    # 6) 构造字符级保留掩码：默认全部保留
    keep = [True] * len(cleaned)

    # 7) 对每个“触发阈值”的事件，只保留第一次重复单元，其余删掉
    for ev in events:
        if ev.type == "char" and ev.repeat_times >= min_char_repeat:
            # 重复块为 cleaned[ev.start:ev.end] = content * repeat_times
            unit_len = len(ev.content)  # 对于 char，这里是 1
            first_end = ev.start + unit_len
            for i in range(first_end, ev.end):
                keep[i] = False

        elif ev.type in ("word", "phrase"):
            threshold = min_word_repeat if ev.type == "word" else min_phrase_repeat
            if ev.repeat_times < threshold:
                continue
            # block 中形如 "X X X X..."，我们保留第一个 X，其余删除
            block = cleaned[ev.start:ev.end]
            pos0 = block.find(ev.content)
            if pos0 == -1:
                # 找不到就保守地不动
                continue
            pos1 = pos0 + len(ev.content)
            first_keep_start = ev.start + pos0
            first_keep_end = ev.start + pos1
            # 删除 first_keep_end ~ ev.end
            for i in range(first_keep_end, ev.end):
                keep[i] = False

    # 8) 重建 cleaned 版文本
    repaired_cleaned = "".join(ch for i, ch in enumerate(cleaned) if i < len(keep) and keep[i])

    return {
        "has_hallucination": has_hallucination,
        "original_text": text,
        "repaired_text": repaired_cleaned,
        "global_ngram_ratio": global_ngram_ratio,
        "events": event_dicts,
    }


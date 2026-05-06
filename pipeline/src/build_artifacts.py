from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import orjson
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer


USER_LINE_RE = re.compile(r"^(User\s+\d+):\s*(.*)$")
USER_SPLIT_RE = re.compile(r"(User\s+\d+):")
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")
ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]"
)
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "so",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "about",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "she",
    "they",
    "them",
    "their",
    "not",
    "no",
    "yes",
    "do",
    "did",
    "does",
    "have",
    "has",
    "had",
    "just",
    "really",
    "very",
    "pretty",
    "too",
    "also",
    "well",
    "okay",
    "ok",
    "yeah",
    "hi",
    "hello",
    "thanks",
    "thank",
}
ACTIVITY_WORDS = {
    "reading",
    "cook",
    "cooking",
    "bake",
    "baking",
    "hike",
    "hiking",
    "run",
    "running",
    "yoga",
    "photography",
    "garden",
    "gardening",
    "dance",
    "dancing",
    "travel",
    "traveling",
    "swim",
    "swimming",
    "music",
    "sing",
    "singing",
    "play",
    "playing",
    "paint",
    "painting",
    "write",
    "writing",
    "read",
    "reading",
    "exercise",
    "exercising",
    "bike",
    "biking",
    "climb",
    "climbing",
    "movies",
    "movie",
    "books",
    "book",
    "gaming",
    "games",
}
FACT_ROLE_HINTS = {
    "student",
    "engineer",
    "developer",
    "designer",
    "nurse",
    "teacher",
    "librarian",
    "firefighter",
    "doctor",
    "lawyer",
    "chef",
    "barista",
    "manager",
    "mom",
    "mother",
    "dad",
    "father",
    "parent",
}
FACT_NOISE_TERMS = {"doing", "going", "feeling", "fine", "good", "great", "okay", "alright", "well"}


@dataclass
class TopicSegment:
    topic_id: str
    start_idx: int
    end_idx: int


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def tokenize_filtered(text: str) -> List[str]:
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 1]


def extract_entities(text: str) -> List[str]:
    return [e for e in ENTITY_RE.findall(text) if e.lower() not in {"user", "hey", "hello"}]


def split_conversation(conversation: str) -> List[dict]:
    if not conversation:
        return []
    matches = list(USER_SPLIT_RE.finditer(conversation))
    if not matches:
        return []
    segments: List[dict] = []
    for idx, match in enumerate(matches):
        speaker = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(conversation)
        text = conversation[start:end]
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            segments.append({"speaker": speaker, "text": text})
    return segments


def parse_csv_rows(csv_path: Path) -> List[dict]:
    messages: List[dict] = []
    global_id = 1
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    # File has no true header; first row is a conversation.
    for row_idx, row in enumerate(rows):
        if not row:
            continue
        conversation = ",".join([cell for cell in row if cell]).strip()
        row_messages = split_conversation(conversation)
        if not row_messages:
            lines = [ln.strip() for ln in conversation.splitlines() if ln.strip()]
            current_msg = None
            for line in lines:
                m = USER_LINE_RE.match(line)
                if m:
                    if current_msg:
                        row_messages.append(current_msg)
                    current_msg = {"speaker": m.group(1), "text": m.group(2).strip()}
                elif current_msg:
                    current_msg["text"] = f"{current_msg['text']} {line}".strip()
            if current_msg:
                row_messages.append(current_msg)

        for msg_idx, msg in enumerate(row_messages):
            text = re.sub(r"\s+", " ", msg["text"]).strip()
            if not text:
                continue
            messages.append(
                {
                    "global_msg_id": global_id,
                    "row_index": row_idx,
                    "msg_index_in_row": msg_idx,
                    "order_key": f"{row_idx:06d}-{msg_idx:06d}",
                    "speaker": msg["speaker"],
                    "text": text,
                }
            )
            global_id += 1
    return messages


def iter_row_spans(messages: List[dict]) -> List[Tuple[int, int]]:
    if not messages:
        return []
    spans: List[Tuple[int, int]] = []
    start = 0
    current_row = messages[0]["row_index"]
    for i, msg in enumerate(messages):
        if msg["row_index"] != current_row:
            spans.append((start, i - 1))
            start = i
            current_row = msg["row_index"]
    spans.append((start, len(messages) - 1))
    return spans


def segment_topics_in_span(
    messages: List[dict], span_start: int, span_end: int, min_len: int, max_len: int
) -> List[TopicSegment]:
    span = messages[span_start : span_end + 1]
    if not span:
        return []
    if len(span) <= min_len:
        return [TopicSegment(topic_id="T0000", start_idx=span_start, end_idx=span_end)]

    texts = [m["text"] for m in span]
    vectorizer = HashingVectorizer(
        n_features=2048, alternate_sign=False, norm="l2", lowercase=True, stop_words="english"
    )
    mat = vectorizer.transform(texts)
    dense = mat.toarray().astype(np.float32)
    dense_norm = np.linalg.norm(dense, axis=1) + 1e-8
    dense = dense / dense_norm[:, None]

    segments: List[TopicSegment] = []
    start = 0
    running_centroid = dense[0].copy()
    running_count = 1
    running_entities = set(extract_entities(texts[0]))
    pending_boundary = False

    for i in range(1, len(span)):
        vec = dense[i]
        sim = float(np.dot(vec, running_centroid) / (np.linalg.norm(running_centroid) + 1e-8))
        semantic_drift = 1.0 - max(min(sim, 1.0), -1.0)

        curr_entities = set(extract_entities(span[i]["text"]))
        if running_entities:
            overlap = len(curr_entities & running_entities) / max(len(running_entities), 1)
            entity_shift = 1.0 - overlap
        else:
            entity_shift = 0.0

        ack_like = bool(
            re.search(r"\b(yes|yeah|no|ok|okay|cool|nice|great|sure|thanks|thank you)\b", span[i]["text"], re.I)
        )
        continuity = 0.12 if ack_like else 0.0

        boundary_score = 0.70 * semantic_drift + 0.20 * entity_shift - continuity
        seg_len = i - start
        can_split = seg_len >= min_len
        must_split = seg_len >= max_len
        candidate = boundary_score > 0.78 and can_split

        should_split = must_split or (candidate and pending_boundary)
        pending_boundary = candidate and not must_split

        if should_split:
            segments.append(
                TopicSegment(topic_id="T0000", start_idx=span_start + start, end_idx=span_start + i - 1)
            )
            start = i
            running_centroid = vec.copy()
            running_count = 1
            running_entities = curr_entities
            pending_boundary = False
            continue

        running_count += 1
        running_centroid = ((running_centroid * (running_count - 1)) + vec) / running_count
        if curr_entities:
            running_entities |= curr_entities

    segments.append(TopicSegment(topic_id="T0000", start_idx=span_start + start, end_idx=span_start + len(span) - 1))
    return merge_tiny_segments(segments, min_len=min_len)


def segment_topics(messages: List[dict], min_len: int = 20, max_len: int = 240) -> List[TopicSegment]:
    segments: List[TopicSegment] = []
    for span_start, span_end in iter_row_spans(messages):
        segments.extend(segment_topics_in_span(messages, span_start, span_end, min_len=min_len, max_len=max_len))

    fixed: List[TopicSegment] = []
    for idx, seg in enumerate(segments, start=1):
        fixed.append(TopicSegment(topic_id=f"T{idx:04d}", start_idx=seg.start_idx, end_idx=seg.end_idx))
    return fixed


def merge_tiny_segments(segments: List[TopicSegment], min_len: int = 8) -> List[TopicSegment]:
    if len(segments) <= 1:
        return segments
    merged: List[TopicSegment] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        seg_len = seg.end_idx - seg.start_idx + 1
        if seg_len >= min_len or i == len(segments) - 1:
            merged.append(seg)
            i += 1
            continue
        nxt = segments[i + 1]
        merged.append(TopicSegment(topic_id=seg.topic_id, start_idx=seg.start_idx, end_idx=nxt.end_idx))
        i += 2

    # Re-number IDs and ensure contiguous order.
    fixed: List[TopicSegment] = []
    for idx, seg in enumerate(merged, start=1):
        fixed.append(TopicSegment(topic_id=f"T{idx:04d}", start_idx=seg.start_idx, end_idx=seg.end_idx))
    return fixed


def truncate_text(text: str, limit: int = 140) -> str:
    if len(text) <= limit:
        return text
    trimmed = text[:limit].rsplit(" ", 1)[0].strip()
    return f"{trimmed}..."


def summarize_texts(texts: List[str], max_keywords: int = 8, max_sentences: int = 3) -> Tuple[str, List[str]]:
    if not texts:
        return "", []
    clean = [t.strip() for t in texts if t.strip()]
    if len(clean) == 1:
        return clean[0], tokenize(clean[0])[:max_keywords]

    vec = TfidfVectorizer(stop_words="english", max_features=1600)
    tfidf = vec.fit_transform(clean)
    terms = np.array(vec.get_feature_names_out())
    scores = np.asarray(tfidf.sum(axis=0)).ravel()
    top_idx = scores.argsort()[::-1][:max_keywords]
    keywords = [terms[i] for i in top_idx if scores[i] > 0]

    sentence_scores = np.asarray(tfidf.sum(axis=1)).ravel()
    ranked = sentence_scores.argsort()[::-1]
    selected: List[str] = []
    for idx in ranked:
        candidate = clean[idx]
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= max_sentences:
            break
    summary = "Key points: " + "; ".join(truncate_text(s, 140) for s in selected)
    return summary, keywords


def build_topic_checkpoints(messages: List[dict], segments: List[TopicSegment]) -> List[dict]:
    topics: List[dict] = []
    for seg in segments:
        span = messages[seg.start_idx : seg.end_idx + 1]
        texts = [m["text"] for m in span]
        summary, keywords = summarize_texts(texts, max_keywords=8)
        evidence_positions = [0, len(span) // 2, len(span) - 1]
        evidence_ids = sorted({span[p]["global_msg_id"] for p in evidence_positions if 0 <= p < len(span)})
        topics.append(
            {
                "topic_id": seg.topic_id,
                "start_msg_id": span[0]["global_msg_id"],
                "end_msg_id": span[-1]["global_msg_id"],
                "message_count": len(span),
                "keywords": keywords,
                "topic_summary": summary,
                "evidence_msg_ids": evidence_ids,
            }
        )
    return topics


def build_100_checkpoints(messages: List[dict], window_size: int = 100) -> List[dict]:
    checkpoints: List[dict] = []
    total = len(messages)
    count = int(math.ceil(total / window_size))
    for i in range(count):
        start = i * window_size
        end = min((i + 1) * window_size, total)
        span = messages[start:end]
        texts = [m["text"] for m in span]
        summary, keywords = summarize_texts(texts, max_keywords=6)
        checkpoints.append(
            {
                "checkpoint_id": f"C{i+1:04d}",
                "start_msg_id": span[0]["global_msg_id"],
                "end_msg_id": span[-1]["global_msg_id"],
                "message_count": len(span),
                "summary": summary,
                "highlights": keywords[:4],
            }
        )
    return checkpoints


def clean_candidate_phrase(value: str) -> str:
    value = re.sub(r"\b(really|very|pretty|kind of|sort of)\b", "", value, flags=re.I)
    value = re.sub(r"\b(too|also|as well)\b$", "", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def is_valid_fact(value: str) -> bool:
    tokens = tokenize(value)
    if not tokens:
        return False
    if len(tokens) > 8:
        return False
    if any(term in FACT_NOISE_TERMS for term in tokens) and not any(term in FACT_ROLE_HINTS for term in tokens):
        return False
    return True


def is_valid_habit(value: str) -> bool:
    tokens = tokenize(value)
    if not tokens:
        return False
    if len(tokens) > 8:
        return False
    if any(term in {"job", "work", "family", "friends"} for term in tokens):
        return False
    if value in {"my job", "my family", "my friends"}:
        return False
    if value.startswith("to "):
        tokens = tokens[1:] if tokens and tokens[0] == "to" else tokens
        if not tokens:
            return False
        if tokens[0] in {"do", "be", "have", "go"} and not any(t in ACTIVITY_WORDS for t in tokens):
            return False
        if any(t in ACTIVITY_WORDS for t in tokens) or any(t.endswith("ing") for t in tokens):
            return True
        return False
    if any(t.endswith("ing") for t in tokens):
        return True
    if any(t in ACTIVITY_WORDS for t in tokens):
        return True
    return False


def candidate_persona_entries(messages: List[dict], target_speaker: str | None = None) -> Dict[str, List[dict]]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for m in messages:
        if target_speaker and m["speaker"] != target_speaker:
            continue
        text = m["text"]
        msg_id = m["global_msg_id"]
        lower = text.lower()

        explicit_fact_patterns = [
            (r"\bi am (a|an)\s+([a-z][a-z\s-]{1,40})", 0.9),
            (r"\bi'?m (a|an)\s+([a-z][a-z\s-]{1,40})", 0.9),
            (r"\bi study\s+([a-z][a-z\s-]{1,40})", 0.88),
            (r"\bi work (as|in|at|for)\s+([a-z][a-z\s-]{1,40})", 0.88),
            (r"\bi live in\s+([a-z][a-z\s-]{1,40})", 0.88),
            (r"\bi am\s+(\d{1,3})\s*(?:years old|yo)?\b", 0.9),
            (r"\bmy name is\s+([a-z][a-z\s'-]{1,40})", 0.92),
        ]
        for pat, confidence in explicit_fact_patterns:
            mm = re.search(pat, lower)
            if mm:
                raw = mm.group(mm.lastindex).strip()
                value = clean_candidate_phrase(raw)
                if "\\d{1,3}" in pat:
                    value = f"age {value}"
                if 2 <= len(value) <= 45 and is_valid_fact(value):
                    buckets["personal_facts"].append(
                        {"value": value, "evidence_msg_ids": [msg_id], "confidence": confidence, "kind": "fact"}
                    )

        hobby_patterns = [
            r"\bi (like|love|enjoy)\s+([a-z][a-z\s-]{1,45})",
            r"\bi (usually|often|tend to)\s+([a-z][a-z\s-]{1,45})",
        ]
        for pat in hobby_patterns:
            mm = re.search(pat, lower)
            if mm:
                raw = mm.group(mm.lastindex).strip()
                value = clean_candidate_phrase(raw)
                if 2 <= len(value) <= 45 and is_valid_habit(value):
                    buckets["habits"].append(
                        {"value": value, "evidence_msg_ids": [msg_id], "confidence": 0.75, "kind": "habit"}
                    )

        if text.count("!") >= 2:
            buckets["personality_traits"].append(
                {"value": "enthusiastic", "evidence_msg_ids": [msg_id], "confidence": 0.64, "kind": "trait"}
            )
        if re.search(r"\b(thanks|glad|happy to help)\b", lower):
            buckets["personality_traits"].append(
                {"value": "supportive", "evidence_msg_ids": [msg_id], "confidence": 0.66, "kind": "trait"}
            )
    return buckets


def consolidate_persona(candidates: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for section, items in candidates.items():
        grouped: Dict[str, dict] = {}
        for item in items:
            key = re.sub(r"\s+", " ", item["value"]).strip().lower()
            if len(key) < 3:
                continue
            if key not in grouped:
                grouped[key] = {"value": key, "evidence_msg_ids": [], "confidence_sum": 0.0, "count": 0}
            grouped[key]["evidence_msg_ids"].extend(item["evidence_msg_ids"])
            grouped[key]["confidence_sum"] += item["confidence"]
            grouped[key]["count"] += 1
        final_items: List[dict] = []
        for g in grouped.values():
            count = g["count"]
            avg_conf = g["confidence_sum"] / max(count, 1)
            # Candidate -> verify rule:
            # keep if repeated, or explicit high-confidence statement.
            if count >= 2 or avg_conf >= 0.85:
                if section == "habits":
                    key_name = "habit"
                elif section == "personal_facts":
                    key_name = "fact"
                else:
                    key_name = "trait"
                final_items.append(
                    {
                        key_name: g["value"],
                        "confidence": round(min(0.98, avg_conf + 0.06 * min(count, 4)), 3),
                        "evidence_msg_ids": sorted(set(g["evidence_msg_ids"]))[:10],
                    }
                )
        out[section] = sorted(final_items, key=lambda x: x["confidence"], reverse=True)[:30]
    return out


def build_communication_style(messages: List[dict], target_speaker: str | None = None) -> dict:
    if target_speaker:
        messages = [m for m in messages if m["speaker"] == target_speaker]
    if not messages:
        return {}
    lengths = np.array([len(m["text"]) for m in messages], dtype=np.float32)
    avg_len = float(np.mean(lengths))
    exclaim_rate = float(np.mean([1.0 if "!" in m["text"] else 0.0 for m in messages]))
    emoji_rate = float(np.mean([1.0 if EMOJI_RE.search(m["text"]) else 0.0 for m in messages]))
    question_rate = float(np.mean([1.0 if "?" in m["text"] else 0.0 for m in messages]))

    if avg_len < 45:
        length_band = "short"
    elif avg_len < 110:
        length_band = "short-medium"
    else:
        length_band = "long"

    tones = []
    if exclaim_rate > 0.25:
        tones.append("energetic")
    if question_rate > 0.20:
        tones.append("curious")
    if not tones:
        tones.append("neutral")

    evidence_ids = [m["global_msg_id"] for m in messages if ("!" in m["text"] or "?" in m["text"])][:20]
    return {
        "avg_length": length_band,
        "tone": tones,
        "emoji_usage": "high" if emoji_rate > 0.1 else "low",
        "evidence_msg_ids": evidence_ids,
    }


def build_persona(messages: List[dict], target_speaker: str | None = None) -> dict:
    candidates = candidate_persona_entries(messages, target_speaker=target_speaker)
    consolidated = consolidate_persona(candidates)
    return {
        "habits": consolidated.get("habits", []),
        "personal_facts": consolidated.get("personal_facts", []),
        "personality_traits": consolidated.get("personality_traits", []),
        "communication_style": build_communication_style(messages, target_speaker=target_speaker),
        "method": {
            "approach": "candidate-then-verify",
            "rules": [
                "Keep persona entries that repeat across messages or have high explicit confidence",
                "Each persona item must include evidence_msg_ids",
                f"Only messages from {target_speaker or 'all speakers'} are used for persona extraction",
            ],
        },
    }


def build_persona_bundle(messages: List[dict], default_speaker: str | None = None) -> dict:
    speakers = sorted({m["speaker"] for m in messages})
    resolved_default = default_speaker if default_speaker in speakers else (speakers[0] if speakers else None)
    personas = {speaker: build_persona(messages, target_speaker=speaker) for speaker in speakers}
    return {
        "default_speaker": resolved_default,
        "speakers": speakers,
        "persona_by_speaker": personas,
    }


def build_chunks(messages: List[dict], chunk_size: int = 10, overlap: int = 3) -> List[dict]:
    chunks: List[dict] = []
    step = max(1, chunk_size - overlap)
    for span_start, span_end in iter_row_spans(messages):
        row_msgs = messages[span_start : span_end + 1]
        for start in range(0, len(row_msgs), step):
            end = min(len(row_msgs), start + chunk_size)
            span = row_msgs[start:end]
            if not span:
                continue
            text = " ".join([f"{m['speaker']}: {m['text']}" for m in span])
            chunks.append(
                {
                    "chunk_id": f"CH{len(chunks)+1:06d}",
                    "start_msg_id": span[0]["global_msg_id"],
                    "end_msg_id": span[-1]["global_msg_id"],
                    "row_start": span[0]["row_index"],
                    "row_end": span[-1]["row_index"],
                    "text": text,
                }
            )
    return chunks


def map_chunks_to_topics(chunks: List[dict], topics: List[dict]) -> Dict[str, List[str]]:
    topic_map: Dict[str, List[str]] = defaultdict(list)
    for ch in chunks:
        cs, ce = ch["start_msg_id"], ch["end_msg_id"]
        for t in topics:
            if not (ce < t["start_msg_id"] or cs > t["end_msg_id"]):
                topic_map[t["topic_id"]].append(ch["chunk_id"])
    return topic_map


def build_sparse_tfidf_index(items: List[dict], text_key: str, id_key: str, top_k: int) -> dict:
    doc_ids: List[str] = [item[id_key] for item in items]
    tokenized: List[List[str]] = [tokenize_filtered(item[text_key]) for item in items]
    df = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    total_docs = max(len(tokenized), 1)
    idf = {term: math.log((total_docs + 1) / (count + 1)) + 1.0 for term, count in df.items()}

    postings: Dict[str, List[List[float]]] = defaultdict(list)
    doc_norms: List[float] = []
    for idx, tokens in enumerate(tokenized):
        tf = Counter(tokens)
        weights = {}
        for term, count in tf.items():
            if term not in idf:
                continue
            weights[term] = (1.0 + math.log(count)) * idf[term]
        if not weights:
            doc_norms.append(1.0)
            continue
        top_terms = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:top_k]
        norm = math.sqrt(sum(w * w for _, w in top_terms))
        doc_norms.append(round(max(norm, 1e-6), 6))
        for term, weight in top_terms:
            postings[term].append([idx, round(weight, 6)])

    idf_out = {term: round(idf[term], 6) for term in postings.keys()}
    return {"doc_ids": doc_ids, "doc_norms": doc_norms, "idf": idf_out, "postings": postings, "top_k": top_k}


def extract_facts(messages: List[dict]) -> List[dict]:
    facts: List[dict] = []
    patterns = [
        ("moving_to", re.compile(r"\b(?:i am|i'm|im)?\s*moving to ([a-zA-Z][a-zA-Z\\s,.-]{1,60})", re.I)),
        ("live_in", re.compile(r"\b(?:i live in|i'm in|im in) ([a-zA-Z][a-zA-Z\\s,.-]{1,60})", re.I)),
        ("work_as", re.compile(r"\b(?:i work (?:as|in|at|for)|i'm a|i am a)\s+([a-zA-Z][a-zA-Z\\s-]{1,40})", re.I)),
        ("study", re.compile(r"\b(?:i study|i'm studying|i am studying) ([a-zA-Z][a-zA-Z\\s-]{1,40})", re.I)),
    ]
    for msg in messages:
        text = msg["text"]
        for fact_type, pattern in patterns:
            for match in pattern.finditer(text):
                value = re.sub(r"\\s+", " ", match.group(1)).strip().strip(" ,.-")
                if not value or len(value) < 2 or len(value) > 60:
                    continue
                if fact_type in {"moving_to", "live_in"}:
                    if "city" in value.lower() or "town" in value.lower():
                        continue
                    if not any(part[:1].isupper() for part in re.split(r"[\\s,.-]+", value) if part):
                        continue
                if fact_type in {"work_as", "study"} and not is_valid_fact(value):
                    continue
                facts.append(
                    {
                        "fact_type": fact_type,
                        "speaker": msg["speaker"],
                        "value": value,
                        "msg_id": msg["global_msg_id"],
                        "row_index": msg["row_index"],
                    }
                )
    return facts


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")


def build_all(csv_path: Path, out_dir: Path, persona_speaker: str | None = None) -> dict:
    messages = parse_csv_rows(csv_path)
    topics = build_topic_checkpoints(messages, segment_topics(messages))
    checkpoints_100 = build_100_checkpoints(messages)
    persona = build_persona_bundle(messages, default_speaker=persona_speaker)
    chunks = build_chunks(messages)
    topic_to_chunks = map_chunks_to_topics(chunks, topics)
    facts = extract_facts(messages)
    topic_index = build_sparse_tfidf_index(topics, text_key="topic_summary", id_key="topic_id", top_k=12)
    chunk_index = build_sparse_tfidf_index(chunks, text_key="text", id_key="chunk_id", top_k=18)
    checkpoint_index = build_sparse_tfidf_index(
        checkpoints_100, text_key="summary", id_key="checkpoint_id", top_k=10
    )

    write_jsonl(Path("pipeline/output/messages.jsonl"), messages)
    write_json(out_dir / "topics.json", topics)
    write_json(out_dir / "checkpoints_100.json", checkpoints_100)
    write_json(out_dir / "persona.json", persona)
    write_json(out_dir / "chunks.json", chunks)
    write_json(out_dir / "facts.json", facts)
    write_json(out_dir / "indexes" / "topic_to_chunks.json", topic_to_chunks)
    write_json(out_dir / "indexes" / "topic_tfidf.json", topic_index)
    write_json(out_dir / "indexes" / "chunk_tfidf.json", chunk_index)
    write_json(out_dir / "indexes" / "checkpoint_tfidf.json", checkpoint_index)

    meta = {
        "total_messages": len(messages),
        "total_topics": len(topics),
        "total_checkpoints_100": len(checkpoints_100),
        "total_chunks": len(chunks),
        "total_facts": len(facts),
        "source_csv": str(csv_path),
        "persona_default_speaker": persona_speaker or "auto",
        "persona_speakers": persona.get("speakers", []),
        "retrieval_index": "sparse-tfidf-v1",
    }
    write_json(out_dir / "meta.json", meta)
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Persona-Aware RAG artifacts from CSV conversations.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("conversations.csv"),
        help="Path to conversations CSV (rows are chronological conversation snapshots).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("app/data"),
        help="Output directory for generated artifacts.",
    )
    parser.add_argument(
        "--persona-speaker",
        type=str,
        default="User 1",
        help="Speaker name to build persona for (e.g., 'User 1').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = build_all(args.csv, args.out, persona_speaker=args.persona_speaker)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()


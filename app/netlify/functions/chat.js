"use strict";

const fs = require("fs");
const path = require("path");

let cache = null;

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

const STOPWORDS = new Set([
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
]);

function tokenize(text) {
  return (text.toLowerCase().match(/[a-z][a-z0-9']+/g) || []).filter(
    (t) => t.length > 1 && !STOPWORDS.has(t)
  );
}

function scoreIndex(index, query, limit = 8) {
  const tokens = tokenize(query);
  const tf = new Map();
  for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);
  let qNorm = 0;
  const qWeights = new Map();
  for (const [term, count] of tf.entries()) {
    const idf = index.idf[term];
    if (!idf) continue;
    const weight = (1 + Math.log(count)) * idf;
    qWeights.set(term, weight);
    qNorm += weight * weight;
  }
  qNorm = Math.sqrt(qNorm) || 1;

  const scores = new Map();
  for (const [term, qWeight] of qWeights.entries()) {
    const postings = index.postings[term] || [];
    for (const [docIdx, docWeight] of postings) {
      scores.set(docIdx, (scores.get(docIdx) || 0) + qWeight * docWeight);
    }
  }

  const results = [];
  for (const [docIdx, score] of scores.entries()) {
    const norm = index.doc_norms[docIdx] || 1;
    results.push({ doc_id: index.doc_ids[docIdx], score: score / (norm * qNorm) });
  }
  results.sort((a, b) => b.score - a.score);
  return results.slice(0, limit);
}

function overlapBoost(text, queryTokens) {
  if (!text || !queryTokens.length) return 0;
  const lower = text.toLowerCase();
  let hits = 0;
  for (const t of queryTokens) {
    if (lower.includes(t)) hits += 1;
  }
  return hits / Math.max(queryTokens.length, 1);
}

function buildPatternBoosts(query, data) {
  const q = query.toLowerCase();
  const patterns = [];
  if ((q.includes("moving") && (q.includes("city") || q.includes("where"))) || q.includes("moving to")) {
    patterns.push({ regex: /\bmoving to ([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*)/g, weight: 2.0 });
  }
  if ((q.includes("live") && (q.includes("where") || q.includes("city"))) || q.includes("live in")) {
    patterns.push({ regex: /\blive in ([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*)/g, weight: 1.6 });
  }
  if ((q.includes("work") && (q.includes("job") || q.includes("living"))) || q.includes("work as")) {
    patterns.push({ regex: /\bwork (?:as|in|at|for) ([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*)/g, weight: 1.2 });
  }
  if (!patterns.length) return new Map();

  const boosts = new Map();
  for (const chunk of data.chunks) {
    let score = 0;
    for (const pattern of patterns) {
      if (pattern.regex.test(chunk.text)) {
        score += pattern.weight;
      }
      pattern.regex.lastIndex = 0;
    }
    if (score > 0) boosts.set(chunk.chunk_id, score);
  }
  return boosts;
}

function detectFactIntent(query) {
  const q = query.toLowerCase();
  if ((q.includes("moving") && (q.includes("where") || q.includes("city"))) || q.includes("moving to")) {
    return "moving_to";
  }
  if ((q.includes("live") && (q.includes("where") || q.includes("city"))) || q.includes("live in")) {
    return "live_in";
  }
  if ((q.includes("work") && (q.includes("job") || q.includes("living") || q.includes("do for work"))) || q.includes("work as")) {
    return "work_as";
  }
  if (q.includes("study") || q.includes("studying")) {
    return "study";
  }
  return null;
}

function findChunkByMsgId(chunks, msgId) {
  let lo = 0;
  let hi = chunks.length - 1;
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    const ch = chunks[mid];
    if (msgId < ch.start_msg_id) {
      hi = mid - 1;
    } else if (msgId > ch.end_msg_id) {
      lo = mid + 1;
    } else {
      return ch;
    }
  }
  return null;
}

function retrieveFacts(data, query, speakerHint) {
  const intent = detectFactIntent(query);
  if (!intent) return { facts: [], fact_chunks: [] };
  const candidates = data.facts.filter(
    (f) => f.fact_type === intent && (!speakerHint || f.speaker === speakerHint)
  );
  if (!candidates.length) return { facts: [], fact_chunks: [] };

  const grouped = new Map();
  for (const fact of candidates) {
    const key = fact.value.toLowerCase();
    if (!grouped.has(key)) {
      grouped.set(key, { value: fact.value, count: 0, min_msg: fact.msg_id, speaker: fact.speaker });
    }
    const entry = grouped.get(key);
    entry.count += 1;
    entry.min_msg = Math.min(entry.min_msg, fact.msg_id);
  }

  const ranked = Array.from(grouped.values()).sort((a, b) => {
    if (b.count !== a.count) return b.count - a.count;
    return a.min_msg - b.min_msg;
  });

  const top = ranked.slice(0, 3);
  const factChunks = [];
  for (const item of top) {
    const source = candidates.find((f) => f.value.toLowerCase() === item.value.toLowerCase());
    if (!source) continue;
    const chunk = findChunkByMsgId(data.chunks, source.msg_id);
    if (chunk) factChunks.push(chunk);
  }
  return { facts: top, fact_chunks: factChunks };
}

function loadData() {
  if (cache) return cache;
  const base = path.resolve(__dirname, "..", "..", "data");
  const topics = readJson(path.join(base, "topics.json"));
  const chunks = readJson(path.join(base, "chunks.json"));
  const checkpoints = readJson(path.join(base, "checkpoints_100.json"));
  const facts = readJson(path.join(base, "facts.json"));
  const persona = readJson(path.join(base, "persona.json"));
  const topicToChunks = readJson(path.join(base, "indexes", "topic_to_chunks.json"));
  const topicIndex = readJson(path.join(base, "indexes", "topic_tfidf.json"));
  const chunkIndex = readJson(path.join(base, "indexes", "chunk_tfidf.json"));
  const checkpointIndex = readJson(path.join(base, "indexes", "checkpoint_tfidf.json"));
  const chunkById = {};
  const topicById = {};
  const checkpointById = {};
  for (const c of chunks) chunkById[c.chunk_id] = c;
  for (const t of topics) topicById[t.topic_id] = t;
  for (const c of checkpoints) checkpointById[c.checkpoint_id] = c;
  const chunkToTopics = {};
  for (const [topicId, chunkIds] of Object.entries(topicToChunks)) {
    for (const id of chunkIds) {
      if (!chunkToTopics[id]) chunkToTopics[id] = [];
      chunkToTopics[id].push(topicId);
    }
  }
  const carClaimsBySpeaker = {
    all: extractCarClaims(chunks, null),
    "User 1": extractCarClaims(chunks, "User 1"),
    "User 2": extractCarClaims(chunks, "User 2"),
  };
  cache = {
    topics,
    chunks,
    checkpoints,
    facts,
    persona,
    topicToChunks,
    chunkToTopics,
    chunkById,
    topicById,
    checkpointById,
    topicIndex,
    chunkIndex,
    checkpointIndex,
    carClaimsBySpeaker,
  };
  return cache;
}

function isPersonaQuery(q) {
  return (
    /\b(persona|habit|habits|talk|style|personality|communication style)\b/i.test(q) ||
    /\b(kind|type|sort)\s+of\s+person\b/i.test(q) ||
    /\bwhat\s+are\s+they\s+like\b/i.test(q)
  );
}

function personaIntent(q) {
  if (/\b(talk|style|communication|speak|tone|emoji|messages?)\b/i.test(q)) return "style";
  if (/\b(habit|habits|routine|routines|usually|often)\b/i.test(q)) return "habits";
  if (/\b(personality|traits?|kind\s+of\s+person|what\s+are\s+they\s+like)\b/i.test(q)) return "overview";
  if (/\b(facts?|personal facts?|about them|who are they)\b/i.test(q)) return "facts";
  return "overview";
}

function itemValue(item) {
  return item?.habit || item?.fact || item?.trait || "";
}

function cleanPersonaList(items, limit = 5) {
  const bad = /\b(a city|john|jason|ben|hannah|doing|sure|thanks|that book|doing that|the music video)\b/i;
  const seen = new Set();
  const values = [];
  for (const item of items || []) {
    const value = itemValue(item).replace(/\s+/g, " ").trim();
    if (!value || bad.test(value)) continue;
    const key = value.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    values.push(value);
    if (values.length >= limit) break;
  }
  return values;
}

function personaEvidenceIds(persona, intent) {
  if (intent === "style") return persona.communication_style?.evidence_msg_ids || [];
  const sections =
    intent === "habits"
      ? [persona.habits || []]
      : intent === "facts"
        ? [persona.personal_facts || []]
        : [persona.personality_traits || [], persona.habits || [], persona.personal_facts || []];
  const ids = [];
  for (const section of sections) {
    for (const item of section.slice(0, 6)) {
      for (const id of item.evidence_msg_ids || []) ids.push(id);
    }
  }
  return [...new Set(ids)].slice(0, 24);
}

function chunksForMessageIds(data, ids, speaker, limit = 3) {
  if (!ids.length) return [];
  const wanted = new Set(ids);
  const scored = [];
  for (const chunk of data.chunks) {
    let hits = 0;
    for (const id of wanted) {
      if (id >= chunk.start_msg_id && id <= chunk.end_msg_id) hits += 1;
    }
    if (!hits) continue;
    const speakerBoost = speaker && chunk.text.includes(`${speaker}:`) ? 0.25 : 0;
    scored.push({ chunk, score: hits + speakerBoost });
  }
  scored.sort((a, b) => b.score - a.score || a.chunk.start_msg_id - b.chunk.start_msg_id);
  return scored.slice(0, limit).map((s) => s.chunk);
}

function topicsForChunks(data, chunks, limit = 3) {
  const ids = new Set();
  for (const chunk of chunks) {
    for (const topicId of data.chunkToTopics[chunk.chunk_id] || []) ids.add(topicId);
  }
  return [...ids]
    .map((id) => data.topicById[id])
    .filter(Boolean)
    .slice(0, limit);
}

function checkpointsForChunks(data, chunks, limit = 2) {
  const found = [];
  const seen = new Set();
  for (const chunk of chunks) {
    for (const checkpoint of data.checkpoints) {
      if (seen.has(checkpoint.checkpoint_id)) continue;
      if (chunk.end_msg_id < checkpoint.start_msg_id || chunk.start_msg_id > checkpoint.end_msg_id) continue;
      seen.add(checkpoint.checkpoint_id);
      found.push(checkpoint);
      break;
    }
    if (found.length >= limit) break;
  }
  return found;
}

function composePersonaAnswer(data, query, personaBundle, personaSpeaker) {
  const resolved = resolvePersona(personaBundle, personaSpeaker);
  const persona = resolved.persona || {};
  const intent = personaIntent(query);
  const style = persona.communication_style || {};
  const habits = cleanPersonaList(persona.habits, 6);
  const facts = cleanPersonaList(persona.personal_facts, 4);
  const traits = cleanPersonaList(persona.personality_traits, 4);
  const evidenceIds = personaEvidenceIds(persona, intent);
  const chunks = chunksForMessageIds(data, evidenceIds, resolved.speaker, 3);
  const topics = topicsForChunks(data, chunks, 3);
  const checkpoints = checkpointsForChunks(data, chunks, 2);

  let direct;
  if (intent === "style") {
    direct = [
      `${resolved.speaker} generally writes ${style.avg_length || "unknown-length"} messages.`,
      `Their tone is ${(style.tone || []).join(", ") || "unknown"}, and emoji usage is ${style.emoji_usage || "unknown"}.`,
      "This is based on aggregate message-level signals such as message length, punctuation, questions, exclamation marks, and emoji presence.",
    ].join(" ");
  } else if (intent === "habits") {
    direct = habits.length
      ? `${resolved.speaker}'s strongest repeated habit signals are: ${habits.join(", ")}.`
      : `I do not have enough verified habit evidence for ${resolved.speaker}.`;
  } else if (intent === "facts") {
    direct = facts.length
      ? `Repeated personal-fact signals for ${resolved.speaker} include: ${facts.join(", ")}.`
      : `I do not have enough clean personal-fact evidence for ${resolved.speaker}.`;
  } else {
    const parts = [];
    if (traits.length) parts.push(`traits: ${traits.join(", ")}`);
    if (habits.length) parts.push(`habits/interests: ${habits.slice(0, 4).join(", ")}`);
    parts.push(`communication: ${style.avg_length || "unknown"} messages, ${(style.tone || []).join(", ") || "unknown"} tone, ${style.emoji_usage || "unknown"} emoji usage`);
    direct = `${resolved.speaker} appears to be characterized by ${parts.join("; ")}.`;
  }

  const topicLine = topics.length
    ? `Relevant topic checkpoints: ${topics.map((t) => `${t.topic_id} (${t.start_msg_id}-${t.end_msg_id})`).join(", ")}.`
    : "No topic checkpoint was needed beyond persona evidence.";
  const checkpointLine = checkpoints.length
    ? `Relevant 100-message checkpoints: ${checkpoints.map((c) => `${c.checkpoint_id} (${c.start_msg_id}-${c.end_msg_id})`).join(", ")}.`
    : "No 100-message checkpoint was needed beyond persona evidence.";
  const evidenceLines = chunks
    .map((c) => `- [${c.start_msg_id}-${c.end_msg_id}] ${c.text.slice(0, 220)}...`)
    .join("\n");
  const evidenceIdLine = evidenceIds.length ? `Persona evidence message IDs: ${evidenceIds.slice(0, 12).join(", ")}.` : "";

  return {
    answer: [
      direct,
      "",
      topicLine,
      checkpointLine,
      evidenceIdLine,
      "",
      evidenceLines ? `Supporting conversation evidence:\n${evidenceLines}` : "No direct evidence chunks were found.",
    ]
      .filter((line) => line !== "")
      .join("\n"),
    used_topics: topics.map((t) => t.topic_id),
    used_chunks: chunks.map((c) => c.chunk_id),
    used_checkpoints: checkpoints.map((c) => c.checkpoint_id),
    used_persona_fields:
      intent === "style"
        ? ["communication_style"]
        : intent === "habits"
          ? ["habits"]
          : intent === "facts"
            ? ["personal_facts"]
            : ["habits", "personal_facts", "personality_traits", "communication_style"],
  };
}

function isCarQuery(q) {
  return /\b(car|cars|vehicle|drive|drives|driving|ride|use)\b/i.test(q);
}

function carIntent(q) {
  if (/\b(name|model|called|make)\b/i.test(q)) return "model";
  return "vehicle";
}

function sentenceSplit(text) {
  return String(text || "")
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function speakerTurns(text) {
  const source = String(text || "");
  const markerPattern = /\b(User\s+\d+):\s*/g;
  const markers = [...source.matchAll(markerPattern)];
  if (!markers.length) return [{ speaker: "", text: source }];

  return markers.map((match, idx) => {
    const start = match.index + match[0].length;
    const end = idx + 1 < markers.length ? markers[idx + 1].index : source.length;
    return {
      speaker: match[1],
      text: source.slice(start, end).trim(),
    };
  });
}

function extractCarClaims(chunks, speaker) {
  const modelPattern =
    /\b(?:\d{4}\s+)?(?:chevrolet|chevy|ford|toyota|honda|nissan|tesla|bmw|mercedes|audi|subaru|jeep|kia|hyundai|volkswagen|vw|volvo|mazda|lexus|dodge|ram|gmc|cadillac|buick|lincoln|porsche|ferrari|lamborghini|maserati|jaguar|land rover|range rover|impala|corvette|camaro|mustang|outback|civic|accord|camry|corolla|prius|rav4|cr-v|wrangler|model\s+[syx3])\b(?:\s+[a-z0-9-]+){0,3}/gi;
  const typePattern = /\b(classic car|classic cars|sports car|truck|suv|sedan|hatchback|convertible|motorcycle)\b/gi;
  const useOrOwnershipContext =
    /\b(my|mine|i\s+(?:have|own|owned|drive|drove|use|used|got\s+back\s+from\s+(?:a\s+)?(?:nice\s+)?drive\s+in|take\s+out))\b/i;
  const weakIntentContext = /\b(want|wanted|always wanted|wish|what kind of car|favorite car)\b/i;
  const claims = [];

  for (const chunk of chunks) {
    for (const turn of speakerTurns(chunk.text)) {
      if (speaker && turn.speaker && turn.speaker !== speaker) continue;
      for (const sentence of sentenceSplit(turn.text)) {
        const models = sentence.match(modelPattern) || [];
        const types = sentence.match(typePattern) || [];
        if (!models.length && !types.length) continue;
        if (!useOrOwnershipContext.test(sentence) || weakIntentContext.test(sentence)) continue;
        const value = [...models, ...types][0].replace(/\s+/g, " ").trim();
        const isType = types.some((t) => t.toLowerCase() === value.toLowerCase());
        claims.push({
          value,
          claim_type: isType ? "type" : "model",
          sentence: turn.speaker ? `${turn.speaker}: ${sentence}` : sentence,
          start_msg_id: chunk.start_msg_id,
          end_msg_id: chunk.end_msg_id,
        });
      }
    }
  }

  const seen = new Set();
  return claims.filter((claim) => {
    const key = `${claim.value.toLowerCase()}|${claim.sentence.toLowerCase()}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function composeCarAnswer(data, retrieved, personaSpeaker) {
  const intent = carIntent(retrieved.query || "");
  const localClaims = extractCarClaims(retrieved.chunks, personaSpeaker);
  const modelClaims = localClaims.filter((claim) => claim.claim_type === "model");
  const typeClaims = localClaims.filter((claim) => claim.claim_type === "type");
  const evidenceTopics = topicsForChunks(data, retrieved.chunks, 3);
  const evidenceCheckpoints = checkpointsForChunks(data, retrieved.chunks, 2);
  const topicLine = evidenceTopics.length
    ? `Relevant topic checkpoints: ${evidenceTopics
        .slice(0, 3)
        .map((t) => `${t.topic_id} (${t.start_msg_id}-${t.end_msg_id})`)
        .join(", ")}.`
    : "No strong topic match was found.";
  const checkpointLine = evidenceCheckpoints.length
    ? `Relevant 100-message checkpoints: ${evidenceCheckpoints
        .slice(0, 2)
        .map((c) => `${c.checkpoint_id} (${c.start_msg_id}-${c.end_msg_id})`)
        .join(", ")}.`
    : "No strong 100-message checkpoint match was found.";

  if (!localClaims.length) {
    const evidenceLines = retrieved.chunks
      .slice(0, 3)
      .map((c) => `- [${c.start_msg_id}-${c.end_msg_id}] ${c.text.slice(0, 180)}...`)
      .join("\n");
    return [
      "I could not confidently identify a specific car from the retrieved evidence.",
      personaSpeaker
        ? `The retrieved snippets mention cars, but they do not clearly state which car ${personaSpeaker} uses.`
        : "The retrieved snippets mention cars, but they do not clearly state which car the selected person uses.",
      "",
      topicLine,
      checkpointLine,
      "",
      evidenceLines ? `Closest evidence:\n${evidenceLines}` : "No direct conversation evidence was found.",
    ].join("\n");
  }

  if (intent === "model" && !modelClaims.length) {
    const typeList = [...new Set(typeClaims.map((claim) => normalizeCarValue(claim.value)))].slice(0, 4);
    const evidenceLines = localClaims
      .slice(0, 3)
      .map((claim) => `- [${claim.start_msg_id}-${claim.end_msg_id}] ${claim.sentence.slice(0, 220)}...`)
      .join("\n");
    return [
      `I could not identify a specific car name or model for ${personaSpeaker || "the selected speaker"} from the retrieved evidence.`,
      typeList.length
        ? `The evidence only supports vehicle-type mentions: ${typeList.join(", ")}.`
        : "The evidence mentions vehicles, but not a named model.",
      "Because the CSV contains many separate conversations, I should not collapse repeated vehicle mentions into one global answer.",
      "",
      topicLine,
      checkpointLine,
      "",
      `Evidence from conversation:\n${evidenceLines}`,
    ].join("\n");
  }

  const answerClaims = intent === "model" ? modelClaims : localClaims;
  const ranked = rankCarClaims(answerClaims);
  const strongest = ranked[0];
  const evidenceLines = ranked
    .slice(0, 3)
    .flatMap((item) => item.examples.slice(0, 1))
    .map((claim) => `- [${claim.start_msg_id}-${claim.end_msg_id}] ${claim.sentence.slice(0, 220)}...`)
    .join("\n");
  const alternatives = ranked
    .slice(1, 3)
    .map((item) => item.value)
    .join(", ");
  const hasMultiple = ranked.length > 1;

  return [
    intent === "model" && hasMultiple
      ? `I found multiple car model/name mentions for ${personaSpeaker || "the selected speaker"}, so there is no single global car name to return.`
      : intent === "model"
        ? `The retrieved evidence names this car/model for ${personaSpeaker || "the selected speaker"}: ${strongest.value}.`
        : hasMultiple
          ? `I found multiple vehicle mentions for ${personaSpeaker || "the selected speaker"}; the strongest retrieved signal is ${strongest.value}.`
          : `The retrieved evidence points to this vehicle signal for ${personaSpeaker || "the selected speaker"}: ${strongest.value}.`,
    alternatives ? `Other retrieved vehicle mentions: ${alternatives}.` : "",
    "This answer is based on retrieved evidence chunks. The CSV contains many separate conversations, so repeated User 1/User 2 labels should not be treated as one continuous real-world identity.",
    "",
    topicLine,
    checkpointLine,
    "",
    `Evidence from conversation:\n${evidenceLines}`,
  ].join("\n");
}

function normalizeCarValue(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/\bclassic cars\b/g, "classic car")
    .replace(/\s+/g, " ")
    .trim();
}

function rankCarClaims(claims) {
  const grouped = new Map();
  for (const claim of claims) {
    const key = normalizeCarValue(claim.value);
    if (!key) continue;
    if (!grouped.has(key)) {
      grouped.set(key, { value: claim.value, count: 0, examples: [] });
    }
    const entry = grouped.get(key);
    entry.count += 1;
    if (entry.examples.length < 3) entry.examples.push(claim);
  }
  return Array.from(grouped.values()).sort((a, b) => b.count - a.count || a.value.localeCompare(b.value));
}

function selectedCarFactLabels(query, claims, speaker) {
  const intent = carIntent(query);
  const filtered = intent === "model" ? claims.filter((claim) => claim.claim_type === "model") : claims;
  return rankCarClaims(filtered.length ? filtered : claims)
    .slice(0, 3)
    .map((r) => `${speaker || "speaker"}:${r.value}`);
}

function carEvidenceChunks(data, speaker, query, limit = 5) {
  const intent = carIntent(query);
  const claims = speaker ? data.carClaimsBySpeaker?.[speaker] || [] : data.carClaimsBySpeaker?.all || [];
  const preferred = intent === "model" ? claims.filter((claim) => claim.claim_type === "model") : claims;
  const pool = preferred.length ? preferred : claims;
  const chunks = [];
  const seen = new Set();
  for (const claim of pool) {
    const chunk = findChunkByMsgId(data.chunks, claim.start_msg_id);
    if (!chunk || seen.has(chunk.chunk_id)) continue;
    seen.add(chunk.chunk_id);
    chunks.push(chunk);
    if (chunks.length >= limit) break;
  }
  return chunks;
}

function resolvePersona(personaBundle, speaker) {
  if (!personaBundle || !personaBundle.persona_by_speaker) {
    return { persona: personaBundle || {}, speaker: speaker || "unknown" };
  }
  const fallbackSpeaker = personaBundle.default_speaker || personaBundle.speakers?.[0];
  const chosen = speaker && personaBundle.persona_by_speaker[speaker] ? speaker : fallbackSpeaker;
  return { persona: personaBundle.persona_by_speaker[chosen] || {}, speaker: chosen || "unknown" };
}

function formatPersona(personaBundle, speaker) {
  const resolved = resolvePersona(personaBundle, speaker);
  const persona = resolved.persona || {};
  const habits = (persona.habits || []).slice(0, 5).map((h) => h.habit || h.trait || h.fact).filter(Boolean);
  const facts = (persona.personal_facts || []).slice(0, 5).map((f) => f.fact || f.trait || f.habit).filter(Boolean);
  const traits = (persona.personality_traits || []).slice(0, 5).map((t) => t.trait || t.fact || t.habit).filter(Boolean);
  const style = persona.communication_style || {};
  return {
    text: [
      `Persona speaker: ${resolved.speaker}.`,
      habits.length ? `Habits: ${habits.join(", ")}.` : "",
      facts.length ? `Personal facts: ${facts.join(", ")}.` : "",
      traits.length ? `Personality traits: ${traits.join(", ")}.` : "",
      `Communication style: ${style.avg_length || "unknown"} messages, tone ${
        (style.tone || []).join(", ") || "unknown"
      }, emoji usage ${style.emoji_usage || "unknown"}.`,
    ]
      .filter(Boolean)
      .join(" "),
    evidence: style.evidence_msg_ids || [],
  };
}

function hierarchicalRetrieve(data, query, selectedSpeaker) {
  const queryTokens = tokenize(query);
  const patternBoosts = buildPatternBoosts(query, data);
  const speakerMatch = query.match(/\buser\s+([12])\b/i);
  const speakerHint = speakerMatch ? `User ${speakerMatch[1]}` : selectedSpeaker || null;
  const factResult = retrieveFacts(data, query, speakerHint);
  const topicScores = scoreIndex(data.topicIndex, query, 8);
  const checkpointScores = scoreIndex(data.checkpointIndex, query, 3);
  const chunkScores = scoreIndex(data.chunkIndex, query, 24);

  const topicScoreById = new Map(topicScores.map((t) => [t.doc_id, t.score]));
  const mergedScores = new Map();
  for (const c of chunkScores) mergedScores.set(c.doc_id, c.score);
  for (const [docId, boost] of patternBoosts.entries()) {
    mergedScores.set(docId, (mergedScores.get(docId) || 0) + boost);
  }

  let boostedChunks = Array.from(mergedScores.entries()).map(([doc_id, baseScore]) => {
    const topics = data.chunkToTopics[doc_id] || [];
    let boost = 0;
    for (const topicId of topics) {
      const score = topicScoreById.get(topicId);
      if (score && score > boost) boost = score;
    }
    const chunk = data.chunkById[doc_id];
    const overlap = chunk ? overlapBoost(chunk.text, queryTokens) : 0;
    return { doc_id, score: baseScore + boost * 0.25 + overlap * 0.15 };
  });

  if (speakerHint) {
    boostedChunks = boostedChunks.filter((c) => {
      const chunk = data.chunkById[c.doc_id];
      return chunk && chunk.text.includes(`${speakerHint}:`);
    });
  }

  boostedChunks.sort((a, b) => b.score - a.score);
  const factBoosted = factResult.fact_chunks.map((c) => ({ doc_id: c.chunk_id, score: 3.5 }));
  const carBoosted = isCarQuery(query)
    ? carEvidenceChunks(data, speakerHint, query, 5).map((c) => ({ doc_id: c.chunk_id, score: 4.0 }))
    : [];
  const mergedFinal = [...carBoosted, ...factBoosted, ...boostedChunks];
  const seen = new Set();
  const deduped = [];
  for (const item of mergedFinal) {
    if (seen.has(item.doc_id)) continue;
    seen.add(item.doc_id);
    deduped.push(item);
  }

  return {
    query,
    topics: topicScores.map((t) => data.topicById[t.doc_id]).filter(Boolean),
    checkpoints: checkpointScores.map((c) => data.checkpointById[c.doc_id]).filter(Boolean),
    chunks: deduped.slice(0, 5).map((c) => data.chunkById[c.doc_id]).filter(Boolean),
    facts: factResult.facts,
    scored_topics: topicScores,
    scored_chunks: boostedChunks.slice(0, 5),
    scored_checkpoints: checkpointScores,
  };
}

function composeAnswer(data, query, retrieved, persona, personaSpeaker) {
  const personaInfo = isPersonaQuery(query) ? formatPersona(persona, personaSpeaker) : null;
  if (!personaInfo && isCarQuery(query)) {
    const localClaims = extractCarClaims(retrieved.chunks, personaSpeaker);
    return {
      answer: composeCarAnswer(data, retrieved, personaSpeaker),
      used_topics: topicsForChunks(data, retrieved.chunks, 8).map((t) => t.topic_id),
      used_chunks: retrieved.chunks.map((c) => c.chunk_id),
      used_checkpoints: checkpointsForChunks(data, retrieved.chunks, 3).map((c) => c.checkpoint_id),
      used_facts: selectedCarFactLabels(query, localClaims, personaSpeaker),
      used_persona_fields: [],
    };
  }

  const topicLine = retrieved.topics.length
    ? `Relevant topics: ${retrieved.topics
        .slice(0, 3)
        .map((t) => `${t.topic_id} (${t.start_msg_id}-${t.end_msg_id})`)
        .join(", ")}.`
    : "No strong topic match was found.";
  const checkpointLine = retrieved.checkpoints.length
    ? `Relevant 100-message checkpoints: ${retrieved.checkpoints
        .slice(0, 2)
        .map((c) => `${c.checkpoint_id} (${c.start_msg_id}-${c.end_msg_id})`)
        .join(", ")}.`
    : "No strong 100-message checkpoint match was found.";
  const evidenceLines = retrieved.chunks
    .slice(0, 3)
    .map((c) => `- [${c.start_msg_id}-${c.end_msg_id}] ${c.text.slice(0, 180)}...`)
    .join("\n");
  const factLines = (retrieved.facts || [])
    .map((f) => `- ${f.speaker}: ${f.value} (${f.count} mentions)`)
    .join("\n");
  const checkpointEvidence = retrieved.checkpoints
    .slice(0, 2)
    .map((c) => `- [${c.start_msg_id}-${c.end_msg_id}] ${c.summary.slice(0, 180)}...`)
    .join("\n");

  let answer = `${topicLine}\n${checkpointLine}\n\n`;
  if (personaInfo) {
    answer += `${personaInfo.text}\n\n`;
  }
  if (retrieved.chunks.length) {
    answer += `Evidence from conversation:\n${evidenceLines}`;
  } else {
    answer += "I couldn't find enough direct evidence for a confident answer.";
  }
  if (factLines) {
    const factIntro =
      (retrieved.facts || []).length > 1
        ? "Multiple conversations match this question; top fact matches:"
        : "Fact match:";
    answer += `\n\n${factIntro}\n${factLines}`;
  }
  if (checkpointEvidence) {
    answer += `\n\n100-message checkpoint context:\n${checkpointEvidence}`;
  }

  return {
    answer,
    used_topics: retrieved.topics.map((t) => t.topic_id),
    used_chunks: retrieved.chunks.map((c) => c.chunk_id),
    used_checkpoints: retrieved.checkpoints.map((c) => c.checkpoint_id),
    used_facts: (retrieved.facts || []).map((f) => `${f.speaker}:${f.value}`),
    used_persona_fields: personaInfo ? ["habits", "personal_facts", "personality_traits", "communication_style"] : [],
  };
}

exports.handler = async (event) => {
  try {
    const body = event.body ? JSON.parse(event.body) : {};
    const query = String(body.query || "").trim();
    const personaSpeaker = String(body.persona_speaker || body.personaSpeaker || "").trim();
    if (!query) {
      return { statusCode: 400, body: JSON.stringify({ error: "query is required" }) };
    }

    const data = loadData();
    if (isPersonaQuery(query)) {
      const result = composePersonaAnswer(data, query, data.persona, personaSpeaker);
      return {
        statusCode: 200,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(result),
      };
    }

    const retrieved = hierarchicalRetrieve(data, query, personaSpeaker);
    const result = composeAnswer(data, query, retrieved, data.persona, personaSpeaker);
    return {
      statusCode: 200,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(result),
    };
  } catch (error) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message || "internal server error" }),
    };
  }
};


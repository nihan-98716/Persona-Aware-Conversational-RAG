# Persona-Aware Conversational RAG

This project builds an end-to-end local RAG system over a CSV of conversations. It processes messages in chronological order, creates topic checkpoints, creates independent 100-message checkpoints, extracts speaker-specific persona data, and serves a simple chatbot through a Netlify function.

## Submission Links

- Loom demo: https://www.loom.com/share/2cac838014d14915907ff5ef42264614
- GitHub repo: add the public GitHub URL after pushing this folder
- Hosted chatbot: https://persona-aware-conversational-rag-anoop.netlify.app
- Local chatbot: `http://localhost:8888`

## What This Implements

- Chronological parsing of `conversations.csv`
- Topic checkpoints when the topic changes
- Independent 100-message checkpoints
- Message chunks for fine-grained retrieval
- Sparse TF-IDF indexes for topics, chunks, and 100-message checkpoints
- Speaker-specific persona JSON with habits, personal facts, personality traits, and communication style
- Chatbot that combines topic summaries, message chunks, 100-message checkpoints, extracted facts, and persona data

Current generated artifact counts:

- Messages: `191592`
- Topic checkpoints: `11912`
- 100-message checkpoints: `1916`
- Message chunks: `32022`
- Extracted facts: `6526`
- Persona speakers: `User 1`, `User 2`

## Project Structure

```text
.
├── conversations.csv
├── netlify.toml
├── README.md
├── pipeline/
│   ├── requirements.txt
│   ├── output/messages.jsonl
│   └── src/build_artifacts.py
└── app/
    ├── index.html
    ├── src/app.js
    ├── src/styles.css
    ├── netlify/functions/chat.js
    └── data/
        ├── topics.json
        ├── checkpoints_100.json
        ├── chunks.json
        ├── facts.json
        ├── persona.json
        ├── meta.json
        └── indexes/
            ├── topic_to_chunks.json
            ├── topic_tfidf.json
            ├── chunk_tfidf.json
            └── checkpoint_tfidf.json
```

## Requirements

- Python 3.11+
- Node.js 20+
- Netlify CLI

Install Netlify CLI if needed:

```powershell
npm install -g netlify-cli
```

## Setup

```powershell
cd "C:\Users\Public\Projects\Persona-Aware Conversational RAG"
python -m venv .venv-rag
.\.venv-rag\Scripts\Activate.ps1
pip install -r .\pipeline\requirements.txt
```

## Build Data Artifacts

Run this after changing the CSV or pipeline:

```powershell
.\.venv-rag\Scripts\python .\pipeline\src\build_artifacts.py --csv .\conversations.csv --out .\app\data --persona-speaker "User 1"
```

Generated files:

- `pipeline/output/messages.jsonl`: canonical chronological messages
- `app/data/topics.json`: topic checkpoints
- `app/data/checkpoints_100.json`: independent 100-message summaries
- `app/data/chunks.json`: retrieval chunks
- `app/data/facts.json`: extracted direct facts such as moving/live/work/study statements
- `app/data/persona.json`: structured persona data
- `app/data/indexes/topic_to_chunks.json`: topic-to-chunk mapping
- `app/data/indexes/topic_tfidf.json`: topic summary search index
- `app/data/indexes/chunk_tfidf.json`: message chunk search index
- `app/data/indexes/checkpoint_tfidf.json`: 100-message checkpoint search index

## Run Chatbot Locally

```powershell
netlify dev
```

Open:

```text
http://localhost:8888
```

Function endpoint:

```text
POST /.netlify/functions/chat
```

Example body:

```json
{
  "query": "What kind of person is this user?",
  "persona_speaker": "User 1"
}
```

The UI includes a persona speaker dropdown for `User 1` and `User 2`.

## Deploy to Cloud

The project is configured for Netlify in `netlify.toml`.

```powershell
netlify login
netlify init
netlify deploy --prod
```

Netlify config:

- Publish directory: `app`
- Functions directory: `app/netlify/functions`
- Required data artifacts are included in the function bundle through `included_files`

After deployment, add the production URL to the "Submission Links" section.

## How Topic Changes Are Detected

Topic checkpointing is implemented in `pipeline/src/build_artifacts.py`.

The pipeline processes messages chronologically. Each CSV row is treated as a separate conversation/day, and topic segmentation never crosses a row boundary. Inside each row, messages are processed in order and a topic boundary is created when the local topic drift is strong enough.

Signals used for topic splitting:

- Semantic drift from a rolling segment centroid using `HashingVectorizer`
- Named-entity shift between the current message and the active segment
- Continuity dampening for short acknowledgement/follow-up messages such as "yes", "okay", "thanks"
- Minimum topic length to avoid noisy micro-topics
- Maximum topic length as a safety split
- Two-step hysteresis through `pending_boundary` so one noisy message does not immediately split a topic
- Tiny segment post-merge within the same row

Each topic checkpoint stores:

- `topic_id`
- `start_msg_id`
- `end_msg_id`
- `message_count`
- `keywords`
- `topic_summary`
- `evidence_msg_ids`

The topic summary is segment-only. It is generated from the messages inside that segment, not from the full conversation.

## How 100-Message Checkpoints Work

The 100-message checkpoints are independent of topics. The pipeline walks through all messages in chronological order and creates one checkpoint per 100 messages.

Each checkpoint stores:

- `checkpoint_id`
- `start_msg_id`
- `end_msg_id`
- `message_count`
- `summary`
- `highlights`

These checkpoints are indexed separately and retrieved separately from topic checkpoints.

## How Retrieval Works

Retrieval is implemented in `app/netlify/functions/chat.js`.

The chatbot uses hierarchical retrieval:

1. Score topic summaries with a sparse TF-IDF index.
2. Score message chunks with a sparse TF-IDF index.
3. Boost chunks that overlap with highly ranked topic checkpoints.
4. Retrieve 100-message checkpoints from their own sparse TF-IDF index.
5. For direct fact-style questions, use extracted facts from `facts.json`.
6. For car/vehicle questions, use speaker-aware evidence extraction from retrieved chunks.
7. For persona questions, answer from `persona.json` first and attach supporting topic/chunk/checkpoint evidence.

The final answer includes:

- A direct answer
- Relevant topic checkpoint IDs
- Relevant 100-message checkpoint IDs
- Supporting message chunks
- Persona fields used, when applicable

Important behavior: the CSV contains many separate conversations using repeated labels like `User 1` and `User 2`. For factual questions, the bot avoids pretending that all repeated `User 1` messages are one single real-world person. If multiple separate conversations support different answers, it says so and shows the evidence.

## How Persona Is Built

Persona extraction is implemented in `pipeline/src/build_artifacts.py`.

The persona builder uses a candidate -> consolidate -> verify pipeline.

Candidate extraction:

- Habits from phrases like `I like`, `I love`, `I enjoy`, `I usually`, `I often`
- Personal facts from explicit statements like `I am a/an`, `I'm a/an`, `I study`, `I work as/in/at/for`, `I live in`, `My name is`
- Personality traits from repeated communication signals such as supportive language and enthusiastic punctuation
- Communication style from aggregate message length, question rate, exclamation rate, and emoji rate

Speaker handling:

- Persona is built separately for each speaker.
- `persona.json` stores `persona_by_speaker.User 1` and `persona_by_speaker.User 2`.
- The UI lets the evaluator choose which speaker to inspect.

Verification:

- Repeated candidates are retained.
- High-confidence explicit statements are retained.
- Each retained persona item includes `evidence_msg_ids`.
- No persona item is created without conversation evidence.

Persona schema:

```json
{
  "default_speaker": "User 1",
  "speakers": ["User 1", "User 2"],
  "persona_by_speaker": {
    "User 1": {
      "habits": [],
      "personal_facts": [],
      "personality_traits": [],
      "communication_style": {}
    }
  }
}
```

## Example Queries

Use these in the chatbot:

```text
What kind of person is this user?
What are their habits?
How do they talk?
What is the communication style of this user?
What personal facts are known about this user?
What hobbies or interests does this user mention?
What kind of car does the person use?
What is the name or model of the car the user has?
What job or profession does the user mention?
What does the user like to do for fun?
```

Switch between `Persona: User 1` and `Persona: User 2` to verify speaker-specific behavior.

## Screenshots and Demo

Demo video:

- https://www.loom.com/share/2cac838014d14915907ff5ef42264614

Recommended screenshots for submission:

- Chatbot answering `What kind of person is this user?`
- Chatbot answering `What are their habits?`
- Chatbot answering `How do they talk?`
- `app/data/persona.json`
- `app/data/topics.json`
- `app/data/checkpoints_100.json`

## Limitations

- This project intentionally avoids external LLM APIs. Answers are generated with local rules, sparse TF-IDF retrieval, extracted facts, and structured persona data.
- Because the CSV contains many separate conversations with repeated `User 1` and `User 2` labels, factual answers can have multiple valid evidence-backed matches.
- Persona is speaker-aggregated across the CSV. The bot therefore reports repeated speaker-level signals and keeps evidence IDs visible.
- Some extracted persona facts can still be noisy because the system uses lightweight rules rather than a large language model.

## Requirement Mapping

| Task requirement | Implementation |
| --- | --- |
| Process conversations chronologically | `parse_csv_rows()` writes ordered `global_msg_id` values |
| Detect topic changes over time | `segment_topics()` and `segment_topics_in_span()` |
| Topic checkpoint summaries | `app/data/topics.json` |
| 100-message checkpoint summaries | `app/data/checkpoints_100.json` |
| Retrieve topic summaries | `topic_tfidf.json` in `chat.js` |
| Retrieve message chunks | `chunk_tfidf.json` and `chunks.json` |
| Combine topic + chunks for answers | `hierarchicalRetrieve()` and answer composers |
| Extract habits | `candidate_persona_entries()` |
| Extract personal facts | `candidate_persona_entries()` and `facts.json` |
| Extract personality traits | `candidate_persona_entries()` |
| Extract communication style | `build_communication_style()` |
| Store persona as JSON | `app/data/persona.json` |
| Chatbot for persona questions | `app/index.html` + `app/netlify/functions/chat.js` |
| Avoid complete dependency on external APIs | Local Python/Node logic, no external LLM calls |

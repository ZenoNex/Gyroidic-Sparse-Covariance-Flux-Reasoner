# Data Pipeline

> Conversational API ingestion, runtime pressure generation, and textbook-quality filtering.

---

## 1. Conversational API Ingestor

**Source**: [`src/data/conversational_api_ingestor.py`](../src/data/conversational_api_ingestor.py) (1,107 lines)

Unified conversational data ingestion from multiple API sources.

### Data Model

| Dataclass | Fields |
|-----------|--------|
| `ConversationTurn` | `speaker_id`, `text`, `timestamp`, `embedding`, `affordance_gradients` |
| `Conversation` | `conversation_id`, `turns[]`, `context`, `source`, `labels`, `pressure_signature` |

### Ingestors

| Class | Source | Capabilities |
|-------|--------|-------------|
| `HuggingFaceConversationalIngestor` | HF Hub API | LMSYS-chat-1m, OASST2, UltraChat. Direct API + `datasets` lib. Synthetic fallback when HF unavailable |
| `RedditConversationalIngestor` | Reddit API | OAuth2 auth, subreddit posts, threaded comments  conversation trees |
| `ConvoKitIngestor` | ConvoKit library | Labeled corpora (Wikipedia talk, Supreme Court, etc.) |

### Orchestrator

`ConversationalAPIIngestor`  coordinates all three ingestors:
- `ingest_huggingface_dataset(dataset_id, max_samples)`  parsed `Conversation[]`
- `ingest_reddit_subreddit(subreddit, max_posts)`  threaded `Conversation[]`
- `ingest_convokit_corpus(corpus_name)`  labeled `Conversation[]`
- Caching via JSON serialization to `data/conversational_cache/`

### Processor

`ConversationalDataProcessor` transforms raw conversations for the gyroidic system:
- `compute_text_embedding(text)`  `[1, dim]` via `CanonicalProjector`
- `compute_affordance_gradients(text)`  dict of soft signals (code, math, conversation, API, etc.)
- `generate_pressure_signature(conversation)`  polynomial CRT-based pressure tensor

---

## 2. Pressure Ingestor

**Source**: [`src/data/pressure_ingestor.py`](../src/data/pressure_ingestor.py) (671 lines)

Runtime code generation for constraint forcing. **No polite APIs. No silent failures.**

### Phase Model

```mermaid
graph LR
    U["UNDISCOVERED"] --> D["DISCOVERED"]
    D --> I["INDEXED"]
    I --> M["MATERIALIZED"]
    M --> V["VERIFIED"]
    D -.->|fail| F["FAILED"]
    I -.->|fail| F
    M -.->|fail| F
```

Each source transitions through 4 phases, with code generated dynamically per phase:

| Phase | Method | Purpose |
|-------|--------|---------|
| Discover | `_generate_discover_code(source)` | Locate data sources, detect formats |
| Index | `_generate_index_code(source, state)` | Build structural index from discovered data |
| Fetch | `_generate_fetch_code(source, state)` | Retrieve and convert to constraint tensors |
| Verify | `_generate_verify_code(source, state)` | Validate constraints against expected properties |

### Key Design

- **Assume failure, prove success**: `assume_failure()`  must call `prove_success()` with evidence
- `SourceDescriptor`: grammar defining discover/index/fetch/verify patterns per source
- `force_pressure_ingestion(source_names)`  materializes across all sources
- `get_constraint_batch(batch_size)`  `[batch, dim]` tensors for gyroidic expansion

---

## 3. Textbook Filter

**Source**: [`src/data/textbook_filter.py`](../src/data/textbook_filter.py) (335 lines)

Phi-1 "Textbooks Are All You Need" inspired quality filtering with **non-scalar admissibility**.

### Quality Dimensions

| Dimension | Threshold | Measures |
|-----------|-----------|----------|
| `self_contained` | 0.3 | Minimal external dependencies, complete examples |
| `instructive` | 0.3 | Teaching patterns, explanations, commented code |
| `algorithmic` | 0.15 | Algorithm keywords, data structure mentions |
| `clarity` | 0.3 | Readability, structure, formatting quality |
| `structural_honesty` | 0.8 | Anti-lobotomy filter, rejects placeholders/TODOs |

### Admissibility

Admissible iff **ALL** dimension gates pass independently  no cross-domain scalarization.

```
QualityReport.admissible = all(dimension_gates.values())
```

| Method | Purpose |
|--------|---------|
| `assess(text, source)`  `QualityReport` | Score all 4 dimensions with per-dimension gates |
| `filter_batch(texts)`  `[{text, admissible, report}]` | Batch filtering |
| `get_statistics(reports)`  aggregate stats | Pass rates, flag counts |

Detects code vs. instruction content automatically (`_is_code`) and applies different heuristics (`_assess_code` vs. `_assess_instruction`).

---

## 4. ArXiv Sovereign Lore Ingestor

**Source**: [`src/data/knowledge_ingestor.py`](../src/data/knowledge_ingestor.py) (190+ lines)

Background slow-drip ingestion pipeline fetching high-density lore residues from ArXiv OAI-PMH.

### Dynamic Meta-State Steering

Rather than cycling hardcoded topics uniformly, the ingestor dynamically samples categories based on the reasoner's live `meta_state` trajectory:

1. **Archetypal Vector Signatures**: Every category maps deterministically to an orthogonal signature vector in engine space:
   ```python
   v = _get_category_signature(category_name)
   ```
2. **Cosine Alignment**: Computes similarities between live `meta_state` and category vectors.
3. **Softmax Sampling**: Feeds similarities into Softmax ($T=0.2$) to construct a steerable probability distribution, drawing the next ingest query.

### Active Category Corpus (Science & Humanities)

Specifically includes hard-to-find humanities and societal overlaps to maintain philosophical depth:

| Domain | ArXiv Set Identifier | Description |
|---|---|---|
| **Mathematics** | `math`, `math.LO` | General Math and Mathematical Logic |
| **Quantum Physics** | `physics:quant-ph` | Quantum Physics / Topology |
| **AI & ML** | `cs:AI` | Artificial Intelligence |
| **History of Math** | `math.HO` | History and Overview of Mathematics |
| **History of Physics** | `physics:hist-ph` | **Deep Humanities**: History/Philosophy of Physics |
| **Computers & Society** | `cs:CY` | Digital Humanities, Ethics, and Social Regulation |
| **Sociophysics** | `physics:physics.soc-ph` | Physics-based social/economic modeling |
| **Computational Linguistics** | `cs:CL` | Language and Computational Philosophy |
| **Cognitive Science** | `q-bio.NC` | Neurons, Cognition, and Emergence |
| **HCI** | `cs:HC` | Human-Computer / Sociotechnical Interaction |
| **Theoretical Econ** | `econ:TH` | Mathematical Economics |
| **Quantitative Finance** | `q-fin:GN` | General Finance / Socio-economic dynamics |

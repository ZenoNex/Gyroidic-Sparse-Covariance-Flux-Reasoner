# Interface Layer

> GUI application and TabbyML code completion client.

---

## 1. Conversational GUI

**Source**: [`src/ui/conversational_gui.py`](../src/ui/conversational_gui.py) (800 lines)

Tkinter-based desktop GUI for managing HF tokens, datasets, ingestion, training, and live chat.

### Components

| Class | Purpose |
|-------|---------|
| `SecureTokenManager` | Token storage: in-memory + optional file persistence at `~/.gyroidic_hf_token` |
| `DatasetAgreementManager` | Tracks required datasets (LMSYS, OASST2, UltraChat) and checks API access |
| `ConversationalGUI` | Main application: 5-tab interface |

### Tab Layout

| Tab | Widgets | Functionality |
|-----|---------|--------------|
| **Setup** | Token entry, show/hide, test, save, clear | HF token management |
| **Datasets** | Dataset list, status check, agreement guide link | Access verification |
| **Ingestion** | Source selector, max samples, progress bar | Data download + processing |
| **Training** | Epoch config, learning rate, start/stop | Async model training |
| **Chat** | Message history, input field, send | Live interaction with engine |

### Threading

All blocking operations (token test, dataset check, ingestion, training, chat) run in background threads with queue-based UI updates via `process_queue()`.

---

## 2. TabbyML Client

**Source**: [`src/integrations/tabby_client.py`](../src/integrations/tabby_client.py) (293 lines)

Connects to a local TabbyML instance using **stdlib urllib only** — zero external dependencies.

### Data Model

| Dataclass | Fields |
|-----------|--------|
| `TabbyConfig` | `host` (localhost), `port` (8080), `timeout` (30s) |
| `TabbyResponse` | `success`, `text`, `model`, `usage`, `raw`, `error` |

### API Methods

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `test_connection()` | `GET /v1/health` | Verify server is running |
| `complete(prompt, language)` | `POST /v1/completions` | Code completion |
| `chat(messages)` | `POST /v1/chat/completions` | Chat-style interaction |
| `generate_training_sample(topic, style)` | `POST /v1/chat/completions` | Synthetic textbook-quality samples |

### Integration

Used by `DiegeticPhysicsEngine` (see [DIEGETIC_ENGINE.md](DIEGETIC_ENGINE.md)) for:
- Code completion suggestions in the web GUI
- Synthetic training data generation via structured prompts
- Chat-based interaction as an alternative to the main engine

<p align="center">
  <img src="./assets/logo.png" alt="Sriti Core Logo" width="220" />
</p>

<h1 align="center">Sriti Core</h1>

<p align="center">
  <strong>Intelligent, Reliability-Aware Model Routing & Cascade Engine for LLMs</strong>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#testing">Testing</a> •
  <a href="#license">License</a>
</p>

---

## Overview

**Sriti Core** is an open-source, ultra-efficient intelligent routing proxy and cascading engine for Large Language Models.

Instead of routing every single request to expensive frontier models (like Claude 3.5 Sonnet or GPT-4o), Sriti dynamically categorizes the request, checks an encrypted semantic cache, and cascades through a hierarchy of models:
1. **Tier 3 (Local / Fast):** Free, local models running on edge devices or on-prem hardware via Ollama (e.g., Qwen 2.5, Llama 3.2, MiniCPM).
2. **Tier 2 (Balanced Cloud):** High-throughput, cost-effective inference providers (e.g., Groq, Fireworks, DeepInfra).
3. **Tier 1 (Frontier):** Capable reasoning models invoked only when task complexity warrants it, or when lower tiers fail verification or latency SLOs.

Sriti continuously learns model reliability, evaluates output quality gates, and guarantees strict latency SLOs while dramatically slashing cloud API bills.

---

## Key Features

- 🎯 **In-Process Task Classifier:** Zero-shot task classification using fast ONNX runtime (`fastembed` BAAI/bge-small-en-v1.5) mapping prompts across categories (e.g., Code, Extraction, Summarization, QA, Math, Reasoning).
- ⚡ **Semantically Aware Caching:** Vector similarity caching backed by Redis/Valkey with task-dependent cosine thresholds. Uncacheable tasks (e.g., non-deterministic reasoning, math) bypass the cache automatically.
- 🔒 **Encrypted Cache at Rest:** Optional AES-GCM (256-bit) authenticated encryption for cached prompt and completion payloads.
- 🗜️ **In-Process Prompt Compression:** LLMLingua-2 integration to compress lengthy contexts before sending them over the wire, with an adaptive quality gate to ensure semantic integrity.
- 📊 **Dynamic Cascade with Quality Gates:** Evaluates lower-tier completions using structural validation (e.g. JSON validation) or semantic quality scoring. If a tier's response fails, it automatically escalates to the next tier.
- 📈 **Online Reliability Profile Learner:** Real-time Exponential Moving Average (EMA) tracking of per-model latency, success rate, and error states without needing external analytical databases.
- 🧠 **Case-Based Memory (KNN Routing):** Remembers past routing decisions and dynamically adjusts candidate selection bias based on historical outcome rewards.
- 🛡️ **Lightweight & Self-Contained:** Uses a native NumPy cosine similarity vector store requiring no proprietary Redis modules (no RediSearch or RedisJSON needed). Works seamlessly on vanilla Redis or Valkey.
- 🔌 **Unified LLM Execution:** Powered by LiteLLM to seamlessly talk to 100+ LLM providers and local inference runtimes.

---

## Architecture

```text
               User / Application Request
                           │
                           ▼
          ┌──────────────────────────────────┐
          │     1. Fast ONNX Classifier      │ ──> Task Category & Embeddings
          └──────────────────────────────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │    2. Semantic Cache (Valkey)    │ ──[Hit]──> Return Cached Response
          └──────────────────────────────────┘
                           │ [Miss]
                           ▼
          ┌──────────────────────────────────┐
          │  3. Model Cascading Engine       │
          │                                  │
          │  ┌────────────────────────────┐  │
          │  │ Tier 3: Local Edge Models  │  │ ──[Success]──> Accept & Learn
          │  └────────────────────────────┘  │
          │                 │ [Fail/Timeout] │
          │  ┌────────────────────────────┐  │
          │  │ Tier 2: Cost-Effective LLM │  │ ──[Success]──> Accept & Learn
          │  └────────────────────────────┘  │
          │                 │ [Fail/Timeout] │
          │  ┌────────────────────────────┐  │
          │  │ Tier 1: Frontier Models    │  │ ──[Guaranteed Execution]
          │  └────────────────────────────┘  │
          └──────────────────────────────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │  4. Update Reliability & Cache   │
          └──────────────────────────────────┘
```

---

## Quickstart

### 1. Installation

Python 3.11+ is required.

**Via pip (recommended):**
```bash
pip install sriti-core
```

**From source:**
```bash
git clone https://github.com/sriti-ai/sriti-core.git
cd sriti-core
pip install -e ".[dev]"
```

### 2. Configure Environment

Create a `.env` file with your credentials and configuration:

```env
SRITI_REDIS_URL=redis://localhost:6379/0
# Optional: Provider API keys (only configure what you use)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
# Optional: AES-256 GCM key for cache encryption (base64-encoded 32 bytes)
# CACHE_ENCRYPTION_KEY=...
```

### 3. Start the Sriti Server

```bash
uvicorn sriti.app:app --host 0.0.0.0 --port 8100
```

### 4. Send a Request via HTTP

```bash
curl -X POST http://localhost:8100/complete \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Extract invoice items and total amount from this ledger snippet...",
    "minimum_tier": "tier_3",
    "cacheable": true,
    "requires_json": true
  }'
```

### 5. Python Client SDK

You can also use the bundled async Python client:

```python
import asyncio
from sriti.client import SritiModelClient

async def main():
    client = SritiModelClient(base_url="http://localhost:8100")
    response = await client.complete(
        prompt="Explain the difference between a mutual fund and an ETF in two sentences.",
        minimum_tier="tier_3",
        cacheable=True
    )
    print(f"Response: {response.text}")
    print(f"Tier Used: {response.tier_used}")
    print(f"Latency: {response.latency_ms}ms")
    print(f"Cost: ${response.cost_usd}")

asyncio.run(main())
```

---

## How It Works

### 1. Task Classification
Prompts are embedded using `fastembed` and compared against predefined semantic task anchors (or an optional lightweight classification head). The classifier runs completely locally in under 10ms.

### 2. Multi-Tier Semantic Cache
- Tasks have configurable similarity thresholds in `policy.yaml`. For example, conversational questions allow a similarity threshold of `0.75`, while structured data extraction requires `0.92`.
- Vectors and encrypted payloads are stored in Valkey/Redis.

### 3. Reliability-Driven Cascading
- For each tier, active candidates in `models.yaml` are evaluated based on their Pareto score: a composite metric factoring in latency, cost, and historical success probability.
- If a lower tier experiences an API error, context window exhaustion, or fails a verification check (e.g. invalid JSON syntax), Sriti escalates seamlessly to the next tier.

---

## Configuration

Sriti's configuration is managed through two clean YAML files in `sriti/config/`:

- **`models.yaml`**: Defines model providers, context windows, cost per token, and tier mappings (`default_tier: 1 | 2 | 3`).
- **`policy.yaml`**: Defines cache TTLs, per-task similarity thresholds, latency SLOs, and quality gate cutoffs.

---

## Testing

Run unit and cascade integration tests using pytest:

```bash
pytest
```

All 39 unit tests run without requiring live LLM API keys or a running Redis instance (mocked in-memory).

---

## License

This project is licensed under the Apache License 2.0.

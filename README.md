# 🏛️ Sovereign AI Engine

A production-grade, self-hosted summarization and question-answering system running an open-weight LLM on private hardware, benchmarked against frontier cloud APIs (Google Gemini 3.8 Flash).

---

## 💡 What This Project Demonstrates to Hiring Managers & Architects

1. **Why fine-tune at all?** Demonstrates domain specialization on complex US Congressional legislative bills (**BillSum**) where generalist LLMs produce generic summaries.
2. **Which method was picked?** Unsloth + Hugging Face PEFT **QLoRA (4-bit NF4 + LoRA rank 16)** to achieve 80% VRAM reduction with <2% loss in ROUGE-L.
3. **What did quantization cost?** Empirically measures ROUGE-1/2/L, P95 tail latency, and VRAM footprints across 16-bit, 8-bit, and 4-bit AWQ/GGUF quantization.
4. **Where the volume crossover sits?** Full financial cost model establishing the exact request volume crossover point (~650,000 calls/month) where private GPU hosting outperforms cloud APIs.
5. **Observability:** Full distributed tracing and evaluation via **Arize Phoenix**.

---

## 🛠️ Architecture

```
                 [ Google Colab / Private GPU ]
                               │
               Fine-Tuning (Unsloth + QLoRA)
               Dataset: BillSum (Hugging Face)
                               │
                       [ LoRA Adapter ]
                               ▼
        ┌──────────────────────────────────────────────┐
        │        Sovereign AI Engine Deployment        │
        ├──────────────────────────────────────────────┤
        │  Inference: vLLM (Continuous Batching / AWQ) │
        │  API Gateway: FastAPI (Port 9000)            │
        │  Observability: Arize Phoenix (Port 6006)    │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │              Benchmark Harness               │
        ├──────────────────────────────────────────────┤
        │  Evaluates Sovereign vs Gemini 3.8 Flash     │
        │  Metrics: ROUGE-1/2/L, P50, P95, Throughput  │
        │  Financial Cost Crossover Analysis           │
        └──────────────────────────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Fine-Tune in Google Colab (Free GPU)
Open `finetune/sovereign_ai_engine_colab.ipynb` directly in [Google Colab](https://colab.research.google.com).
- Uses free 15GB T4 GPU.
- Fine-tunes Qwen 2.5 7B or Llama 3.2 3B in ~15 minutes.
- Exports adapter and GGUF.

### 2. Run Locally or in Docker
```powershell
# Copy environment configuration
copy .env.example .env

# Run via Docker Compose
docker compose up -d

# Or run locally with Python 3.12
pip install -r requirements.txt
python -m api.main
```

### 3. Endpoints

- **FastAPI Documentation:** `http://localhost:9000/docs`
- **Health Check:** `http://localhost:9000/health`
- **Arize Phoenix Dashboard:** `http://localhost:6006`

#### Test Summarization:
```bash
curl -X POST http://localhost:9000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "SECTION 1. SHORT TITLE. This Act may be cited as the Modern Clean Energy Act...", "provider": "sovereign"}'
```

### 4. Run Benchmark Harness
```powershell
python -m benchmark.run_benchmark
```
Outputs complete ROUGE-L, latency percentiles, and cost crossover table to `benchmark_results.md`.

# ☁️ Azure AI & Azure AI Foundry — Engineering Reference

> **A practical cheat sheet for AI/ML Engineers building real-world systems on Azure.**  
> Covers Azure AI Services, Azure AI Foundry, RAG pipelines, model deployment, NLP integrations, and production patterns.

---

## 📋 Table of Contents

1. [Azure AI Services — Quick Map](#1-azure-ai-services--quick-map)
2. [Azure AI Foundry — Core Concepts](#2-azure-ai-foundry--core-concepts)
3. [Model Deployment Patterns](#3-model-deployment-patterns)
4. [RAG System Architecture on Azure](#4-rag-system-architecture-on-azure)
5. [NLP Applications with Azure AI](#5-nlp-applications-with-azure-ai)
6. [Azure AI Search (Cognitive Search)](#6-azure-ai-search-cognitive-search)
7. [Prompt Flow — Engineering Workflows](#7-prompt-flow--engineering-workflows)
8. [Integrating AI Services into Applications](#8-integrating-ai-services-into-applications)
9. [Security, Monitoring & Production Readiness](#9-security-monitoring--production-readiness)
10. [Key SDK & CLI References](#10-key-sdk--cli-references)
11. [Real-World Use Cases](#11-real-world-use-cases)

---

## 1. Azure AI Services — Quick Map

> Formerly "Azure Cognitive Services" — a family of pre-built AI APIs requiring zero ML training.

| Service                   | What It Does                                  | When to Use                               |
|---------------------------|-----------------------------------------------|-------------------------------------------|
| **Azure OpenAI**          | GPT-4, GPT-4o, Embeddings, DALL-E via API     | LLM-powered apps, completions, chat, RAG  |
| **Language Service**      | NER, sentiment, summarization, key phrases    | NLP pipelines, document processing        |
| **Document Intelligence** | OCR, form/invoice extraction, layout parsing  | Structured data from PDFs, scans          |
| **Speech Service**        | STT, TTS, speaker recognition, translation    | Voice apps, transcription pipelines       |
| **Vision Service**        | Image classification, OCR, object detection   | Visual inspection, multimodal pipelines   |
| **Translator**            | Text translation (100+ languages)             | Multilingual content pipelines            |
| **Content Safety**        | Detect harmful/offensive content              | Content moderation in LLM output chains   |

**Key concept:** All services are consumed via **REST API** or **Azure SDK** (Python, .NET, JS). Authentication uses **API keys** or **Azure Managed Identity** (preferred in production).

---

## 2. Azure AI Foundry — Core Concepts

> Azure AI Foundry (formerly Azure Machine Learning Studio + AI Studio unified) is the **end-to-end platform** for building, evaluating, and deploying AI systems.

### Core Building Blocks

```
Azure AI Foundry
├── Hub               → Shared resource container (network, storage, keys)
│   └── Project       → Isolated workspace per team/use case
│       ├── Models    → Catalog: OpenAI, Meta Llama, Mistral, Phi, etc.
│       ├── Deployments → Managed online/batch endpoints
│       ├── Prompt Flow → Visual + code LLM pipeline builder
│       ├── Evaluations → Automated quality assessment of LLM outputs
│       └── Indexes   → Vector stores connected to Azure AI Search
```

### Hub vs. Project

|                | Hub                                              | Project                                  |
|----------------|--------------------------------------------------|------------------------------------------|
| **Scope**      | Organization-level                               | Team/use-case-level                      |
| **Contains**   | Shared compute, networking, storage, connections | Flows, deployments, evaluations, indexes |
| **Analogy**    | Azure subscription                               | Resource group                           |

### Model Catalog — Key Models Available

- **OpenAI family:** `gpt-4o`, `gpt-4`, `gpt-35-turbo`, `text-embedding-ada-002`, `text-embedding-3-large`
- **Open-source (serverless/managed):** `Meta-Llama-3.1`, `Mistral-Large`, `Phi-3-mini`, `Cohere-Command-R`
- **Multimodal:** `gpt-4o` (text + vision), `Phi-3-vision`

---

## 3. Model Deployment Patterns

### Deployment Types

| Type                        | Use Case                           | Billing          |
|-----------------------------|------------------------------------|------------------|
| **Managed Online Endpoint** | Real-time inference, REST API      | Per compute hour |
| **Serverless API (MaaS)**   | Pay-per-token, no infra management | Per 1K tokens    |
| **Batch Endpoint**          | Large-scale async processing       | Per compute hour |
| **Local / Edge**            | ONNX or Phi models on-device       | N/A              |

### Deploying a Model via Azure AI Foundry (Python SDK)

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="<SUB_ID>",
    resource_group_name="<RG>",
    workspace_name="<PROJECT_NAME>"
)

# 1. Create endpoint
endpoint = ManagedOnlineEndpoint(name="my-model-endpoint", auth_mode="key")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# 2. Create deployment
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="my-model-endpoint",
    model=Model(path="./model"),
    instance_type="Standard_DS3_v2",
    instance_count=1
)
ml_client.online_deployments.begin_create_or_update(deployment).result()

# 3. Route 100% traffic to deployment
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
```

### Deploying Azure OpenAI Models

```python
# Azure OpenAI deployments are managed in the Azure OpenAI resource
# Once deployed, call via:
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="<AZURE_OPENAI_KEY>",
    azure_endpoint="https://<RESOURCE>.openai.azure.com/",
    api_version="2024-02-01"
)

response = client.chat.completions.create(
    model="gpt-4o",          # deployment name, not model name
    messages=[{"role": "user", "content": "Summarize this document."}]
)
```

### Blue/Green Deployment (Traffic Splitting)

```python
# Split traffic: 90% to stable (blue), 10% to new (green)
endpoint.traffic = {"blue": 90, "green": 10}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
```

---

## 4. RAG System Architecture on Azure

> **Retrieval-Augmented Generation (RAG):** Ground LLM responses in your own data using vector search + generation.

### Full Azure RAG Stack

```
┌─────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                  │
│                                                         │
│  Documents (PDF/DOCX/TXT)                               │
│       │                                                 │
│       ▼                                                 │
│  Document Intelligence  ──►  Chunking Strategy          │
│  (OCR, layout parsing)        (fixed / semantic)        │
│       │                                                 │
│       ▼                                                 │
│  Embedding Model (text-embedding-3-large)               │
│       │                                                 │
│       ▼                                                 │
│  Azure AI Search  ──►  Vector Index + Keyword Index     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                      │
│                                                         │
│  User Query                                             │
│       │                                                 │
│       ▼                                                 │
│  Embed Query  ──►  Hybrid Search (vector + BM25)        │
│                         │                               │
│                         ▼                               │
│               Reranker (semantic ranker)                │
│                         │                               │
│                         ▼                               │
│               Top-K Chunks  ──►  GPT-4o (generation)    │
│                                        │                │
│                                        ▼                │
│                                  Grounded Answer        │
└─────────────────────────────────────────────────────────┘
```

### Implementation (Python)

```python
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential

# Clients
search_client = SearchClient(
    endpoint="https://<SEARCH>.search.windows.net",
    index_name="my-index",
    credential=AzureKeyCredential("<SEARCH_KEY>")
)
openai_client = AzureOpenAI(
    api_key="<OPENAI_KEY>",
    azure_endpoint="https://<RESOURCE>.openai.azure.com/",
    api_version="2024-02-01"
)

def embed_query(query: str) -> list[float]:
    return openai_client.embeddings.create(
        model="text-embedding-3-large", input=query
    ).data[0].embedding

def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    vector_query = VectorizedQuery(
        vector=embed_query(query),
        k_nearest_neighbors=top_k,
        fields="content_vector"
    )
    results = search_client.search(
        search_text=query,                 # BM25 keyword search
        vector_queries=[vector_query],     # vector search
        query_type="semantic",             # semantic re-ranking
        semantic_configuration_name="default",
        top=top_k
    )
    return [{"content": r["content"], "source": r["source"]} for r in results]

def rag_query(user_query: str) -> str:
    chunks = hybrid_search(user_query)
    context = "\n\n".join([c["content"] for c in chunks])

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Answer using only the context below:\n{context}"},
            {"role": "user",   "content": user_query}
        ]
    )
    return response.choices[0].message.content
```

### Chunking Strategies

| Strategy                          | Best For                                      |
|-----------------------------------|-----------------------------------------------|
| Fixed-size (512 tokens + overlap) | General documents                             |
| Sentence/paragraph boundary       | Narrative text, articles                      |
| Section-based (by heading)        | Technical docs, reports                       |
| Hierarchical (parent-child)       | Long documents with multi-granularity queries |

---

## 5. NLP Applications with Azure AI

### Azure Language Service — Key Tasks

```python
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

client = TextAnalyticsClient(
    endpoint="https://<RESOURCE>.cognitiveservices.azure.com/",
    credential=AzureKeyCredential("<KEY>")
)

documents = ["The aircraft engine overheated during departure from Cairo International Airport."]

# Named Entity Recognition
ner_result = client.recognize_entities(documents)[0]
for entity in ner_result.entities:
    print(f"{entity.text} → {entity.category} ({entity.confidence_score:.2f})")

# Sentiment Analysis
sentiment = client.analyze_sentiment(documents)[0]
print(sentiment.sentiment, sentiment.confidence_scores)

# Extractive Summarization
from azure.ai.textanalytics import ExtractiveSummaryAction
poller = client.begin_analyze_actions(
    documents,
    actions=[ExtractiveSummaryAction(max_sentence_count=3)]
)
for result in poller.result():
    for summary in result:
        print([s.text for s in summary.sentences])
```

### Custom NLP Models (Language Studio)

| Feature                                         | Use Case                                                     |
|-------------------------------------------------|--------------------------------------------------------------|
| **Custom NER**                                  | Domain-specific entity extraction (aviation, legal, medical) |
| **Custom Text Classification**                  | Route support tickets, classify documents                    |
| **Custom Question Answering**                   | FAQ-style Q&A from custom knowledge base                     |
| **Conversational Language Understanding (CLU)** | Intent detection + entity extraction for chatbots            |

**Workflow:** Label data in Language Studio → Train → Evaluate → Deploy → Call via API  
**Training data minimum:** ~50 examples per entity/intent for acceptable performance.

---

## 6. Azure AI Search (Cognitive Search)

> The backbone of RAG systems and enterprise search on Azure.

### Index Schema Design (Vector + Keyword)

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SemanticConfiguration, SemanticSearch, SemanticPrioritizedFields, SemanticField
)

index = SearchIndex(
    name="docs-index",
    fields=[
        SearchField(name="id",             type=SearchFieldDataType.String, key=True),
        SearchField(name="content",        type=SearchFieldDataType.String, searchable=True),
        SearchField(name="source",         type=SearchFieldDataType.String, filterable=True),
        SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=3072,
                    vector_search_profile_name="hnsw-profile"),
    ],
    vector_search=VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw", parameters={"m": 4, "efConstruction": 400})],
        profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw")]
    ),
    semantic_search=SemanticSearch(configurations=[
        SemanticConfiguration(
            name="default",
            prioritized_fields=SemanticPrioritizedFields(content_fields=[SemanticField(field_name="content")])
        )
    ])
)
```

### Search Modes Comparison

| Mode                  | Description                        | Best For                       |
|-----------------------|------------------------------------|--------------------------------|
| **Keyword (BM25)**    | Traditional full-text search       | Exact terms, IDs, codes        |
| **Vector**            | Semantic similarity via embeddings | Conceptual/paraphrase queries  |
| **Hybrid**            | BM25 + vector combined             | Most production RAG systems    |
| **Semantic Reranker** | ML model re-scores hybrid results  | Highest accuracy, adds latency |

### Indexer Pipeline (Auto-ingest from Storage)

```
Azure Blob Storage → Indexer → Skillset (OCR, split, embed) → Search Index
```
- **Skillsets** apply AI enrichment: OCR, language detection, entity extraction, custom embedding.
- **Incremental enrichment:** Only re-processes changed documents.

---

## 7. Prompt Flow — Engineering Workflows

> Prompt Flow is a **DAG-based pipeline builder** for LLM applications inside Azure AI Foundry.

### Flow Types

| Type                | Use Case                                         |
|---------------------|--------------------------------------------------|
| **Standard Flow**   | Single-pass: input → LLM → output                |
| **Chat Flow**       | Multi-turn with conversation history             |
| **Evaluation Flow** | Score outputs (relevance, groundedness, fluency) |

### Standard RAG Flow Structure

```
[Inputs] → [embed_query] → [vector_search] → [build_prompt] → [llm_call] → [Outputs]
```

Each node is either a **Python tool**, **LLM tool** (with Jinja2 prompt template), or **Prompt tool**.

### Running Evaluations

```python
# Evaluate RAG output quality using built-in metrics
# Metrics: groundedness, relevance, coherence, fluency, similarity
# Run via Azure AI Foundry UI or SDK

from azure.ai.evaluation import evaluate, GroundednessEvaluator, RelevanceEvaluator

groundedness_eval = GroundednessEvaluator(model_config={
    "azure_endpoint": "https://<RESOURCE>.openai.azure.com/",
    "api_key": "<KEY>",
    "azure_deployment": "gpt-4o"
})

result = evaluate(
    data="eval_dataset.jsonl",          # {query, context, response}
    evaluators={"groundedness": groundedness_eval},
    output_path="eval_results.json"
)
```

---

## 8. Integrating AI Services into Applications

### Connection Patterns

```python
# RECOMMENDED: Use Managed Identity (no keys in code)
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.ai.textanalytics import TextAnalyticsClient

credential = DefaultAzureCredential()   # works locally + in Azure
client = TextAnalyticsClient(endpoint="https://<RESOURCE>.cognitiveservices.azure.com/", 
                             credential=credential)
```

### Azure AI Foundry SDK (Unified Client)

```python
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

project = AIProjectClient.from_connection_string(
    conn_str="<PROJECT_CONNECTION_STRING>",
    credential=DefaultAzureCredential()
)

# Get an Azure OpenAI client scoped to your project
openai_client = project.inference.get_azure_openai_client(api_version="2024-02-01")

# Get connections (AI Search, Storage, etc.)
search_connection = project.connections.get("<CONNECTION_NAME>")
```

### Document Intelligence Integration

```python
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

doc_client = DocumentIntelligenceClient(
    endpoint="https://<RESOURCE>.cognitiveservices.azure.com/",
    credential=AzureKeyCredential("<KEY>")
)

# Analyze a PDF — extract layout, tables, key-value pairs
with open("report.pdf", "rb") as f:
    poller = doc_client.begin_analyze_document("prebuilt-layout", body=f)
result = poller.result()

for table in result.tables:
    for cell in table.cells:
        print(f"[{cell.row_index},{cell.column_index}] {cell.content}")
```

### Streaming LLM Responses

```python
stream = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 9. Security, Monitoring & Production Readiness

### Authentication — Priority Order

```
1. Managed Identity (Azure-to-Azure, no keys) ← ALWAYS prefer this
2. Azure Key Vault reference (fetch secret at runtime)
3. Environment variable / app config
4. Hardcoded key                               ← NEVER in production
```

### Key Vault Integration

```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

kv_client = SecretClient(
    vault_url="https://<VAULT>.vault.azure.net/",
    credential=DefaultAzureCredential()
)
api_key = kv_client.get_secret("openai-api-key").value
```

### Monitoring

| Tool                             | What to Monitor                                    |
|----------------------------------|----------------------------------------------------|
| **Azure Monitor / App Insights** | Latency, error rates, throughput per endpoint      |
| **Prompt Flow traces**           | Per-node execution time, token usage               |
| **Azure OpenAI metrics**         | TPM (tokens/min), RPM (requests/min), quota usage  |
| **Managed Endpoint logs**        | Inference latency P50/P95, failed requests         |

```python
# Enable App Insights tracing in Prompt Flow
import promptflow
promptflow.tracing.start_trace(collection="my-rag-app")
```

### Content Filtering (Azure OpenAI)

- Default filters: hate, violence, self-harm, sexual content (severity 0–6)
- Configure per-deployment in Azure OpenAI Studio
- Add **Azure AI Content Safety** for additional screening (pre/post LLM call)

### Cost Optimization

- Use **Provisioned Throughput Units (PTU)** for predictable high-volume workloads
- Use **Pay-As-You-Go** for bursty or low-volume workloads
- Cache frequent queries with **Azure Cache for Redis** before hitting the LLM
- Use `text-embedding-3-small` over `large` when retrieval accuracy difference is negligible (~10% cost)

---

## 10. Key SDK & CLI References

### Essential Python Packages

```bash
pip install \
  azure-ai-projects \           # AI Foundry unified SDK
  azure-ai-ml \                 # Azure ML / model deployment
  openai \                      # Azure OpenAI client
  azure-search-documents \      # AI Search (vector + keyword)
  azure-ai-textanalytics \      # Language Service
  azure-ai-documentintelligence \ # Document Intelligence
  azure-identity \              # Auth (Managed Identity, DefaultAzureCredential)
  azure-keyvault-secrets \      # Key Vault
  promptflow[azure]             # Prompt Flow SDK
```

### Azure CLI — Useful Commands

```bash
# Login & set subscription
az login
az account set --subscription "<SUB_ID>"

# Create AI Foundry Hub and Project
az ml workspace create --kind hub -n my-hub -g my-rg
az ml workspace create --kind project -n my-project -g my-rg --hub-id <HUB_ID>

# List deployments
az ml online-endpoint list -w <WORKSPACE> -g <RG>

# Invoke an endpoint
az ml online-endpoint invoke -n my-endpoint -r '{"query": "test"}' -w <WS> -g <RG>

# Monitor endpoint logs
az ml online-deployment get-logs -n blue --endpoint-name my-endpoint -w <WS> -g <RG>
```

---

## 11. Real-World Use Cases

### ✈️ Aviation Document RAG System
**Stack:** Document Intelligence → Azure AI Search (hybrid) → GPT-4o → Prompt Flow  
**Flow:** Ingest AMM/FCOM PDFs with OCR → Chunk by section → Embed with `text-embedding-3-large` → Hybrid search → Grounded answer generation with source citation  
**Key config:** Semantic ranker enabled, `top=5`, `query_type="semantic"`, system prompt enforces answer grounding

---

### 📄 Intelligent Document Processing Pipeline
**Stack:** Document Intelligence → Language Service → Azure Functions → Cosmos DB  
**Flow:** Upload invoice/form → Extract key-value pairs → Run NER for vendor/date/amount → Validate → Store structured output  
**Key config:** `prebuilt-invoice` model, confidence threshold filtering, retry logic for low-confidence fields

---

### 🤖 Domain-Specific Chatbot with CLU
**Stack:** CLU (Language Studio) → Azure Bot Framework → Azure OpenAI → CosmosDB (conversation history)  
**Flow:** User utterance → CLU detects intent + entities → Route to handler → Optionally call GPT-4o for complex responses → Return reply + persist history  
**Key config:** CLU confidence threshold > 0.8 for direct handling, fallback to GPT for low-confidence

---

### 📊 Batch NLP Analytics on Text Data
**Stack:** Azure ML Batch Endpoint → Language Service → Azure Data Lake → Synapse  
**Flow:** CSV of support tickets → Batch sentiment + NER enrichment → Write enriched Parquet to Data Lake → Analyze in Synapse Analytics  
**Key config:** Batch endpoint with autoscaling, `RecognizeEntitiesAction` + `AnalyzeSentimentAction` in a single `begin_analyze_actions` call (batching reduces API calls 10×)

---

### 🔍 Semantic Search Application
**Stack:** Azure AI Search → `text-embedding-3-large` → FastAPI → Azure Container Apps  
**Flow:** User query → Embed → Vector search with pre-filters (category, date) → Return ranked results with highlights  
**Key config:** `filter="category eq 'technical'"` + `VectorizedQuery` + `select` field projection for fast response payloads

---

## ⚡ Quick-Reference Cheatsheet

```
DEPLOYMENT
  Real-time inference       → ManagedOnlineEndpoint
  High-volume / cheap       → Serverless API (MaaS / pay-per-token)
  Large async jobs          → BatchEndpoint

RAG
  Embedding model           → text-embedding-3-large (3072d) or -small (1536d)
  Search mode               → Hybrid + SemanticRanker for best accuracy
  Chunk size                → 512 tokens / 10% overlap (start here)
  Grounding enforcement     → System prompt: "only answer from context"

SECURITY
  Auth                      → DefaultAzureCredential everywhere
  Secrets                   → Azure Key Vault, never hardcoded
  Content safety            → Enable default filters + Content Safety API

MONITORING
  LLM metrics               → Token usage, latency P95, error rate
  Eval metrics              → Groundedness, relevance, coherence (AI-assisted eval)

COST
  High volume stable        → Provisioned (PTU)
  Variable / dev            → Pay-As-You-Go
  Reduce embedding cost     → Cache embeddings, use -small model when sufficient
```

---

> **Maintained by:** Abdelurahman  
> **Focus:** AI/ML Engineering · NLP · RAG Systems  

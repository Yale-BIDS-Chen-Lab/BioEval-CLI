# BioEval: A Framework for End-to-End Biomedical NLP Workflows

## What is BioEval?

BioEval is a toolkit for benchmarking large language models across biomedical NLP tasks. It offers a modular end-to-end pipeline—from prompt generation and few-shot selection to inference, postprocessing, and evaluation—grounded in rigorous methods.

## Key Features

- **Universal LLM Benchmarking** - Effortlessly evaluate any LLM across multiple biomedical tasks with consistent, automated workflows.
- **Fully Configurable & Reproducible**  - Every stage of the pipeline—from prompt generation and inference settings to output postprocessing and evaluation metrics—is fully customizable, transparent, and designed for reproducible experimentation.
- **Built-in Statistical Analysis** - Confidence intervals and significance testing via bootstrapping enable rigorous, side-by-side model comparisons.
- **5 Biomedical NLP Tasks** - Named Entity Recognition, Single Label Classification, Multi-Label Classification, Multiple Choice QA, and Text Generation.
- **Compatible with All Major LLM Providers** - Benchmark both open-source and commercial LLMs from Hugging Face and Azure OpenAI, along with locally hosted models.
---

## Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/Yale-BIDS-Chen-Lab/BioEval-CLI.git
cd BioEval-CLI

# Create virtual environment
python3 -m venv bioeval-env

# Install dependencies
pip install -r requirements.txt
```

### Run Your First Experiment

```bash
# Quick test with ncbi_disease dataset
python3 main.py --config config/model_cards/ner/benchmark/ncbi_disease.yaml
```

Results will be saved in the `data/outputs/` directory.

### Pipeline Stages

Run specific stages independently: `python3 main.py --config config/model_cards/ner/benchmark/ncbi_disease.yaml` with one of the flag sets below:

| Flags | Prompt&nbsp;Gen | Inference | Evaluation |
|-------|:--------------:|:---------:|:----------:|
| *(none)* | ✅ | ✅ | ✅ |
| `--do-prompt-generation` | ✅ | ❌ | ❌ |
| `--do-prompt-generation --do-inference` | ✅ | ✅ | ❌ |
| `--do-inference --do-evaluation` | ❌ | ✅ | ✅ |
| `--do-evaluation` | ❌ | ❌ | ✅ |

---

## Model Cards Organization

Model configurations are organized by task type for easy navigation:

```
config/model_cards/
├── ner/
│   ├── benchmark/          # Named Entity Recognition benchmarks
│   └── custom/            # Custom dataset configurations
├── mcq/
│   ├── benchmark/          # Multiple Choice Questions benchmarks
│   └── custom/            # Custom dataset configurations
├── mlc/
│   ├── benchmark/          # Multi-Label Classification benchmarks
│   └── custom/            # Custom dataset configurations
├── slc/
│   ├── benchmark/          # Single Label Classification benchmarks
│   └── custom/            # Custom dataset configurations
└── generation/
    ├── benchmark/          # Text Generation benchmarks
    └── custom/            # Custom dataset configurations
```

**Quick access to configurations:**
- **Benchmark datasets**: `config/model_cards/[task]/benchmark/[dataset].yaml`
- **Custom configurations**: `config/model_cards/[task]/custom/[task]_custom.yaml`

## Default Tasks & Datasets

| Task | Datasets | Key Metrics |
|------|----------|-------------|
| **NER** | ncbi_disease, bc5cdr_chemical | Exact Match Precision, Recall, F1 |
| **SLC** | chemprot, ddi | Macro F1, Weighted F1 |
| **MLC** | litcovid, hoc | Macro F1, Weighted F1 |
| **MCQ** | medqa, pubmedqa | Accuracy, Macro F1, Weighted F1 |
| **Generation** | ms2, pubmed_summarization, cochrane, plos | ROUGE-1/2/L, BERTScore, BARTScore, METEOR |

## Custom Tasks & Datasets

1. **Prepare your data** in JSON format:
```json
[
  {
    "id": "001", 
    "input": "Patient presents with...",
    "reference": "Expected output..."
  }
]
```

2. **Create a configuration file** by customizing one of the provided templates:

```bash
# Copy the appropriate template for your task
cp config/model_cards/ner/custom/ner_custom.yaml ner_custom.yaml

# Available templates for each task:
# - config/model_cards/ner/custom/ner_custom.yaml           (Named Entity Recognition)
# - config/model_cards/mcq/custom/mcq_custom.yaml           (Multiple Choice Questions)
# - config/model_cards/mlc/custom/mlc_custom.yaml                      (Multi-Label Classification)
# - config/model_cards/slc/custom/slc_custom.yaml           (Single Label Classification)
# - config/model_cards/generation/custom/generation_custom.yaml          (Text Generation)
```

**Example: Custom MCQ Configuration**
```yaml
# ------------------------------
# Task and Dataset Settings
# ------------------------------
task: mcq                                  # Task type: generation | mcq | ner | slc | mlc
dataset: medmcqa                           # Dataset name

data_files:
  test_file: "datasets/medmcqa_test.json"   # Path to the test set
  train_file: "datasets/medmcqa_train.json" # Optional: used for few-shot examples


# ------------------------------
# Prompt Generation Settings
# ------------------------------
prompt:
  system_prompt: "prompts/medmcqa_prompt.txt"  # Path to the prompt template. Use placeholders such as {example_0}, {example_1}, ... and {{input}} in the template
  shot_indices: 0, 10, 20                      # Indices of few-shot examples (omit for zero-shot)
  output_file: null                            # Auto-named as medmcqa_[hash].json; shared across pipeline stages


# ------------------------------
# Model Inference Settings
# ------------------------------
model:
  provider: hf                                # Model provider: hf | vllm | azure | google | anthropic | local
  name: "meta-llama/Llama-3.2-3B-Instruct"    # Model identifier

inference:
  temperature: 0.1                            # Sampling temperature
  max_new_tokens: 100                         # Maximum number of generated tokens
  # ... other inference parameters, provider-specific
  output_file: null                           # Will be auto-named as medmcqa_[hash].json; shared across pipeline stages

# ------------------------------
# Evaluation Settings
# ------------------------------
evaluation:
  postprocessing: process_mcq_custom          # Function to parse MCQ model outputs. 
                                              # Other functions: load_normalized_data | load_raw_data | process_ner_token_indices | process_mlc_custom
  labels: A, B, C, D                          # The parser extracts the first valid option from the model output or returns "missing" if no valid option is found
                                              # A, B, C | A, B, C, D, E | Yes, No, Maybe
                                              # For ner or mlc, specify class labels as comma-separated list (e.g., mechanism, transmission)
  output_file: null                           # Auto-named as medmcqa_[hash].json; shared across pipeline stages
  metrics:
    - accuracy
    - macro_f1
    - weighted_f1
```

---

## Provider-Specific Configuration

### HuggingFace
```yaml
model:
  provider: hf
  name: "meta-llama/Llama-3.2-3B-Instruct"
inference:
  temperature: 0.7
  max_new_tokens: 50
```

### vLLM
```yaml
model:
  provider: vllm  
  name: "meta-llama/Llama-3.2-3B-Instruct"
inference:
  temperature: 0.7
  max_tokens: 50
```

### Azure OpenAI
```yaml
model:
  provider: azure
  name: "gpt-4o"
inference:
  endpoint: "https://your-resource.openai.azure.com/"
  api_version: "2025-01-01-preview"
  api_key: "your-api-key"
  temperature: 0.7
  max_tokens: 512
```

For **reasoning models** — the `o1`/`o3`/`o4` and `gpt-5` families (e.g. `gpt-5.4`) — sampling parameters (`temperature`, `top_p`, penalties) are ignored; use `max_completion_tokens` and, optionally, `reasoning_effort`:
```yaml
model:
  provider: azure
  name: "gpt-5"
inference:
  endpoint: "https://your-resource.openai.azure.com/"
  api_version: "2025-01-01-preview"
  api_key: "your-api-key"
  max_completion_tokens: 4096
  reasoning_effort: medium        # model-specific, e.g. minimal | low | medium | high
```

### Google Gemini
```yaml
model:
  provider: google
  name: "gemini-2.5-pro"
inference:
  api_key: "your-api-key"          # or set GOOGLE_API_KEY / GEMINI_API_KEY
  temperature: 0.7
  top_p: 0.95
  max_tokens: 4096
```

### Anthropic (Claude)
```yaml
model:
  provider: anthropic
  name: "claude-sonnet-4-5"
inference:
  api_key: "your-api-key"          # or set ANTHROPIC_API_KEY
  temperature: 0.7
  max_tokens: 4096
```

---

## Statistical Analysis

**Rigorous model comparison with significance testing**

BioEval compares models with bootstrap confidence intervals and a significance test. The bootstrap 95% CI is computed by resampling the **per-example** metric values and taking their mean; the significance test runs on the **raw per-example paired scores** — the paired Wilcoxon signed-rank test (default) or the unpaired Wilcoxon rank-sum test. No random seed is set, so results reflect genuine sampling variability.

Only **per-example** metrics can be compared (`rouge1`/`rouge2`/`rougeL`, `bertscore`, `bartscore`, `meteor`, `accuracy`); corpus-level metrics such as `macro_f1` or `exact_match_f1` have no per-example scores.

**Example Usage:**
```bash
# Paired Wilcoxon signed-rank (default) on ROUGE-L for the PLOS dataset
python3 scripts/compare_models.py \
  --metric rougeL \
  --dataset plos \
  --input_directory data/outputs \
  --output_file model_comparison.csv

# Unpaired Wilcoxon rank-sum on accuracy
python3 scripts/compare_models.py \
  --metric accuracy --dataset medqa --test_method rank-sum \
  --input_directory data/outputs --output_file medqa_comparison.csv
```

`--test_method` is `signed-rank` (paired, default) or `rank-sum` (unpaired). `--n_samples` (bootstrap iterations, default 1000) and `--sample_size` (default 40, clamped to the number of examples) are optional.

**Sample Output:**
| Model 1 | Model 2 | Mean 1 | Std 1 | 95% CI 1 | Mean 2 | Std 2 | 95% CI 2 | p-value | test_method |
|:--|:--|--:|--:|:--|--:|--:|:--|--:|:--|
| model_a_1a2b3c4d | model_b_5e6f7a8b | 0.3173 | 0.008 | [0.302, 0.332] | 0.2872 | 0.0073 | [0.272, 0.3] | 8.83e-05 | signed-rank |

---

## Experiment Search & Retrieval

BioEval includes a powerful search system that allows you to quickly find and analyze experiment results without manually browsing through configuration files and output directories.

**Quick Examples:**

**1. List all available experiments:**
```bash
$ python3 scripts/query_experiments.py --list-available

Available Experiments
============================================================
Config directory: /path/to/BioEval-CLI/config/model_cards
Results directory: /path/to/BioEval-CLI/data/outputs

Datasets (12):
   • bc5cdr_chemical
   • chemprot
   • cochrane
   • ddi
   • hoc
   • litcovid
   • medqa
   • ms2
   • ncbi_disease
   • plos
   • pubmed_summarization
   • pubmedqa

Tasks (5):
   • mlc
   • generation
   • mcq
   • ner
   • slc

Models (2):
   • meta-llama/Llama-3.2-1B-Instruct
   • meta-llama/Llama-3.2-3B-Instruct
```

**2. Search by dataset:**
```bash
$ python3 scripts/query_experiments.py --dataset medmcqa

🔍 Searching for experiments with dataset='medmcqa'
============================================================
📁 Config directory: /path/to/BioEval-CLI/config/model_cards
📁 Results directory: /path/to/BioEval-CLI/data/outputs

📋 Found 1 Configuration(s):
----------------------------------------
📁 /path/to/BioEval-CLI/config/model_cards/mcq/benchmark/medmcqa.yaml
   Task: mcq
   Dataset: medmcqa
   Model: meta-llama/Llama-3.2-3B-Instruct
   Provider: hf

📊 Found 1 Result(s):
----------------------------------------
📁 /path/to/BioEval-CLI/data/outputs/medmcqa_1fa215ad.json
   Task: mcq
   Dataset: medmcqa
   Model: meta-llama/Llama-3.2-3B-Instruct
   Provider: huggingface
   Timestamp: 2025-11-09T16:47:20.199760
   📈 Metrics:
      accuracy: 0.0000
      macro_f1: 0.0000
      weighted_f1: 0.0000

✅ Summary: 1 configuration(s), 1 result file(s).
```

**Additional commands:**
```bash
# Search by model or task
python3 scripts/query_experiments.py --model meta-llama/Llama-3.2-3B-Instruct
python3 scripts/query_experiments.py --task ner

# Search with multiple filters
python3 scripts/query_experiments.py --task mcq --dataset medqa
```

---

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 The BioEval authors.

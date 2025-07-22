# Construction-Disclosure-Documents-Reviewing-System
This is the official repo for our paper: "Generative Knowledge-Guided Review System for Construction Disclosure Documents Reviewing"

# Data
- Relevant data files could be acquired in `./data/`.
- Relevant data process tools are in `./data/process.ipynb`

# Weights
The pretrained weights can be acquired at [google_drive]().

# Train
You can train the extraction modules in this commend:
```bash
# Train with default parameters
python train_extract.py -i dataset.csv

# Custom output file and training parameters
python train_extract.py -i dataset.csv -o my_model.pth -e 100 -l 1e-5 -b 32

# Use different BERT model
python train_extract.py -i dataset.csv --model_name bert-base-multilingual-cased
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input` | `-i` | Required | Input CSV file path |
| `--output` | `-o` | `pretrain-model.pth` | Output model file path |
| `--epochs` | `-e` | `200` | Number of training epochs |
| `--learning_rate` | `-l` | `5e-6` | Learning rate |
| `--batch_size` | `-b` | `16` | Batch size |
| `--split_ratio` | `-s` | `0.9` | Train/validation split ratio |
| `--max_length` | `-m` | `512` | Maximum sequence length |
| `--weight_decay` | `-w` | `0.01` | Weight decay |
| `--warmup_steps` | | `0` | Number of warmup steps |
| `--print_interval` | | `20` | Print F1 score interval |
| `--model_name` | | `bert-base-chinese` | BERT model name |

## Dataset Format

The input CSV file should contain the following columns:

| Column | Description | Required | Example |
|--------|-------------|----------|---------|
| `Query` | The input text/query to be processed | ✅ | "How to train a machine learning model?" |
| `max` | Maximum priority chunk/span to extract | ✅ | "machine learning model" |
| `mid` | Medium priority chunk/span to extract | ❌ | "train" |
| `lit` | Low priority chunk/span to extract | ❌ | "How to" |

### Sample CSV Structure
```csv
Query,max,mid,lit
"How to train a machine learning model?","machine learning model","train","How to"
"What is deep learning?","deep learning",,
"Explain neural networks","neural networks","Explain",
```


 # Inference
Retrieval inference example.

```python
from CDDRS import GKGR

source_knowledge_base = 'path_to_knowledge_base'
query = 'your_retrieval_query'
retrieval_result = GKGR(
    query, 
    source_knowledge_base, 
    topk=3, 
    llm='deepseek', 
    api='your_own_deepseek_api', 
    base_url='https://api.deepseek.com'
)
```

## Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | **Required** | The search query text |
| `source_knowledge_base` | str | **Required** | Path to the document directory |
| `topk` | int | `3` | Number of top results to return |
| `llm` | str | `'deepseek-chat'` | LLM model name (`'gpt-4o'`, `'deepseek-chat'`, etc.) |
| `api` | str | `'your-api-key'` | API key for the LLM service |
| `base_url` | str | `'https://api.deepseek.com/v1'` | API base URL |
| `embedding_model` | str | `'./models/bge-m3'` | Path to embedding model |
| `bert_model_path` | str | `'pretrain_model.pth'` | Path to BERT query expansion model |
| `chunk_size` | int | `512` | Document max chunk size for processing |
| `retrieval_mode` | str | `'gkgr'` | Retrieval mode: `'vector'`, `'kg'`, or `'gkgr'` |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force_reinit` | bool | `False` | Force reinitialization of cached instances |
| `fusion_weights` | List[float] | `[0.6, 0.4]` | Weights for combining vector and KG retrieval |
| `expansion_weights` | List[float] | `[0.5, 0.3, 0.2]` | Weights for original and expanded queries |


# Test
```
from utils.test import retrieve_test, generate_test
annotated_files = 'retrieve_files_with_annotated'
metric = ['MRR', 'Acc'] # generate is 'F1'
test_results = retrieve_test(annotated_files, metric, mode='retrieve')
```

# Note
The all relevant source will be released before 7.25, 2025!!

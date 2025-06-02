# Construction-Disclosure-Documents-Reviewing-System
This is the official repo for our paper: "Generative Knowledge-Guided Review System for Construction Disclosure Documents Reviewing"

# Data
- Relevant data files could be acquired in `./data/`.
- Relevant data process tools are in `./data/process.ipynb`

# Weights
The pretrained weights can be acquired in at [google_drive]().

# Train
You can train the extraction modules in this commend:
```
python train_extract.py
```

 # Inference
 ```
frome CDDRS import 
frome CDDRS import GKGR

source_knowledge_base = 'path_to_knowledge_base'
query = 'your_retrieval_query'
retrieval_result = GKGR(query, source_knowledge_base, topk=3, llm='deepseek', api='your_own_deepseek_api', base_url='https://api.deepseek.com')
```

# Test
```
from utils.test import retrieve_test, generate_test
annotated_files = 'retrieve_files_with_annotated'
metric = ['MRR', 'Acc'] # generate is 'F1'
test_results = retrieve_test(annotated_files, metric, mode='retrieve')
```

# Note
The relevant source will be released after acceptance.

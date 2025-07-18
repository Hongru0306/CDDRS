from transformers import BertModel, BertTokenizer
import torch
from torch import nn

class TripleOutputBERT(torch.nn.Module):
    def __init__(self, vocab_size=21128):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(768, 768),
                torch.nn.GELU(),
                torch.nn.Linear(768, vocab_size)
            ) for _ in range(3)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, 768]
        return [head(sequence_output) for head in self.heads]  # [batch, seq_len, vocab_size]

class QueryExpander:
    def __init__(self, model_path, tokenizer_name='bert-base-chinese', max_length=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        

        self.model = TripleOutputBERT(vocab_size=self.tokenizer.vocab_size)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def expand(self, query):
        encoding = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'].to(self.device),
                attention_mask=encoding['attention_mask'].to(self.device)
            )
        
        results = []
        for logits in outputs:
            pred_ids = torch.argmax(logits, dim=-1)[0]  # [max_length]
            text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
            results.append(text.strip())
        
        return results[:3]  

if __name__ == "__main__":
    expander = QueryExpander(model_path='pretrain-model.pth')
    test_query = "可以描述一下索膜结构初始状态确定的具体方法吗？特别是力密度法、动力松弛法和非线性有限单元法的应用原理和特点。"
    query1, query2, query3 = expander.expand(test_query)
    print(f"Query1: {query1}")
    print(f"Query2: {query2}")
    print(f"Query3: {query3}")

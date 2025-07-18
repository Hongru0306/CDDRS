import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup

def preprocess_data(data, tokenizer, max_length=512):
    input_ids_list = []
    attention_mask_list = []
    start_labels_list = []
    end_labels_list = []

    for _, row in data.iterrows():
        query = row['Query']
        max_chunk = row['max']
        mid_chunk = row['mid'] if not pd.isna(row['mid']) else ""
        lit_chunk = row['lit'] if not pd.isna(row['lit']) else ""

        def find_chunk_positions(query, chunk):
            if not chunk:
                return 0, 0
            tokenized_query = tokenizer(query, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
            input_ids = tokenized_query['input_ids'][0]
            tokenized_chunk = tokenizer(chunk, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
            for i in range(len(input_ids) - len(tokenized_chunk) + 1):
                if torch.equal(input_ids[i:i + len(tokenized_chunk)], tokenized_chunk):
                    return i, i + len(tokenized_chunk) - 1
            return 0, 0

        def update_query(query, start, end):
            tokenized_query = tokenizer.tokenize(query)
            return tokenizer.convert_tokens_to_string(tokenized_query[:start] + tokenized_query[end + 1:])

        tokenized_input = tokenizer(query, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        input_ids = tokenized_input['input_ids'][0]

        start_positions = [0] * 512
        end_positions = [0] * 512

        max_start, max_end = find_chunk_positions(query, max_chunk)
        if max_start != 0 or max_end != 0:
            start_positions[max_start] = 1
            end_positions[max_end] = 1
            query = update_query(query, max_start, max_end)

            mid_start, mid_end = find_chunk_positions(query, mid_chunk)
            if mid_start != 0 or mid_end != 0:
                start_positions[mid_start] = 1
                end_positions[mid_end] = 1
                query = update_query(query, mid_start, mid_end)

                lit_start, lit_end = find_chunk_positions(query, lit_chunk)
                if lit_start != 0 or lit_end != 0:
                    start_positions[lit_start] = 1
                    end_positions[lit_end] = 1

        input_ids_list.append(input_ids)
        attention_mask_list.append(tokenized_input['attention_mask'][0])
        start_labels_list.append(torch.tensor(start_positions))
        end_labels_list.append(torch.tensor(end_positions))

    dataset = TensorDataset(torch.stack(input_ids_list), torch.stack(attention_mask_list), torch.stack(start_labels_list), torch.stack(end_labels_list))
    return dataset

class BERTChunkClassifier(nn.Module):
    def __init__(self, model_name='bert-base-chinese'):
        super(BERTChunkClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.start_classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.end_classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        start_logits = self.start_classifier(sequence_output)
        end_logits = self.end_classifier(sequence_output)
        return start_logits, end_logits

def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_start_labels = batch[2].to(device)
        b_end_labels = batch[3].to(device)

        model.zero_grad()

        start_logits, end_logits = model(b_input_ids, b_attention_mask)

        loss_fct = nn.CrossEntropyLoss()

        start_loss = loss_fct(start_logits.view(-1, 2), b_start_labels.view(-1))
        end_loss = loss_fct(end_logits.view(-1, 2), b_end_labels.view(-1))

        loss = start_loss + end_loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_start_preds = []
    all_end_preds = []
    all_start_labels = []
    all_end_labels = []

    with torch.no_grad():
        for batch in dataloader:
            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)
            b_start_labels = batch[2].to(device)
            b_end_labels = batch[3].to(device)

            start_logits, end_logits = model(b_input_ids, b_attention_mask)

            loss_fct = nn.CrossEntropyLoss()

            start_loss = loss_fct(start_logits.view(-1, 2), b_start_labels.view(-1))
            end_loss = loss_fct(end_logits.view(-1, 2), b_end_labels.view(-1))

            loss = start_loss + end_loss
            total_loss += loss.item()

            all_start_preds.extend(torch.argmax(start_logits, dim=-1).view(-1).cpu().numpy())
            all_end_preds.extend(torch.argmax(end_logits, dim=-1).view(-1).cpu().numpy())
            all_start_labels.extend(b_start_labels.view(-1).cpu().numpy())
            all_end_labels.extend(b_end_labels.view(-1).cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    start_f1 = f1_score(all_start_labels, all_start_preds, average='macro')
    end_f1 = f1_score(all_end_labels, all_end_preds, average='macro')
    avg_f1 = (start_f1 + end_f1) / 2

    return avg_loss, avg_f1

def main():
    parser = argparse.ArgumentParser(description='Train BERT Chunk Classifier')
    parser.add_argument('-i', '--input', required=True, help='Input CSV file path')
    parser.add_argument('-o', '--output', default='pretrain-model.pth', help='Output model file path (default: pretrain-model.pth)')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='Number of training epochs (default: 200)')
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-6, help='Learning rate (default: 5e-6)')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('-s', '--split_ratio', type=float, default=0.9, help='Train/validation split ratio (default: 0.9)')
    parser.add_argument('-m', '--max_length', type=int, default=512, help='Maximum sequence length (default: 512)')
    parser.add_argument('-w', '--weight_decay', type=float, default=0.01, help='Weight decay (default: 0.01)')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps (default: 0)')
    parser.add_argument('--print_interval', type=int, default=20, help='Print F1 score interval (default: 20)')
    parser.add_argument('--model_name', default='bert-base-chinese', help='BERT model name (default: bert-base-chinese)')
    args = parser.parse_args()

    try:
        data = pd.read_csv(args.input)
        print(f"Successfully loaded data from {args.input}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    print(f"Loaded BERT tokenizer: {args.model_name}")

    print("Preprocessing data...")
    dataset = preprocess_data(data, tokenizer, args.max_length)

    train_size = int(args.split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    print("Initializing model...")
    model = BERTChunkClassifier(args.model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8, weight_decay=args.weight_decay)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    print(f"Training parameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Train/val split: {args.split_ratio}")
    print(f"  Max length: {args.max_length}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"Starting training...")
    
    for epoch in range(args.epochs):
        train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        val_loss, val_f1 = evaluate(model, val_dataloader, device)

        print(f"Epoch {epoch+1}/{args.epochs} - Train loss: {train_loss:.4f} - Validation loss: {val_loss:.4f}")

        if (epoch + 1) % args.print_interval == 0:
            print(f"Epoch {epoch+1} F1 Score: {val_f1:.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()
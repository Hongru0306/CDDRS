import pandas as pd
import numpy as np
import Levenshtein
from tqdm import tqdm
import os
from typing import Optional, Any, List, Dict, Union
import json
from datetime import datetime


class GenerationEvaluator:
    def __init__(self, generator, top_k: int = 5, metrics: List[str] = None):
        self.generator = generator
        self.top_k = top_k
        self.metrics = metrics or ['bleu', 'rouge_1', 'rouge_2', 'precision', 'recall', 'f1']
    
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        try:
            from nltk.translate.bleu_score import sentence_bleu
            from nltk.tokenize import word_tokenize
            import nltk
            nltk.download('punkt', quiet=True)
            
            ref_tokens = word_tokenize(reference.lower())
            cand_tokens = word_tokenize(candidate.lower())
            
            return sentence_bleu([ref_tokens], cand_tokens)
        except ImportError:
            ref_words = set(reference.lower().split())
            cand_words = set(candidate.lower().split())
            
            if not cand_words:
                return 0.0
            
            intersection = ref_words.intersection(cand_words)
            return len(intersection) / len(cand_words)
    
    def calculate_rouge_score(self, reference: str, candidate: str) -> Dict[str, float]:
        try:
            import jieba
            ref_words = list(jieba.cut(reference.lower()))
            cand_words = list(jieba.cut(candidate.lower()))
        except ImportError:
            ref_words = reference.lower().split()
            cand_words = candidate.lower().split()
        
        if not ref_words or not cand_words:
            return {'rouge_1': 0.0, 'rouge_2': 0.0}
        
        ref_unigrams = set(ref_words)
        cand_unigrams = set(cand_words)
        rouge_1 = len(ref_unigrams.intersection(cand_unigrams)) / len(ref_unigrams) if len(ref_unigrams) > 0 else 0.0
        
        if len(ref_words) < 2 or len(cand_words) < 2:
            rouge_2 = 0.0
        else:
            ref_bigrams = set(zip(ref_words[:-1], ref_words[1:]))
            cand_bigrams = set(zip(cand_words[:-1], cand_words[1:]))
            rouge_2 = len(ref_bigrams.intersection(cand_bigrams)) / len(ref_bigrams) if len(ref_bigrams) > 0 else 0.0
        
        return {
            'rouge_1': rouge_1,
            'rouge_2': rouge_2
        }
    
    def calculate_precision_recall_f1(self, reference: str, candidate: str) -> Dict[str, float]:
        try:
            import jieba
            ref_tokens = set(jieba.cut(reference.lower()))
            cand_tokens = set(jieba.cut(candidate.lower()))
        except ImportError:
            ref_tokens = set(reference.lower().split())
            cand_tokens = set(candidate.lower().split())
        
        ref_tokens = {token for token in ref_tokens if token.strip()}
        cand_tokens = {token for token in cand_tokens if token.strip()}
        
        if not ref_tokens and not cand_tokens:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        if not cand_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        if not ref_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        intersection = ref_tokens.intersection(cand_tokens)
        
        precision = len(intersection) / len(cand_tokens)
        recall = len(intersection) / len(ref_tokens)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def generate_test(self, input_csv: str, output_csv: str) -> Dict[str, float]:
        df = pd.read_csv(input_csv, encoding='utf-8')
        
        required_columns = ['question', 'ground_truth']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV file missing required columns: {missing_columns}")
        
        metric_scores = {metric: [] for metric in ['bleu', 'rouge_1', 'rouge_2', 'precision', 'recall', 'f1']}
        results = []
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing generation test"):
            question = row['question']
            ground_truth = row['ground_truth']
            context = row.get('context', '')
            
            try:
                if hasattr(self.generator, 'query'):
                    generated_answer = self.generator.query(question)
                elif hasattr(self.generator, 'generate'):
                    generated_answer = self.generator.generate(question, context=context)
                else:
                    raise AttributeError("Generator must have query or generate method")
                
                generated_answer = str(generated_answer).strip()
            except Exception as e:
                print(f"Generation failed - Question: {question}, Error: {e}")
                generated_answer = ""
            
            result_row = {
                'question': question,
                'ground_truth': ground_truth,
                'context': context,
                'generated_answer': generated_answer,
            }
            
            if 'bleu' in self.metrics:
                bleu_score = self.calculate_bleu_score(ground_truth, generated_answer)
                metric_scores['bleu'].append(bleu_score)
                result_row['bleu_score'] = bleu_score
            
            if 'rouge_1' in self.metrics or 'rouge_2' in self.metrics:
                rouge_scores = self.calculate_rouge_score(ground_truth, generated_answer)
                if 'rouge_1' in self.metrics:
                    metric_scores['rouge_1'].append(rouge_scores['rouge_1'])
                    result_row['rouge_1'] = rouge_scores['rouge_1']
                if 'rouge_2' in self.metrics:
                    metric_scores['rouge_2'].append(rouge_scores['rouge_2'])
                    result_row['rouge_2'] = rouge_scores['rouge_2']
            
            if any(metric in self.metrics for metric in ['precision', 'recall', 'f1']):
                prf_scores = self.calculate_precision_recall_f1(ground_truth, generated_answer)
                if 'precision' in self.metrics:
                    metric_scores['precision'].append(prf_scores['precision'])
                    result_row['precision'] = prf_scores['precision']
                if 'recall' in self.metrics:
                    metric_scores['recall'].append(prf_scores['recall'])
                    result_row['recall'] = prf_scores['recall']
                if 'f1' in self.metrics:
                    metric_scores['f1'].append(prf_scores['f1'])
                    result_row['f1'] = prf_scores['f1']
            
            results.append(result_row)
        
        final_metrics = {}
        for metric in self.metrics:
            if metric in metric_scores:
                final_metrics[f'average_{metric}'] = np.mean(metric_scores[metric])
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False, encoding='utf-8')
        
        summary_path = output_csv.replace('.csv', '_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_questions': len(df),
                'metrics': final_metrics,
                'config': {
                    'top_k': self.top_k,
                    'selected_metrics': self.metrics
                }
            }, f, indent=2, ensure_ascii=False)
        
        return final_metrics


def generate_test(generator,
                 input_csv: str, 
                 output_csv: str,
                 top_k: int = 5,
                 metrics: List[str] = None) -> Dict[str, float]:
    evaluator = GenerationEvaluator(generator, top_k, metrics)
    return evaluator.generate_test(input_csv, output_csv)
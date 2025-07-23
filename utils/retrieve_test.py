import pandas as pd
import numpy as np
import Levenshtein
from snownlp import SnowNLP
import re
from llama_index.core.schema import QueryBundle
from tqdm import tqdm
import os
from typing import Optional, Any, List, Dict, Union
import json
from datetime import datetime


class RetrievalEvaluator:
    def __init__(self, retriever, top_k: int = 5, metrics: List[str] = None):
        self.retriever = retriever
        self.top_k = top_k
        self.metrics = metrics or ['mrr@5', 'hr@5', 'ndcg@3']
        self._validate_metrics()
    
    def _validate_metrics(self):
        valid_prefixes = ['mrr@', 'hr@', 'ndcg@']
        
        for metric in self.metrics:
            if not any(metric.startswith(prefix) for prefix in valid_prefixes):
                raise ValueError(f"Invalid metric format: {metric}. Use format like 'mrr@3', 'hr@5', 'ndcg@3'")
            
            try:
                n = int(metric.split('@')[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid metric format: {metric}. Use format like 'mrr@3', 'hr@5', 'ndcg@3'")
            
            if n > self.top_k:
                raise ValueError(f"Metric {metric}: n={n} cannot be larger than top_k={self.top_k}")
    
    def split_sentences(self, text: str) -> List[str]:
        s = SnowNLP(text)
        return s.sentences
    
    def normalized_levenshtein_similarity(self, str1: str, str2: str, threshold: float = 0.7) -> bool:
        if len(str1) > len(str2):
            return False

        max_similarity = 0.0

        for i in range(len(str2) - len(str1) + 1):
            substring = str2[i:i + len(str1)]
            distance = Levenshtein.distance(str1, substring)
            similarity = 1 - distance / max(len(str1), len(substring))
            max_similarity = max(max_similarity, similarity)
            
            if max_similarity >= threshold:
                return True

        return max_similarity >= threshold
    
    def calculate_ndcg_at_k(self, relevance_scores: List[int], k: int) -> float:
        if not relevance_scores or k <= 0:
            return 0.0
        
        rel_scores = relevance_scores[:k]
        
        dcg = rel_scores[0]
        for i in range(1, len(rel_scores)):
            dcg += rel_scores[i] / np.log2(i + 1)
        
        ideal_scores = sorted(rel_scores, reverse=True)
        idcg = ideal_scores[0] if ideal_scores else 0
        for i in range(1, len(ideal_scores)):
            idcg += ideal_scores[i] / np.log2(i + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_mrr_at_k(self, relevance_scores: List[int], k: int) -> float:
        for i, score in enumerate(relevance_scores[:k]):
            if score > 0:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_hr_at_k(self, relevance_scores: List[int], k: int) -> float:
        return 1.0 if any(relevance_scores[:k]) else 0.0
    
    def retrieve_test(self, 
                     input_csv: str, 
                     output_csv: str, 
                     reranker: Optional[Any] = None,
                     threshold: float = 0.7) -> Dict[str, float]:
        df = pd.read_csv(input_csv, encoding='utf-8')
        
        required_columns = ['question', 'retrieved_contexts']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV file missing required columns: {missing_columns}")
        
        metric_scores = {metric: [] for metric in self.metrics}
        results = []

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing retrieval test"):
            question = row['question']
            retrieved_context = row['retrieved_contexts']
            sentences = self.split_sentences(retrieved_context)

            try:
                retrievals = self.retriever.retrieve(question)
                original_retrievals = [node.get_content() for node in retrievals]
            except Exception as e:
                print(f"Retrieval failed - Question: {question}, Error: {e}")
                original_retrievals = []
                retrievals = []

            rerank_info = "nothing"
            if reranker is not None and retrievals:
                try:
                    query_bundle = QueryBundle(query_str=question)
                    retrievals = reranker._postprocess_nodes(retrievals, query_bundle)
                    rerank_info = [node.get_content() for node in retrievals]
                except Exception as e:
                    print(f"Reranking failed - Question: {question}, Error: {e}")

            retrievals = retrievals[:self.top_k]

            all_sentence_metrics = {}
            for metric in self.metrics:
                all_sentence_metrics[metric] = []

            for sentence in sentences:
                relevance_scores = []
                
                for rank, node in enumerate(retrievals):
                    is_relevant = self.normalized_levenshtein_similarity(
                        sentence, re.sub(r'\s+', '', node.get_content()), threshold
                    )
                    relevance_scores.append(1 if is_relevant else 0)

                for metric in self.metrics:
                    metric_type, k = metric.split('@')
                    k = int(k)
                    
                    if metric_type == 'mrr':
                        score = self.calculate_mrr_at_k(relevance_scores, k)
                    elif metric_type == 'hr':
                        score = self.calculate_hr_at_k(relevance_scores, k)
                    elif metric_type == 'ndcg':
                        score = self.calculate_ndcg_at_k(relevance_scores, k)
                    
                    all_sentence_metrics[metric].append(score)

            result_row = {
                'question': question,
                'retrieved_contexts': retrieved_context,
                'original_retrievals': original_retrievals,
                'reranked_retrievals': rerank_info,
            }
            
            for metric in self.metrics:
                sentence_scores = all_sentence_metrics[metric]
                mean_score = float(np.mean(sentence_scores)) if sentence_scores else 0.0
                metric_scores[metric].append(mean_score)
                result_row[metric] = mean_score
            
            results.append(result_row)

        final_metrics = {}
        for metric in self.metrics:
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
                    'threshold': threshold,
                    'has_reranker': reranker is not None,
                    'selected_metrics': self.metrics
                }
            }, f, indent=2, ensure_ascii=False)

        return final_metrics


def retrieve_test(retriever, 
                 input_csv: str, 
                 output_csv: str, 
                 top_k: int = 5,
                 metrics: List[str] = None,
                 reranker: Optional[Any] = None,
                 threshold: float = 0.7) -> Dict[str, float]:
    evaluator = RetrievalEvaluator(retriever, top_k, metrics)
    return evaluator.retrieve_test(input_csv, output_csv, reranker, threshold)
import numpy as np
from typing import Any, Tuple, Dict, Iterable, List, NamedTuple, Union, Optional
from collections import Counter
import math

def get_unique_tokens(corpus_tokens):
    unique_tokens = set()
    for doc_tokens in corpus_tokens:
        unique_tokens.update(doc_tokens)
    return unique_tokens

class Tokenized(NamedTuple):

    ids: List[List[int]]
    vocab: Dict[str, int]

def is_list_of_list_of_type(obj, type_=int):
    if not isinstance(obj, list):
        return False
    if len(obj) == 0:
        return False
    first_elem = obj[0]
    if not isinstance(first_elem, list):
        return False
    if len(first_elem) == 0:
        return False
    first_token = first_elem[0]
    if not isinstance(first_token, type_):
        return False
    return True

class Results(NamedTuple):
    documents: np.ndarray
    scores: np.ndarray
    def __len__(self):
        return len(self.documents)

class GK:
    def __init__(
        self,
        k1=1.5,
        b=0.75,
        delta=0.5,
        method="lucene",
        idf_method=None,
        dtype="float32",
        int_dtype="int32",
        corpus=None,
        backend="numpy",
    ):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.dtype = dtype
        self.int_dtype = int_dtype
        self.method = method
        self.idf_method = idf_method if idf_method is not None else method
        self.methods_requiring_nonoccurrence = ("gkl", "gk+")
        self.corpus = corpus
        self.backend = backend

    @staticmethod
    def _calculate_doc_freqs(corpus_tokens, unique_tokens):
        unique_tokens = set(unique_tokens)
        doc_frequencies = {token: 0 for token in unique_tokens}
        for doc_tokens in corpus_tokens:
            shared_tokens = unique_tokens.intersection(doc_tokens)
            for token in shared_tokens:
                doc_frequencies[token] += 1
        return doc_frequencies

    @staticmethod
    def _build_idf_array(doc_frequencies, n_docs, compute_idf_fn, dtype="float32"):
        n_vocab = len(doc_frequencies)
        idf_array = np.zeros(n_vocab, dtype=dtype)
        for token_id, df in doc_frequencies.items():
            idf_array[token_id] = compute_idf_fn(df, N=n_docs)
        return idf_array

    @staticmethod
    def _build_5 (doc_frequencies, n_docs, compute_idf_fn, calculate_tfc_fn, l_d, l_avg, k1, b, delta, dtype="float32"):
        n_vocab = len(doc_frequencies)
        nonoccurrence_array = np.zeros(n_vocab, dtype=dtype)
        for token_id, df in doc_frequencies.items():
            idf = compute_idf_fn(df, N=n_docs)
            tfc = calculate_tfc_fn(tf_array=0, l_d=l_d, l_avg=l_avg, k1=k1, b=b, delta=delta)
            nonoccurrence_array[token_id] = idf * tfc
        return nonoccurrence_array

    @staticmethod
    def _score_tfc_robertson(tf_array, l_d, l_avg, k1, b, delta=None):
        return tf_array / (k1 * ((1 - b) + b * l_d / l_avg) + tf_array)

    @staticmethod
    def _score_tfc_lucene(tf_array, l_d, l_avg, k1, b, delta=None):
        return GK._score_tfc_robertson(tf_array, l_d, l_avg, k1, b)

    @staticmethod
    def _score_tfc_atire(tf_array, l_d, l_avg, k1, b, delta=None):
        return (tf_array * (k1 + 1)) / (tf_array + k1 * (1 - b + b * l_d / l_avg))

    @staticmethod
    def _score_tfc_gkl(tf_array, l_d, l_avg, k1, b, delta):
        c_array = tf_array / (1 - b + b * l_d / l_avg)
        return ((k1 + 1) * (c_array + delta)) / (k1 + c_array + delta)

    @staticmethod
    def _score_tfc_gkplus(tf_array, l_d, l_avg, k1, b, delta):
        num = (k1 + 1) * tf_array
        den = k1 * (1 - b + b * l_d / l_avg) + tf_array
        return (num / den) + delta

    @staticmethod
    def _select_tfc_scorer(method):
        if method == "robertson":
            return GK._score_tfc_robertson
        elif method == "lucene":
            return GK._score_tfc_lucene
        elif method == "atire":
            return GK._score_tfc_atire
        elif method == "gkl":
            return GK._score_tfc_gkl
        elif method == "gk+":
            return GK._score_tfc_gkplus
        else:
            raise ValueError(f"Invalid score_tfc value: {method}.")

    @staticmethod
    def _score_idf_robertson(df, N, allow_negative=False):
        inner = (N - df + 0.5) / (df + 0.5)
        if not allow_negative and inner < 1:
            inner = 1
        return math.log(inner)

    @staticmethod
    def _score_idf_lucene(df, N):
        return math.log(1 + (N - df + 0.5) / (df + 0.5))

    @staticmethod
    def _score_idf_atire(df, N):
        return math.log(N / df)

    @staticmethod
    def _score_idf_gkl(df, N):
        return math.log((N + 1) / (df + 0.5))

    @staticmethod
    def _score_idf_gkplus(df, N):
        return math.log((N + 1) / df)

    @staticmethod
    def _select_idf_scorer(method):
        if method == "robertson":
            return GK._score_idf_robertson
        elif method == "lucene":
            return GK._score_idf_lucene
        elif method == "atire":
            return GK._score_idf_atire
        elif method == "gkl":
            return GK._score_idf_gkl
        elif method == "gk+":
            return GK._score_idf_gkplus
        else:
            raise ValueError(f"Invalid score_idf_inner value: {method}.")
 
    def _build_nonoccurrence_array(
        self,
        doc_frequencies: dict,
        n_docs: int,
        compute_idf_fn,
        calculate_tfc_fn,
        l_d,
        l_avg,
        k1,
        b,
        delta,
        dtype="float32",
    ) -> np.ndarray:
        n_vocab = len(doc_frequencies)
        nonoccurrence_array = np.zeros(n_vocab, dtype=dtype)
        for token_id, df in doc_frequencies.items():
            idf = compute_idf_fn(df, N=n_docs)
            tfc = calculate_tfc_fn(
                tf_array=0, l_d=l_d, l_avg=l_avg, k1=k1, b=b, delta=delta
            )
            nonoccurrence_array[token_id] = idf * tfc
        return nonoccurrence_array
    
    @staticmethod
    def _get_counts_from_token_ids(token_ids, dtype, int_dtype):
        token_counter = Counter(token_ids)
        voc_ind = np.array(list(token_counter.keys()), dtype=int_dtype)
        tf_array = np.array(list(token_counter.values()), dtype=dtype)
        return voc_ind, tf_array

    @staticmethod
    def _build_scores_and_indices_for_matrix(
        corpus_token_ids, idf_array, avg_doc_len, doc_frequencies, k1, b, delta, show_progress=True, leave_progress=False, dtype="float32", int_dtype="int32", method="robertson", nonoccurrence_array=None
    ):
        array_size = sum(doc_frequencies.values())
        scores = np.empty(array_size, dtype=dtype)
        doc_indices = np.empty(array_size, dtype=int_dtype)
        voc_indices = np.empty(array_size, dtype=int_dtype)
        calculate_tfc = GK._select_tfc_scorer(method)
        i = 0
        for doc_idx, token_ids in enumerate(corpus_token_ids):
            doc_len = len(token_ids)
            voc_ind_doc, tf_array = GK._get_counts_from_token_ids(
                token_ids, dtype=dtype, int_dtype=int_dtype
            )
            tfc = calculate_tfc(
                tf_array=tf_array, l_d=doc_len, l_avg=avg_doc_len, k1=k1, b=b, delta=delta
            )
            idf = idf_array[voc_ind_doc]
            scores_doc = idf * tfc
            if method in ("gkl", "gk+"):
                if nonoccurrence_array is not None:
                    scores_doc -= nonoccurrence_array[voc_ind_doc]
            doc_len = len(scores_doc)
            start, end = i, i + doc_len
            i = end
            doc_indices[start:end] = doc_idx
            voc_indices[start:end] = voc_ind_doc
            scores[start:end] = scores_doc
        return scores, doc_indices, voc_indices

    def build_index_from_ids(
        self, unique_token_ids: List[int], corpus_token_ids: List[List[int]],
        show_progress=True, leave_progress=False,
    ):
        import scipy.sparse as sp
        avg_doc_len = np.array([len(doc_ids) for doc_ids in corpus_token_ids]).mean()
        n_docs = len(corpus_token_ids)
        n_vocab = len(unique_token_ids)
        doc_frequencies = self._calculate_doc_freqs(corpus_token_ids, unique_token_ids)
        if self.method in self.methods_requiring_nonoccurrence:
            self.nonoccurrence_array = self._build_nonoccurrence_array(
                doc_frequencies=doc_frequencies,
                n_docs=n_docs,
                compute_idf_fn=self._select_idf_scorer(self.idf_method),
                calculate_tfc_fn=self._select_tfc_scorer(self.method),
                l_d=avg_doc_len,
                l_avg=avg_doc_len,
                k1=self.k1,
                b=self.b,
                delta=self.delta,
            )
        else:
            self.nonoccurrence_array = None
        idf_array = self._build_idf_array(
            doc_frequencies=doc_frequencies,
            n_docs=n_docs,
            compute_idf_fn=self._select_idf_scorer(self.idf_method),
        )
        scores_flat, doc_idx, vocab_idx = self._build_scores_and_indices_for_matrix(
            corpus_token_ids=corpus_token_ids,
            idf_array=idf_array,
            avg_doc_len=avg_doc_len,
            doc_frequencies=doc_frequencies,
            k1=self.k1,
            b=self.b,
            delta=self.delta,
            dtype=self.dtype,
            int_dtype=self.int_dtype,
            method=self.method,
            nonoccurrence_array=self.nonoccurrence_array,
        )
        score_matrix = sp.csc_matrix(
            (scores_flat, (doc_idx, vocab_idx)),
            shape=(n_docs, n_vocab),
            dtype=self.dtype,
        )
        data = score_matrix.data
        indices = score_matrix.indices
        indptr = score_matrix.indptr
        scores = {
            "data": data,
            "indices": indices,
            "indptr": indptr,
            "num_docs": n_docs,
        }
        return scores

    def build_index_from_tokens(self, corpus_tokens, show_progress=True, leave_progress=False):
        unique_tokens = get_unique_tokens(corpus_tokens)
        vocab_dict = {token: i for i, token in enumerate(unique_tokens)}
        unique_token_ids = [vocab_dict[token] for token in unique_tokens]
        corpus_token_ids = [[vocab_dict[token] for token in tokens] for tokens in corpus_tokens]
        scores = self.build_index_from_ids(
            unique_token_ids=unique_token_ids,
            corpus_token_ids=corpus_token_ids,
        )
        return scores, vocab_dict

    def index(self, corpus: Union[Iterable, Tuple], show_progress=True, leave_progress=False):
        if isinstance(corpus, list) and isinstance(corpus[0], list):
            scores, vocab_dict = self.build_index_from_tokens(corpus)
        elif isinstance(corpus, tuple) and len(corpus) == 2:
            corpus_token_ids, vocab_dict = corpus
            unique_token_ids = list(vocab_dict.values())
            scores = self.build_index_from_ids(
                unique_token_ids=unique_token_ids,
                corpus_token_ids=corpus_token_ids,
            )
        else:
            raise ValueError("Corpus must be a list of lists or a (list, vocab_dict) tuple.")
        self.scores = scores
        self.vocab_dict = vocab_dict

    def get_tokens_ids(self, query_tokens: List[str]) -> List[int]:
        return [self.vocab_dict[token] for token in query_tokens if token in self.vocab_dict]

    def _compute_relevance_from_scores(
        self, data: np.ndarray, indptr: np.ndarray, indices: np.ndarray, num_docs: int,
        query_tokens_ids: np.ndarray, dtype: np.dtype, weights_dict: Dict[int, float] = {}
    ) -> np.ndarray:
        indptr_starts = indptr[query_tokens_ids]
        indptr_ends = indptr[query_tokens_ids + 1]
        scores = np.zeros(num_docs, dtype=dtype)
        for i in range(len(query_tokens_ids)):
            start, end = indptr_starts[i], indptr_ends[i]
            token_id = query_tokens_ids[i]
            token_weight = weights_dict.get(token_id, 1.0) if weights_dict else 1.0
            for j in range(start, end):
                scores[indices[j]] += data[j] * token_weight
        return scores

    def get_scores_from_ids(self, query_tokens_id: List[int], weight_mask=None, weights_dict=None) -> np.ndarray:
        data = self.scores["data"]
        indices = self.scores["indices"]
        indptr = self.scores["indptr"]
        num_docs = self.scores["num_docs"]
        dtype = np.dtype(self.dtype)
        int_dtype = np.dtype(self.int_dtype)
        query_tokens_ids: np.ndarray = np.asarray(query_tokens_id, dtype=int_dtype)
        max_token_id = int(query_tokens_ids.max(initial=0))
        if max_token_id >= len(indptr) - 1:
            raise ValueError("The maximum token ID in the query is higher than the number of tokens in the index.")
        scores = self._compute_relevance_from_scores(
            data=data,
            indptr=indptr,
            indices=indices,
            num_docs=num_docs,
            query_tokens_ids=query_tokens_ids,
            dtype=dtype,
            weights_dict=weights_dict or {}
        )
        if weight_mask is not None:
            scores *= weight_mask
        if hasattr(self, "nonoccurrence_array") and self.nonoccurrence_array is not None:
            nonoccurrence_scores = self.nonoccurrence_array[query_tokens_ids].sum()
            scores += nonoccurrence_scores
        return scores

    def get_scores(self, query_tokens_single, weight_mask=None, weights_dict: Dict[int, float] = {}) -> np.ndarray:
        if not isinstance(query_tokens_single, list):
            raise ValueError("The query_tokens must be a list of tokens.")
        if isinstance(query_tokens_single[0], str):
            query_tokens_ids = self.get_tokens_ids(query_tokens_single)
        elif isinstance(query_tokens_single[0], int):
            query_tokens_ids = query_tokens_single
        else:
            raise ValueError("The query_tokens must be a list of tokens or a list of token IDs.")
        return self.get_scores_from_ids(query_tokens_ids, weight_mask=weight_mask, weights_dict=weights_dict)

    def _get_top_k_results(self, query_tokens_single: List[str], k: int = 1000, sorted: bool = False, weight_mask: Optional[np.ndarray] = None, weights_dict: Dict[int, float] = {}):
        if len(query_tokens_single) == 0:
            scores_q = np.zeros(self.scores["num_docs"], dtype=self.dtype)
        else:
            scores_q = self.get_scores(query_tokens_single, weight_mask=weight_mask, weights_dict=weights_dict)
        topk_indices = np.argpartition(-scores_q, min(k, len(scores_q)-1))[:k]
        topk_scores = scores_q[topk_indices]
        if sorted:
            order = np.argsort(-topk_scores)
            topk_scores = topk_scores[order]
            topk_indices = topk_indices[order]
        return topk_scores, topk_indices
    

    def retrieve(
        self,
        query_tokens: Union[List[List[str]], 'Tokenized'],
        corpus: Optional[List[Any]] = None,
        k: int = 10,
        sorted: bool = True,
        return_as: str = "tuple",
        weight_mask: Optional[np.ndarray] = None,
        weights_dict: Optional[Dict[int, float]] = None
    ):
        if weights_dict is None:
            weights_dict = {}

        if isinstance(query_tokens, Tokenized):

            id2token = {v: k for k, v in query_tokens.vocab.items()}
            queries = [
                [id2token.get(token_id, "") for token_id in query]
                for query in query_tokens.ids
            ]
        else:
            queries = query_tokens


        results = [
            self._get_top_k_results(
                q, k=k, sorted=sorted, weight_mask=weight_mask, weights_dict=weights_dict
            ) for q in queries
        ]
        scores, indices = zip(*results)
        scores, indices = np.array(scores), np.array(indices)
        corpus = corpus if corpus is not None else self.corpus
        if corpus is None:
            retrieved_docs = indices
        else:
            index_flat = indices.flatten().tolist()
            docs = [corpus[i] for i in index_flat]
            retrieved_docs = np.array(docs).reshape(indices.shape)
        if return_as == "tuple":
            return Results(documents=retrieved_docs, scores=scores)
        elif return_as == "documents":
            return retrieved_docs
        else:
            raise ValueError("`return_as` must be either 'tuple' or 'documents'")
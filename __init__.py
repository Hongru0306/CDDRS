import os
import warnings
import jieba
from typing import List, Optional, Any, Dict, Union
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, QueryBundle
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, MetadataMode
from llama_index.core.vector_stores.utils import node_to_metadata_dict, metadata_dict_to_node
from llama_index.core.retrievers import QueryFusionRetriever
import kg_retriever

warnings.filterwarnings('ignore')

def load_chinese_stopwords() -> set:
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        chinese_stopwords = set(stopwords.words('chinese'))
        return chinese_stopwords
    except Exception as e:
        return {'的', '在', '是', '了', '和', '与', '及', '或', '但', '而', '等'}

def jieba_tokenizer(text):
    return list(jieba.cut(text))

def custom_stemmer(tokens):
    return tokens


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


class BERTQueryExpander:
    def __init__(self, model_path: str = 'pretrain_model.pth', 
                 tokenizer_name: str = 'bert-base-chinese'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        
        try:
            self.model = BERTChunkClassifier()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            self.model = None
    
    def expand(self, query: str) -> List[str]:
        if self.model is None:
            return [query]
        
        try:
            chunks = self._predict_chunks(query)
            valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip() and len(chunk.strip()) > 1]
            
            if not valid_chunks:
                return [query]
            
            if len(valid_chunks) == 1:
                return [query] + valid_chunks
            
            return [query] + valid_chunks
            
        except Exception as e:
            return [query]
    
    def _predict_chunks(self, query: str) -> List[str]:
        def get_chunk(text):
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                  padding='max_length', max_length=512)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            with torch.no_grad():
                start_logits, end_logits = self.model(input_ids, attention_mask)

            start_probs = torch.softmax(start_logits, dim=-1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=-1).cpu().numpy()

            start_idx = start_probs[0][:, 1].argmax()
            end_idx = end_probs[0][:, 1].argmax()

            if end_idx < start_idx:
                end_idx = start_idx

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

            def clean_tokens(tokens):
                return "".join([token.replace("##", "") for token in tokens])

            chunk = clean_tokens(tokens[start_idx:end_idx + 1])
            return chunk, start_idx, end_idx, input_ids[0]

        def update_query(input_ids, start, end):
            updated_input_ids = torch.cat((input_ids[:start], input_ids[end + 1:]))
            updated_query = self.tokenizer.decode(updated_input_ids, skip_special_tokens=True)
            return updated_query

        chunks = []
        current_query = query
        
        for i in range(3):
            chunk, start_idx, end_idx, input_ids = get_chunk(current_query)
            
            if start_idx == 0 and end_idx == 0:
                break
                
            chunks.append(chunk)
            current_query = update_query(input_ids, start_idx, end_idx)
            
            if len(current_query.strip()) < 2:
                break

        return chunks


class KGRetriever(BaseRetriever):
    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        chinese_stopwords=None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[Any] = None,
        objects: Optional[List[Any]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
    ):
        self.similarity_top_k = similarity_top_k
        self.chinese_stopwords = chinese_stopwords if chinese_stopwords is not None else set()

        if nodes is None:
            raise ValueError("Please pass nodes.")

        self.corpus = [node_to_metadata_dict(node) for node in nodes]
        corpus_tokens = self.tokenize(
            [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
            self.chinese_stopwords,
        )
        self.kg = kg_retriever.GK()
        self.kg.index(corpus_tokens, show_progress=verbose)

        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    @staticmethod
    def tokenize(texts, chinese_stopwords):
        if isinstance(texts, str):
            texts = [texts]

        result = []
        for text in texts:
            tokens = [t for t in jieba.cut(text) if t not in chinese_stopwords and t.strip()]
            result.append(tokens)
        return result

    @classmethod
    def from_defaults(
        cls,
        nodes: Optional[List[BaseNode]] = None,
        docstore: Optional[Any] = None,
        index: Optional[Any] = None,
        chinese_stopwords=None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
    ) -> "KGRetriever":
        if sum(bool(val) for val in [index, nodes, docstore]) != 1:
            raise ValueError("Please pass exactly one of index, nodes, or docstore.")

        if index is not None:
            docstore = index.docstore

        if docstore is not None:
            nodes = list(docstore.docs.values())

        assert nodes is not None, "Please pass nodes, docstore, or index."

        return cls(
            nodes=nodes,
            chinese_stopwords=chinese_stopwords,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        tokenized_query = self.tokenize(query, self.chinese_stopwords)
        indexes, scores = self.kg.retrieve(tokenized_query, k=self.similarity_top_k)
        indexes = indexes[0]
        scores = scores[0]
        nodes: List[NodeWithScore] = []
        for idx, score in zip(indexes, scores):
            node_dict = self.corpus[int(idx)]
            node = metadata_dict_to_node(node_dict)
            nodes.append(NodeWithScore(node=node, score=float(score)))
        return nodes


class CustomKGRetriever(KGRetriever):
    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        chinese_stopwords=None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        query_expander: Optional[BERTQueryExpander] = None,
        enable_query_expansion: bool = False,
        expansion_weights: List[float] = None,
        callback_manager: Optional[Any] = None,
        verbose: bool = False,
    ):
        self.query_expander = query_expander
        self.enable_query_expansion = enable_query_expansion
        self.expansion_weights = expansion_weights or [0.5, 0.3, 0.2]
        
        super().__init__(
            nodes=nodes,
            chinese_stopwords=chinese_stopwords,
            similarity_top_k=similarity_top_k,
            callback_manager=callback_manager,
            verbose=verbose,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str

        if not self.enable_query_expansion or not self.query_expander:
            return super()._retrieve(query_bundle)

        try:
            sub_queries = self.query_expander.expand(query)
        except Exception as e:
            return super()._retrieve(query_bundle)

        weights = self.expansion_weights[:len(sub_queries)]
        if len(weights) < len(sub_queries):
            remaining = len(sub_queries) - len(weights)
            last_weight = weights[-1] if weights else 0.1
            for i in range(remaining):
                weights.append(last_weight * 0.8 ** (i + 1))

        score_dict = {}

        for sub_query, weight in zip(sub_queries, weights):
            try:
                tokenized_query = self.tokenize(sub_query, self.chinese_stopwords)

                if not tokenized_query or not tokenized_query[0]:
                    continue

                indexes, scores = self.kg.retrieve(tokenized_query, k=self.similarity_top_k)
                indexes = indexes[0] if isinstance(indexes[0], list) else indexes
                scores = scores[0] if isinstance(scores[0], list) else scores

                for idx, score in zip(indexes, scores):
                    doc_id = int(idx)
                    weighted_score = score * weight
                    
                    if doc_id in score_dict:
                        score_dict[doc_id] += weighted_score
                    else:
                        score_dict[doc_id] = weighted_score

            except Exception as e:
                continue

        nodes = []
        for doc_id, total_score in score_dict.items():
            try:
                node_dict = self.corpus[doc_id]
                node = metadata_dict_to_node(node_dict)
                nodes.append(NodeWithScore(node=node, score=total_score))
            except (IndexError, KeyError) as e:
                continue

        nodes.sort(key=lambda x: x.score, reverse=True)
        return nodes[:self.similarity_top_k]


class CustomTokenizedRetriever(BaseRetriever):
    def __init__(
        self,
        kg_retriever: CustomKGRetriever,
        original_nodes_dict: dict,
        mode: str = "AND",
    ) -> None:
        self._kg_retriever = kg_retriever
        self.original_nodes_dict = original_nodes_dict
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        tokenized_nodes = self._kg_retriever.retrieve(query_bundle)

        original_nodes_with_scores = []
        for tokenized_node in tokenized_nodes:
            original_node = self.original_nodes_dict.get(tokenized_node.node.node_id)
            if original_node:
                score = tokenized_node.score
                original_nodes_with_scores.append(NodeWithScore(node=original_node, score=score))

        return original_nodes_with_scores


class DocumentRetriever:
    def __init__(self, 
                 documents_path: str = "./guifan",
                 api_key: str = "your-api-key",
                 base_url: str = "https://api.deepseek.com/v1",
                 model_name: str = "./models/bge-m3",
                 llm_model: str = "gpt-4o-mini",
                 chunk_size: int = 512,
                 chunk_overlap: int = 64,
                 top_k: int = 5,
                 enable_hybrid_search: bool = False,
                 enable_query_expansion: bool = False,
                 stopwords_path: str = "./chinese_stopwords.txt",
                 bert_model_path: str = "pretrain_model.pth",
                 bert_tokenizer_name: str = "bert-base-chinese",
                 retrieval_mode: str = "vector",
                 fusion_weights: List[float] = None,
                 expansion_weights: List[float] = None):
        
        self.documents_path = documents_path
        self.top_k = top_k
        self.retriever = None
        self.index = None
        
        self.enable_hybrid_search = enable_hybrid_search
        self.enable_query_expansion = enable_query_expansion
        self.retrieval_mode = retrieval_mode
        self.fusion_weights = fusion_weights or [0.6, 0.4]
        self.expansion_weights = expansion_weights or [0.5, 0.3, 0.2]
        
        self.chinese_stopwords = load_chinese_stopwords()
        
        self.bert_query_expander = None
        if enable_query_expansion:
            try:
                self.bert_query_expander = BERTQueryExpander(
                    model_path=bert_model_path,
                    tokenizer_name=bert_tokenizer_name
                )
            except Exception as e:
                self.enable_query_expansion = False
        
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = base_url
        
        self._setup_models(llm_model, model_name)
        
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        self._build_index()
        
        if enable_hybrid_search:
            self._setup_hybrid_retrievers()
    
    def _setup_models(self, llm_model: str, embedding_model: str):
        try:
            Settings.llm = OpenAI(model=llm_model)
            
            self.embed_model = HuggingFaceEmbedding(
                model_name=embedding_model,
                trust_remote_code=True,
                cache_folder="./model_cache"
            )
            Settings.embed_model = self.embed_model
        except Exception as e:
            raise
    
    def _build_index(self):
        try:
            if not os.path.exists(self.documents_path):
                raise ValueError(f"Document directory not found: {self.documents_path}")
            
            reader = SimpleDirectoryReader(self.documents_path)
            docs = reader.load_data()
            
            if not docs:
                raise ValueError(f"No documents found in directory: {self.documents_path}")
            
            self.all_nodes = []
            for doc in docs:
                file_path = doc.metadata.get('file_path', '')
                standard_name = self._extract_standard_name(file_path)
                
                doc.metadata['standard_name'] = standard_name
                
                nodes = self.splitter.get_nodes_from_documents([doc])
                
                for node in nodes:
                    node.metadata['standard_name'] = standard_name
                
                self.all_nodes.extend(nodes)
            
            self.index = VectorStoreIndex(self.all_nodes, embed_model=self.embed_model)
            self.retriever = self.index.as_retriever(similarity_top_k=self.top_k)
            
        except Exception as e:
            raise
    
    def _create_tokenized_nodes(self):
        tokenized_nodes = []
        for node in self.all_nodes:
            tokens = jieba.cut(node.get_content())
            filtered_tokens = [token for token in tokens if token not in self.chinese_stopwords and token.strip()]
            tokenized_content = ' '.join(filtered_tokens)
            
            from llama_index.core.schema import Document
            new_node = Document(text=tokenized_content, metadata=node.metadata)
            new_node.node_id = node.node_id
            
            tokenized_nodes.append(new_node)
        
        return tokenized_nodes
    
    def _setup_hybrid_retrievers(self):
        try:
            self.vector_retriever = self.index.as_retriever(similarity_top_k=self.top_k)
            
            self.tokenized_nodes = self._create_tokenized_nodes()
            self.all_nodes_dict = {node.node_id: node for node in self.all_nodes}
            
            self.kg_retriever = CustomKGRetriever(
                nodes=self.tokenized_nodes,
                chinese_stopwords=self.chinese_stopwords,
                similarity_top_k=self.top_k,
                query_expander=self.bert_query_expander,
                enable_query_expansion=self.enable_query_expansion,
                expansion_weights=self.expansion_weights,
                verbose=False
            )
            
            self.custom_retriever = CustomTokenizedRetriever(
                kg_retriever=self.kg_retriever,
                original_nodes_dict=self.all_nodes_dict,
                mode="AND"
            )
            
            self.fusion_retriever = QueryFusionRetriever(
                [self.vector_retriever, self.custom_retriever],
                retriever_weights=self.fusion_weights,
                similarity_top_k=self.top_k,
                num_queries=1,
                mode="relative_score",
                use_async=False,
                verbose=False,
            )
            
        except Exception as e:
            raise
    
    def _extract_standard_name(self, file_path: str) -> str:
        if not file_path:
            return "unknown"
        
        filename = os.path.basename(file_path)
        standard_name = os.path.splitext(filename)[0]
        
        return standard_name
    
    def retrieve(self, query: str) -> List:
        if self.retriever is None:
            raise ValueError("Retriever not initialized")
        
        try:
            if self.enable_hybrid_search:
                return self._hybrid_retrieve(query)
            else:
                return self.retriever.retrieve(query)
        except Exception as e:
            return []
    
    def _hybrid_retrieve(self, query: str) -> List[NodeWithScore]:
        try:
            if self.retrieval_mode == "vector":
                return self.vector_retriever.retrieve(query)
            elif self.retrieval_mode == "kg":
                query_bundle = QueryBundle(query_str=query)
                return self.custom_retriever.retrieve(query_bundle)
            elif self.retrieval_mode == "gkgr":
                return self.fusion_retriever.retrieve(query)
            else:
                return self.fusion_retriever.retrieve(query)
        except Exception as e:
            return []
    
    def query(self, question: str) -> str:
        if self.index is None:
            raise ValueError("Index not built")
        
        try:
            if self.enable_hybrid_search and hasattr(self, 'fusion_retriever'):
                query_engine = self.index.as_query_engine(
                    retriever=self.fusion_retriever,
                    similarity_top_k=self.top_k
                )
            else:
                query_engine = self.index.as_query_engine(similarity_top_k=self.top_k)
            
            response = query_engine.query(question)
            return str(response)
        except Exception as e:
            return ""


class CDDRS:
    _instances = {}
    
    def __init__(self):
        pass
    
    @classmethod
    def _get_instance_key(cls, source_knowledge_base: str, llm: str, api: str, base_url: str) -> str:
        return f"{source_knowledge_base}:{llm}:{api}:{base_url}"
    
    @classmethod
    def _initialize_retriever(cls, 
                            source_knowledge_base: str,
                            llm: str = 'gpt-4o-mini',
                            api: str = None,
                            base_url: str = "https://api.deepseek.com/v1",
                            embedding_model: str = "./models/bge-m3",
                            chunk_size: int = 512,
                            chunk_overlap: int = 64,
                            enable_hybrid_search: bool = True,
                            enable_query_expansion: bool = True,
                            retrieval_mode: str = "gkgr",
                            bert_model_path: str = "pretrain_model.pth",
                            **kwargs) -> DocumentRetriever:
        
        if api is None:
            api = "your-api-key"
        
        try:
            retriever = DocumentRetriever(
                documents_path=source_knowledge_base,
                api_key=api,
                base_url=base_url,
                model_name=embedding_model,
                llm_model=llm,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                enable_hybrid_search=enable_hybrid_search,
                enable_query_expansion=enable_query_expansion,
                retrieval_mode=retrieval_mode,
                bert_model_path=bert_model_path,
                **kwargs
            )
            
            return retriever
            
        except Exception as e:
            raise


def GKGR(query: str,
         source_knowledge_base: str,
         topk: int = 3,
         llm: str = 'gpt-4o-mini',
         api: str = None,
         base_url: str = "https://api.deepseek.com/v1",
         return_type: str = "answer",
         embedding_model: str = "./models/bge-m3",
         chunk_size: int = 512,
         chunk_overlap: int = 64,
         enable_hybrid_search: bool = True,
         enable_query_expansion: bool = True,
         retrieval_mode: str = "gkgr",
         bert_model_path: str = "pretrain_model.pth",
         force_reinit: bool = False,
         **kwargs) -> Union[str, List[Dict], Dict]:
    
    instance_key = CDDRS._get_instance_key(source_knowledge_base, llm, api or "default", base_url)
    
    if force_reinit or instance_key not in CDDRS._instances:
        try:
            retriever = CDDRS._initialize_retriever(
                source_knowledge_base=source_knowledge_base,
                llm=llm,
                api=api,
                base_url=base_url,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                enable_hybrid_search=enable_hybrid_search,
                enable_query_expansion=enable_query_expansion,
                retrieval_mode=retrieval_mode,
                bert_model_path=bert_model_path,
                **kwargs
            )
            
            CDDRS._instances[instance_key] = retriever
            
        except Exception as e:
            raise
    else:
        retriever = CDDRS._instances[instance_key]
    
    retriever.top_k = topk
    if hasattr(retriever, 'vector_retriever'):
        retriever.vector_retriever.similarity_top_k = topk
    if hasattr(retriever, 'kg_retriever'):
        retriever.kg_retriever.similarity_top_k = topk
    if hasattr(retriever, 'fusion_retriever'):
        retriever.fusion_retriever.similarity_top_k = topk
    
    try:
        if return_type == "documents":
            retrieved_docs = retriever.retrieve(query)
            
            formatted_docs = []
            for i, node_with_score in enumerate(retrieved_docs[:topk]):
                doc_info = {
                    "rank": i + 1,
                    "score": node_with_score.score,
                    "content": node_with_score.node.get_content(),
                    "metadata": node_with_score.node.metadata,
                    "node_id": node_with_score.node.node_id
                }
                formatted_docs.append(doc_info)
            
            return formatted_docs
            
        elif return_type == "both":
            retrieved_docs = retriever.retrieve(query)
            
            formatted_docs = []
            for i, node_with_score in enumerate(retrieved_docs[:topk]):
                doc_info = {
                    "rank": i + 1,
                    "score": node_with_score.score,
                    "content": node_with_score.node.get_content(),
                    "metadata": node_with_score.node.metadata,
                    "node_id": node_with_score.node.node_id
                }
                formatted_docs.append(doc_info)
            
            answer = retriever.query(query)
            
            result = {
                "answer": answer,
                "documents": formatted_docs,
                "query": query,
                "total_docs": len(formatted_docs),
                "retrieval_info": {
                    "topk": topk,
                    "llm": llm,
                    "retrieval_mode": retrieval_mode,
                    "hybrid_search": enable_hybrid_search,
                    "query_expansion": enable_query_expansion
                }
            }
            
            return result
            
        else:
            answer = retriever.query(query)
            return answer
            
    except Exception as e:
        if return_type == "documents":
            return []
        elif return_type == "both":
            return {
                "answer": f"Retrieval failed: {e}",
                "documents": [],
                "query": query,
                "total_docs": 0,
                "error": str(e)
            }
        else:
            return f"Retrieval failed: {e}"


def clear_gkgr_cache():
    CDDRS._instances.clear()


def list_gkgr_instances():
    return list(CDDRS._instances.keys())


def get_gkgr_config(source_knowledge_base: str,
                   llm: str = 'gpt-4o-mini',
                   api: str = None,
                   base_url: str = "https://api.deepseek.com/v1") -> Dict:
    instance_key = CDDRS._get_instance_key(source_knowledge_base, llm, api or "default", base_url)
    
    if instance_key not in CDDRS._instances:
        return {"error": "Instance not found"}
    
    retriever = CDDRS._instances[instance_key]
    
    config = {
        "knowledge_base": source_knowledge_base,
        "llm": llm,
        "base_url": base_url,
        "top_k": retriever.top_k,
        "hybrid_search": retriever.enable_hybrid_search,
        "query_expansion": retriever.enable_query_expansion,
        "retrieval_mode": retriever.retrieval_mode,
        "total_nodes": len(retriever.all_nodes) if hasattr(retriever, 'all_nodes') else 0,
        "fusion_weights": retriever.fusion_weights if hasattr(retriever, 'fusion_weights') else None,
        "expansion_weights": retriever.expansion_weights if hasattr(retriever, 'expansion_weights') else None
    }
    
    return config


def update_gkgr_config(source_knowledge_base: str,
                      llm: str = 'gpt-4o-mini',
                      api: str = None,
                      base_url: str = "https://api.deepseek.com/v1",
                      **config_updates) -> bool:
    instance_key = CDDRS._get_instance_key(source_knowledge_base, llm, api or "default", base_url)
    
    if instance_key not in CDDRS._instances:
        return False
    
    retriever = CDDRS._instances[instance_key]
    
    try:
        for key, value in config_updates.items():
            if hasattr(retriever, key):
                setattr(retriever, key, value)
        
        if 'retrieval_mode' in config_updates and hasattr(retriever, 'set_retrieval_mode'):
            retriever.set_retrieval_mode(config_updates['retrieval_mode'])
        
        if 'expansion_weights' in config_updates and hasattr(retriever, 'set_expansion_weights'):
            retriever.set_expansion_weights(config_updates['expansion_weights'])
        
        return True
        
    except Exception as e:
        return False


def bert_expand_query(query: str, 
                     model_path: str = 'pretrain_model.pth',
                     tokenizer_name: str = 'bert-base-chinese') -> List[str]:
    try:
        expander = BERTQueryExpander(model_path=model_path, tokenizer_name=tokenizer_name)
        return expander.expand(query)
    except Exception as e:
        return [query]
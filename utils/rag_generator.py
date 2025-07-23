from typing import List, Optional, Any, Dict, Union
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core import Settings
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.response.schema import Response, StreamingResponse
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.llms.types import ChatMessage, MessageRole


DEFAULT_QA_PROMPT_TEMPLATE = """
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer: """

DEFAULT_QA_PROMPT = PromptTemplate(DEFAULT_QA_PROMPT_TEMPLATE)


class RAGGenerator(BaseQueryEngine):
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Optional[Any] = None,
        prompt_template: Optional[BasePromptTemplate] = None,
        top_k: int = 5,
        response_mode: str = "compact",
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
    ):
        self._retriever = retriever
        self._llm = llm or Settings.llm
        self._prompt_template = prompt_template or DEFAULT_QA_PROMPT
        self._top_k = top_k
        self._response_mode = response_mode
        self._verbose = verbose
        
        if hasattr(self._retriever, 'similarity_top_k'):
            self._retriever.similarity_top_k = top_k
        
        super().__init__(callback_manager=callback_manager)
    
    @property
    def retriever(self) -> BaseRetriever:
        return self._retriever
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        try:
            nodes = self._retriever.retrieve(query_bundle)
            return nodes[:self._top_k]
        except Exception as e:
            if self._verbose:
                print(f"Retrieval error: {e}")
            return []
    
    def _format_context(self, nodes: List[NodeWithScore]) -> str:
        if not nodes:
            return ""
        
        if self._response_mode == "compact":
            context_parts = []
            for i, node in enumerate(nodes):
                context_parts.append(f"[Document {i+1}]\n{node.node.get_content()}")
            return "\n\n".join(context_parts)
        else:
            return "\n\n".join([node.node.get_content() for node in nodes])
    
    def _query(self, query_bundle: QueryBundle) -> Response:
        retrieved_nodes = self._retrieve(query_bundle)
        
        if self._verbose:
            print(f"Retrieved {len(retrieved_nodes)} documents")
        
        context_str = self._format_context(retrieved_nodes)
        
        prompt = self._prompt_template.format(
            context_str=context_str,
            query_str=query_bundle.query_str
        )
        
        if self._verbose:
            print(f"Generated prompt: {prompt}")
        
        try:
            if hasattr(self._llm, 'complete'):
                llm_response = self._llm.complete(prompt)
                response_text = str(llm_response)
            elif hasattr(self._llm, 'chat'):
                messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
                llm_response = self._llm.chat(messages)
                response_text = str(llm_response.message.content)
            else:
                llm_response = self._llm(prompt)
                response_text = str(llm_response)
        except Exception as e:
            if self._verbose:
                print(f"LLM generation error: {e}")
            response_text = "Error generating response"
        
        response = Response(
            response=response_text,
            source_nodes=retrieved_nodes,
        )
        
        return response
    
    def _aquery(self, query_bundle: QueryBundle) -> Any:
        raise NotImplementedError("Async query not implemented")
    
    def query(self, query_str: str) -> Response:
        query_bundle = QueryBundle(query_str=query_str)
        return self._query(query_bundle)
    
    def generate(self, question: str, context: str = "") -> str:
        if context:
            custom_template = f"""
Additional context: {context}

Retrieved context information is below.
---------------------
{{context_str}}
---------------------
Given both the additional context and retrieved context information, answer the query.
Query: {{query_str}}
Answer: """
            custom_prompt = PromptTemplate(custom_template)
            
            original_prompt = self._prompt_template
            self._prompt_template = custom_prompt
            
            try:
                response = self.query(question)
                answer = str(response.response)
            finally:
                self._prompt_template = original_prompt
            
            return answer
        else:
            response = self.query(question)
            return str(response.response)
    
    def get_retrieved_documents(self, query_str: str) -> List[NodeWithScore]:
        query_bundle = QueryBundle(query_str=query_str)
        return self._retrieve(query_bundle)
    
    def update_retriever(self, retriever: BaseRetriever) -> None:
        self._retriever = retriever
        if hasattr(self._retriever, 'similarity_top_k'):
            self._retriever.similarity_top_k = self._top_k
    
    def update_top_k(self, top_k: int) -> None:
        self._top_k = top_k
        if hasattr(self._retriever, 'similarity_top_k'):
            self._retriever.similarity_top_k = top_k
    
    def update_prompt_template(self, prompt_template: BasePromptTemplate) -> None:
        self._prompt_template = prompt_template


class SimpleRAGGenerator:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Optional[Any] = None,
        top_k: int = 5,
        prompt_template: str = None,
        verbose: bool = False,
    ):
        self.retriever = retriever
        self.llm = llm or Settings.llm
        self.top_k = top_k
        self.verbose = verbose
        
        if prompt_template is None:
            self.prompt_template = """Context information:
{context}

Question: {question}
Answer: """
        else:
            self.prompt_template = prompt_template
        
        if hasattr(self.retriever, 'similarity_top_k'):
            self.retriever.similarity_top_k = top_k
    
    def query(self, question: str) -> str:
        try:
            if hasattr(self.retriever, 'retrieve'):
                try:
                    from llama_index.core.schema import QueryBundle
                    query_bundle = QueryBundle(query_str=question)
                    retrieved_nodes = self.retriever.retrieve(query_bundle)
                except:
                    retrieved_nodes = self.retriever.retrieve(question)
            else:
                retrieved_nodes = []
                
            if self.verbose:
                print(f"Retrieved {len(retrieved_nodes)} documents")
                
        except Exception as e:
            if self.verbose:
                print(f"Retrieval error: {e}")
            retrieved_nodes = []
        
        if retrieved_nodes:
            context_parts = []
            for i, node in enumerate(retrieved_nodes[:self.top_k]):
                if hasattr(node, 'node'):
                    content = node.node.get_content()
                elif hasattr(node, 'get_content'):
                    content = node.get_content()
                else:
                    content = str(node)
                context_parts.append(f"Document {i+1}: {content}")
            context = "\n\n".join(context_parts)
        else:
            context = "No relevant context found."
        
        prompt = self.prompt_template.format(context=context, question=question)
        
        if self.verbose:
            print(f"Generated prompt: {prompt}")
        
        try:
            if hasattr(self.llm, 'complete'):
                response = self.llm.complete(prompt)
                return str(response)
            elif hasattr(self.llm, 'chat'):
                from llama_index.core.base.llms.types import ChatMessage, MessageRole
                messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
                response = self.llm.chat(messages)
                return str(response.message.content)
            elif callable(self.llm):
                response = self.llm(prompt)
                return str(response)
            else:
                return "Error: LLM not properly configured"
        except Exception as e:
            if self.verbose:
                print(f"Generation error: {e}")
            return "Error generating response"
    
    def generate(self, question: str, context: str = "") -> str:
        if context:
            original_template = self.prompt_template
            self.prompt_template = f"""Additional Context: {context}

Retrieved Context:
{{context}}

Question: {{question}}
Answer: """
            
            try:
                answer = self.query(question)
            finally:
                self.prompt_template = original_template
            
            return answer
        else:
            return self.query(question)


def create_rag_generator(retriever: BaseRetriever,
                        llm: Optional[Any] = None,
                        top_k: int = 5,
                        prompt_template: Optional[str] = None,
                        generator_type: str = "simple",
                        verbose: bool = False) -> Union[RAGGenerator, SimpleRAGGenerator]:
    if generator_type == "advanced":
        if prompt_template:
            from llama_index.core.prompts import PromptTemplate
            template = PromptTemplate(prompt_template)
        else:
            template = None
        
        return RAGGenerator(
            retriever=retriever,
            llm=llm,
            prompt_template=template,
            top_k=top_k,
            verbose=verbose
        )
    else:
        return SimpleRAGGenerator(
            retriever=retriever,
            llm=llm,
            top_k=top_k,
            prompt_template=prompt_template,
            verbose=verbose
        )
import logging
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import os
import chromadb
import torch
from helpers import fetch_wikipedia_pages, search_wikipedia
from transformers import BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WikiRAG:
    """
    A class for implementing a Wikipedia-based Retrieval-Augmented Generation (RAG) system.
    """

    def __init__(self, config_dict) -> None:
        """
        Initialize the WikiRAG instance.

        Args:
            config_dict (dict): Configuration parameters for the RAG system.
        """
        self.config = config_dict
        logging.info(f"Initializing WikiRAG with config: {self.config}")
        self._build_pipeline()

    def _build_pipeline(self):
        """
        Build the RAG pipeline, including embedding model, LLM, and index.
        """
        self._setup_embedding_model()
        self._setup_llm()
        self._setup_index()
        self.query_engine = self._build_query_engine()

    def _setup_embedding_model(self):
        """Set up the embedding model."""
        self.embedding_model = HuggingFaceEmbedding(model_name=self.config["embedding_model"], cache_folder='./hf_cache')

    def _setup_llm(self):
        """Set up the language model."""
        kwargs = {
            "token": os.environ['HF_TOKEN'],
            "torch_dtype": torch.bfloat16,
        }
        if self.config['quantized']:
            # Configure quantization if enabled
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            del kwargs['torch_dtype']
            kwargs["quantization_config"] = quantization_config

        self.llm = HuggingFaceLLM(
            model_name=self.config['llm'],
            tokenizer_name=self.config['llm'],
            model_kwargs=kwargs,
            tokenizer_kwargs={"token": os.environ['HF_TOKEN']},
            generate_kwargs={
                "do_sample": True,
                "temperature": self.config['temperature']
            }
        )

    def _setup_index(self):
        """Set up or load the vector index."""
        if not os.path.exists(self.config['index_path']):
            self.index = self._build_index(self.config['index_path'])
        else:
            self.index = self._load_index(self.config['index_path'])

    def _build_index(self, index_path):
        """
        Build a new vector index.

        Args:
            index_path (str): Path to store the index.

        Returns:
            VectorStoreIndex: The created vector index.
        """
        logging.info("Building the vector index")
        db = chromadb.PersistentClient(path=index_path)
        chroma_collection = db.get_or_create_collection("wikipedia-pages")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self.embedding_model,
            transformations=[SentenceSplitter(chunk_size=self.config['chunk_size'], chunk_overlap=self.config['chunk_overlap'])]
        )
        vector_index.storage_context.persist(persist_dir=index_path)
        return vector_index

    def _load_index(self, index_path):
        """
        Load an existing vector index from disk.

        Args:
            index_path (str): Path to the stored index.

        Returns:
            VectorStoreIndex: The loaded vector index.
        """
        logging.info("Loading index from disk")
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        return index

    def _build_query_engine(self):
        """
        Build the query engine for RAG.

        Returns:
            QueryEngine: The configured query engine.
        """
        return self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=self.config['similarity_k'],
            response_mode="compact",
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=self.config['similarity_cutoff'])]
        )

    def query(self, prompt):
        """
        Execute a RAG query.

        Args:
            prompt (str): The query prompt.

        Returns:
            tuple: A tuple containing the response and the contexts used.
        """
        wikipedia_links = search_wikipedia(prompt, n_articles=self.config['n_articles'])
        page_titles = fetch_wikipedia_pages(wikipedia_links, self.index)
        logging.info(f"Fetched the following Wikipedia pages: {page_titles}")
        response = self.query_engine.query(prompt)
        contexts = [node.dict()['node']['text'] for node in response.source_nodes]
        return response.response, contexts
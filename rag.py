from llama_index.core.readers import SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import  VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import os
import chromadb
import torch
from helpers import fetch_wikipedia_pages, search_wikipedia
from transformers import BitsAndBytesConfig


class WikiRAG:
    def __init__(self, config_dict) -> None:
        self.config = config_dict
        print(self.config)
        self._build_pipleine()


    def _build_pipleine(self):
        self.embedding_model = HuggingFaceEmbedding(model_name=self.config["embedding_model"],cache_folder='./hf_cache') 
        kwargs = {
            "token": os.environ['HF_TOKEN'],
            "torch_dtype": torch.bfloat16, 
        }

        if self.config['quantized']:
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
            model_kwargs = kwargs,
            tokenizer_kwargs={"token": os.environ['HF_TOKEN']},
        )
        if not os.path.exists(self.config['index_path']):
            self.index = self.build_index(self.config['index_path'])
        else:
            self.index = self.load_index(self.config['index_path'])


        self.query_engine = self.build_query_engine(k = self.config['k'])

    def build_index(self, index_path):
        print("building the vector index")
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
    
    def load_index(self, index_path):
        print("loading index from disk")
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        return index
    
    def build_query_engine(self):
        return self.index.as_query_engine(llm=self.llm,
                                        similarity_top_k=self.config['similarity_k'],
                                        response_mode="compact",
                                        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=self.config['similarity_cutoff'])])
    def query(self, prompt):
        # refresh the vector index and then answer the query
        wikipedia_links = search_wikipedia(prompt,n_articles=self.config['n_articles'])
        page_titles = fetch_wikipedia_pages(wikipedia_links,self.index)
        print(f"fetched the following wikipedia pages: {page_titles}")
        response = self.query_engine.query(prompt)
        answer = response.text
        context = [node.dict()['node']['text'] for node in response.source_nodes]
        return answer,context
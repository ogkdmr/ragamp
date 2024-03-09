"""Code for Building and querying a RAG vector store index using an LLM.

This module contains code for querying a vector store index using a
language model and generating responses. It uses the HuggingFace library for
language model and tokenizer, and the llama_index library for vector store
index operations. The module also includes code for creating and loading the
index from storage, as well as saving the query responses to a JSON file.
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
import sys
import time
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor

import torch
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.schema import BaseNode
from llama_index.core.schema import Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


PERSIST_DIR = '/home/ac.ogokdemir/lucid_index'
PAPERS_DIR = '/rbstor/ac.ogokdemir/md_outs'
QUERY_DIR = '/home/ogokdemir/ragamp/examples/lucid_queries.txt'
OUTPUT_DIR = '/rbstor/ac.ogokdemir/ragamp/output/lucid/'
NODE_INFO_PATH = osp.join(OUTPUT_DIR, 'node_info.jsonl')

os.makedirs(OUTPUT_DIR, exist_ok=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
)

# TODO: Move the generator and encoder factories out.

# mistral7b = HuggingFaceLLM(
#     model_name="mistralai/Mistral-7B-Instruct-v0.1",
#     tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
#     query_wrapper_prompt=PromptTemplate(
#         "<s>[INST] {query_str} [/INST] </s>\n",
#     ),
#     context_window=32000,
#     max_new_tokens=1024,
#     model_kwargs={"quantization_config": quantization_config},
#     # tokenizer_kwargs={},
#     generate_kwargs={
#         "temperature": 0.2,
#         "top_k": 5,
#         "top_p": 0.95,
#         "do_sample": True,
#     },
#     device_map="auto",
# )

mixtral8x7b = HuggingFaceLLM(
    model_name='mistralai/Mixtral-8x7B-v0.1',
    tokenizer_name='mistralai/Mixtral-8x7B-v0.1',
    query_wrapper_prompt=PromptTemplate(
        '<s>[INST] {query_str} [/INST] </s>\n',
    ),
    context_window=32000,
    max_new_tokens=1024,
    model_kwargs={'quantization_config': quantization_config},
    # tokenizer_kwargs={},
    generate_kwargs={
        'temperature': 0.2,
        'top_k': 5,
        'top_p': 0.95,
        'do_sample': True,
    },
    device_map='auto',
)


# indexer = encoder + chunker.
def get_encoder(device: int = 0) -> BaseEmbedding:
    """Get the encoder for the vector store index."""
    return HuggingFaceEmbedding(
        model_name='pritamdeka/S-PubMedBert-MS-MARCO',
        tokenizer_name='pritamdeka/S-PubMedBert-MS-MARCO',
        max_length=512,
        embed_batch_size=64,
        cache_folder=os.environ.get('HF_HOME'),
        device=f'cuda:{device}',
    )


def get_splitter(encoder: BaseEmbedding) -> NodeParser:
    """Get the splitter for the vector store index."""
    return SemanticSplitterNodeParser(
        buffer_size=1,
        include_metadata=True,
        embed_model=encoder,
    )


def chunk_encode_unit(device: int, docs: list[Document]) -> list[BaseNode]:
    """Encode documents using the given embedding model."""
    # create the encoder
    encoder = get_encoder(device)
    splitter = get_splitter(encoder)
    return splitter.get_nodes_from_documents(docs)


def chunk_encode_parallel(
    docs: list[Document],
    num_workers: int = 8,
) -> list[BaseNode]:
    """Encode documents in parallel using the given embedding model."""
    batches = [(i, docs[i::num_workers]) for i in range(num_workers)]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(chunk_encode_unit, device, docs)  #
            for device, docs in batches
        ]
        results = [future.result() for future in as_completed(futures)]

    return [node for result in results for node in result]


if not osp.exists(PERSIST_DIR):
    logging.info('Creating index from scratch')

    start = time.time()
    documents = SimpleDirectoryReader(PAPERS_DIR).load_data()
    end = time.time()

    logging.info(f'Loaded documents in {end - start} seconds.')

    nodes = chunk_encode_parallel(documents, num_workers=8)

    # Code for visually inspecting the success of semantic chunking.
    with open(NODE_INFO_PATH, 'w') as f:
        for rank, node in enumerate(nodes, 1):
            node_info = {
                'rank': rank,
                'content': node.get_content(),
                'metadata': node.get_metadata_str(),
            }
            f.write(json.dumps(node_info) + '\n')

    index = VectorStoreIndex(
        nodes,
        embed_model=get_encoder(),  # for now,this has to be serial
        insert_batch_size=16384,
        show_progress=True,
        use_async=True,
    )

    os.makedirs(PERSIST_DIR)
    index.storage_context.persist(PERSIST_DIR)
    logging.info(f'Saved the new index to {PERSIST_DIR}')

else:
    logging.info('Loading index from storage')
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(PERSIST_DIR),
        vector_store=SimpleVectorStore.from_persist_dir(
            PERSIST_DIR,
            namespace='default',
        ),
        index_store=SimpleIndexStore.from_persist_dir(PERSIST_DIR),
    )
    index = load_index_from_storage(storage_context, embed_model=get_encoder())

    logging.info(f'Loaded the index from {PERSIST_DIR}.')

# TODO: Add these query engine information in a config file.
query_engine = index.as_query_engine(
    llm=mixtral8x7b,
    similarity_top_k=10,
    similarity_threshold=0.5,
)

logging.info('Query engine ready, running inference')

with open(QUERY_DIR) as f:
    queries = f.read().splitlines()

query_2_response = {}

for query in queries:
    query_2_response[query] = query_engine.query(query)

with open(osp.join(OUTPUT_DIR, 'query_responses.json'), 'w') as f:
    json.dump(query_2_response, f)

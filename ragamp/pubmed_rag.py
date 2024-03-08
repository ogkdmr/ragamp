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

import torch
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


PERSIST_DIR = '/lus/eagle/projects/LUCID/ogokdemir/ragamp/indexes/lucid_index/'
PAPERS_DIR = '/lus/eagle/projects/LUCID/ogokdemir/parsed_lucid_papers/md_outs'
QUERY_AMPS_DIR = '/home/ogokdemir/ragamp/examples/with_no_int.txt'
OUTPUT_DIR = '/lus/eagle/projects/LUCID/ogokdemir/ragamp/outputs/lucid/'
NODE_INFO_PATH = osp.join(OUTPUT_DIR, 'node_info.jsonl')

os.makedirs(OUTPUT_DIR, exist_ok=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
)

llm = HuggingFaceLLM(
    model_name='mistralai/Mistral-7B-Instruct-v0.1',
    tokenizer_name='mistralai/Mistral-7B-Instruct-v0.1',
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

encoder = HuggingFaceEmbedding(
    model_name='pritamdeka/S-PubMedBert-MS-MARCO',
    tokenizer_name='pritamdeka/S-PubMedBert-MS-MARCO',
    max_length=512,
    embed_batch_size=64,
    cache_folder='/lus/eagle/projects/LUCID/ogokdemir/hf_cache',
    device='cuda',
)

splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    include_metadata=True,
    embed_model=encoder,
)

if not osp.exists(PERSIST_DIR):
    logging.info('Creating index from scratch')

    documents = SimpleDirectoryReader(PAPERS_DIR).load_data()

    # TODO: Parallelize the call below to get_nodes_from_docs to all GPUS.

    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

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
        embed_model='local',
        insert_batch_size=16384,
        show_progress=True,
        use_async=True,
    )

    os.makedirs(PERSIST_DIR)
    index.storage_context.persist(PERSIST_DIR)
    logging.info('Saving the first 100 node content for investigation')

else:
    logging.info('Loading index from storage')
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context, embed_model=encoder)

logging.info('Built and saved the semantically chunked LUCID index.')

# TODO: Add these query engine information in a config file.
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=10,
    similarity_threshold=0.5,
)

logging.info('Query engine ready, running inference')

with open(QUERY_AMPS_DIR) as f:
    amps = f.read().splitlines()

PROMPTS = [
    'What bacterial strains does {} act on?',
    'What are the likely mechanism of actions the {} has?',
    'What are the gene targets of {}?',
    'What cellular processes does {} disrupt?',
]

# create a dictionary of dictionaries
q2r = {}

os.makedirs(OUTPUT_DIR, exist_ok=True)

for rank, prompt in enumerate(PROMPTS, 1):
    for amp in amps:
        query = prompt.format(amp)
        response = query_engine.query(query)
        q2r[amp] = str(response)

    out_filepath = osp.join(OUTPUT_DIR, f'template_{rank}.json')
    with open(out_filepath, 'w') as f:
        json.dump(q2r, f)
    q2r = {}

# TODO: Move dataloading and encoding to functions and parallelize them.
# TODO: Once that is done, build the index directly from the embeddings.

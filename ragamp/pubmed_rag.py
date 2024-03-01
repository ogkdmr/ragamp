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
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.huggingface import HuggingFaceLLM
from tqdm import tqdm

os.environ['HF_HOME'] = '/lus/eagle/projects/LUCID/ogokdemir/hf_cache'
from transformers import BitsAndBytesConfig  # noqa

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


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
    context_window=3900,
    max_new_tokens=256,
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

# TODO: pritamdeka/S-PubMedBert-MS-MARCO, look into this encoder alternative.

encoder = HuggingFaceBgeEmbeddings(
    model_name='pritamdeka/S-PubMedBert-MS-MARCO',
)

PERSIST_DIR = '/lus/eagle/projects/LUCID/ogokdemir/ragamp/indexes/amp_index/'
AMP_PAPERS_DIR = '/lus/eagle/projects/candle_aesp/ogokdemir/pdfwf_runs/AmpParsedDocs/md_outs/'  # noqa
QUERY_AMPS_DIR = '/home/ogokdemir/ragamp/examples/antimicrobial_peptides.txt'

if not osp.exists(PERSIST_DIR):
    logging.info('Creating index from scratch')
    documents = SimpleDirectoryReader(AMP_PAPERS_DIR).load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=encoder,
        show_progress=True,
    )
    index.storage_context.persist(PERSIST_DIR)
else:
    logging.info('Loading index from storage')
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(llm=llm)
logging.info('Query engine ready, running inference')

with open(QUERY_AMPS_DIR) as f:
    amps = f.read().splitlines()

q2r = {}
for amp in tqdm(amps, desc='Querying', total=len(amps)):
    query = f'What bacterial strains does {amp} act on?.'
    response = query_engine.query(query)
    q2r[amp] = str(response)

with open('data/query_responses_strains.json', 'w') as f:
    json.dump(q2r, f)

# TODO: Move dataloading and encoding to functions and parallelize them.
# TODO: Once that is done, build the index directly from the embeddings.

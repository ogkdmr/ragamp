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
from llama_index import load_index_from_storage
from llama_index import ServiceContext
from llama_index import set_global_service_context
from llama_index import SimpleDirectoryReader
from llama_index import StorageContext
from llama_index import VectorStoreIndex
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
from tqdm import tqdm
from transformers import BitsAndBytesConfig

os.environ['HF_HOME'] = '/lambda_stor/data/ogokdemir/transformers_cache'
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

embed_model = HuggingFaceBgeEmbeddings(
    model_name='dmis-lab/biobert-base-cased-v1.1',
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

set_global_service_context(service_context)

PERSIST_DIR = 'data/vectorstores'

if not osp.exists(PERSIST_DIR):
    logging.info('Creating index from scratch')
    documents = SimpleDirectoryReader('data/pmc').load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(PERSIST_DIR)
else:
    logging.info('Loading index from storage')
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()
logging.info('Query engine ready, running inference')

amps = [
    'Amoebapore A',
    'BACTENECIN 5',
    'CCL20',
    'DEFB118',
    'Drosomycin',
    'Eotaxin2',
    'Gm cecropin A',
    'Human alphasynuclein',
    'Human granulysin',
    'Microcin B',
    'Microcin S',
    'NLP31',
    'Amoebapore B',
    'BACTENECIN 7',
    'CXCL2',
    'DEFB24',
    'Drosomycin2',
    'Eotaxin3',
    'Gm cecropin B',
    'Human beta defensin 2',
    'Human histatin 9',
    'Microcin C7',
    'Microcin V',
    'Peptide 2',
    'Amoebapore C',
    'CAP18',
    'CXCL3',
    'Defensin 1',
    'Drosophila cecropin B',
    'EP2',
    'Gm cecropin C',
    'Human beta defensin 3',
    'Human TC2',
    'Microcin L',
    'NLP27',
    'Peptide 5',
    'Bactenecin',
    'Cathepsin G',
    'CXCL6',
    'Dermcidin',
    'Elafin',
    'FGG',
    'Gm defensinlike peptide',
    'Human beta defensin 4',
    'LL23',
    'Microcin M',
    'NLP29',
]

q2r = {}
for amp in tqdm(amps, desc='Querying', total=len(amps)):
    query = f'What bacterial strains does {amp} act on?.'
    response = query_engine.query(query)
    q2r[amp] = str(response)

with open('data/query_responses_strains.json', 'w') as f:
    json.dump(q2r, f)

# TODO: Find a way to customize the number of documents returned by
# the query engine. Currently, it returns 10 documents by default.

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

import faiss
import torch
from llama_index.core import get_response_synthesizer
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import BaseNode
from llama_index.core.schema import Document
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.faiss import FaissVectorStore
from transformers import BitsAndBytesConfig

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


PERSIST_DIR = '/rbstor/ac.ogokdemir/ragamp/indexes/lucid/faiss_index'
PAPERS_DIR = '/rbstor/ac.ogokdemir/md_outs'
QUERY_DIR = '/home/ac.ogokdemir/ragamp/examples/hypo_ol_parsed.jsonl'
OUTPUT_DIR = '/rbstor/ac.ogokdemir/ragamp/output/lucid_hypo_eval'
NODE_INFO_PATH = osp.join(OUTPUT_DIR, 'node_info.jsonl')

os.makedirs(OUTPUT_DIR, exist_ok=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
)

# TODO: Move the generator and encoder creation to factory functions.
# Create BaseGenerator and BaseEncoder interfaces make them take config.

# mistral7b = HuggingFaceLLM(
#     model_name="mistralai/Mistral-7B-Instruct-v0.1",
#     tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
#     # query_wrapper_prompt=PromptTemplate(
#     #     "<s>[INST] {query_str} [/INST] </s>\n",
#     # ),
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
    model_name='mistralai/Mixtral-8x7B-Instruct-v0.1',
    tokenizer_name='mistralai/Mixtral-8x7B-Instruct-v0.1',
    context_window=32000,
    max_new_tokens=4096,
    query_wrapper_prompt=PromptTemplate(
        '<s>[INST] {query_str} [/INST] </s>\n',
    ),
    model_kwargs={'quantization_config': quantization_config},
    generate_kwargs={
        'temperature': 0.2,
        'top_k': 5,
        'top_p': 0.95,
        'do_sample': False,
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
    """Takes list of documents and a GPU index and runs encoding on that GPU.

    Args:
        device (int): the GPU that will run this unit of work.
        docs (list[Document]): list of documents

    Returns:
        list[BaseNode]: list of nodes
    """
    # create the encoder
    encoder = get_encoder(device)
    splitter = get_splitter(encoder)
    return splitter.get_nodes_from_documents(docs)


def chunk_encode_parallel(
    docs: list[Document],
    num_workers: int = 8,
) -> list[BaseNode]:
    """Encode documents in parallel using the given embedding model.

    Args:
        docs (list[Document]): list of documents
        num_workers (int, optional): Number of GPUs on your system.

    Returns:
        list[BaseNode]: list of nodes

    """
    batches = [(i, docs[i::num_workers]) for i in range(num_workers)]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(chunk_encode_unit, device, docs)  #
            for device, docs in batches
        ]
        results = [future.result() for future in as_completed(futures)]

    return [node for result in results for node in result]


# TODO: Put this in a create index from scratch function.
if not osp.exists(PERSIST_DIR):
    logging.info('Creating index from scratch')
    logging.info('Starting to load the documents.')

    load_start = time.time()
    documents = SimpleDirectoryReader(PAPERS_DIR).load_data(show_progress=True)
    load_end = time.time()
    logging.info(
        f"""Finished loading the documents in {load_end - load_start} seconds.
                Starting to chunk and encode.""",
    )
    chunk_start = time.time()
    nodes = chunk_encode_parallel(documents, num_workers=8)
    chunk_end = time.time()

    logging.info(
        f"""Finished encoding in {chunk_end - chunk_start} seconds,
                creating Faiss index.""",
    )
    embed_dim = 768  # for pubmedbert
    faiss_index = faiss.IndexFlatL2(embed_dim)
    vector_store = FaissVectorStore(faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    pack_start = time.time()
    index = VectorStoreIndex(
        nodes,
        embed_model=get_encoder(),
        insert_batch_size=16384,
        use_async=True,
        storage_context=storage_context,
        show_progress=True,
    )
    pack_end = time.time()
    logging.info(
        f"""Finished packing the index in
                 {pack_end - pack_start} seconds.""",
    )

    # Code for visually inspecting the success of semantic chunking.
    with open(NODE_INFO_PATH, 'w') as f:
        for rank, node in enumerate(nodes, 1):
            node_info = {
                'rank': rank,
                'content': node.get_content(),
                'metadata': node.get_metadata_str(),
            }
            f.write(json.dumps(node_info) + '\n')

    os.makedirs(PERSIST_DIR)
    index.storage_context.persist(PERSIST_DIR)
    logging.info(f'Saved the new index to {PERSIST_DIR}')

# TODO: Put this in a "load_index_from_storage" function.
else:
    logging.info('Loading index from storage')

    vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=PERSIST_DIR,
    )

    index = load_index_from_storage(storage_context, embed_model=get_encoder())

    logging.info(f'Loaded the index from {PERSIST_DIR}.')

# TODO: Refactor these into query_engine creation and inference.


# Creating the query engine.
retriever = VectorIndexRetriever(
    index, embed_model=get_encoder(), similarity_top_k=30
)

ldr_prompt_template_str = (
    """<s> [INST] You are a super smart AI that knows about science. You follow
    directions and you are always truthful and concise in your responses.
    Below is an hypothesis submitted to your consideration.\n"""
    '---------------------\n'
    'Hypothesis: {query_str}\n'
    '---------------------\n'
    'Below is some context provided to assist you in your analysis.\n'
    '---------------------\n'
    'Context: {context_str}\n'
    '---------------------\n'
    """Based on your background knowledge and the context provided, please
       determine if this hypothesis could have some connection to low dose
       radiation biology.  If the answer is yes, please generate one or more
       specific conjectures of biological mechanisms, that could relate low
       dose radiation to the effects detailed in the hypothesis. Please be
       as specific as possible, by naming specific biological pathways, genes
       or proteins and their interactions. It is okay to speculate as long as
       you give reasons for your conjectures. Finally, please estimate to the
       best of your knowledge the likelihood of the hypothesis being true, and
       please give step by step reasoning for your answers.
       Answer:  [/INST] </s>"""
    ''
)

ldr_prompt_template = PromptTemplate(ldr_prompt_template_str)


prompt_helper = PromptHelper(
    context_window=32000,
    num_output=2048,
    chunk_overlap_ratio=0,
)

response_synthesizer = get_response_synthesizer(
    llm=mixtral8x7b,
    response_mode=ResponseMode.COMPACT,
    # prompt_helper=prompt_helper,
    text_qa_template=ldr_prompt_template,
)

# query_engine = RetrieverQueryEngine(
#     retriever=retriever,
#     node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
#     response_synthesizer=response_synthesizer,
# )

# print(query_engine.query("What is the meaning of life?"))
qe = index.as_query_engine(
    llm=mixtral8x7b,
    similarity_top_k=5,
    response_mode=ResponseMode.COMPACT,
    # prompt_helper=prompt_helper,
    text_qa_template=ldr_prompt_template,
    # response_synthesizer=response_synthesizer,
    # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.2)],
)

logging.info('Query engine ready, running inference')
response = qe.query(
    """The hypothesis in this study is that low doses of X-rays can induce
        a repair mechanism in human lymphocytes. This mechanism reduces
        the number of broken chromosome ends that can take part in aberration
        formation,even in the case of high-LET radiation from radon. This
        adaptive response is suggested to be a general phenomenon, not
        specific to X-rays, and potentially applicable to other types of
        radiation. The hypothesis is tested through experiments involving
        different types of radiation, varying doses and timing of low-dose
        X-ray exposure, and examining different cell types."""
)

print(response.response)
print('-' * 20)
print(response.metadata)
print('-' * 20)
print(response.source_nodes)
print('-' * 20)

sys.exit()

# # TODO: Make this query processing a generic function.

# with open(QUERY_DIR) as f:
#     queries = [json.loads(line) for line in f.readlines()]
#     for query in queries:
#         question = query['output']  # e.g., hypothesis
#         query_source_file = query['source']
#         response_dict = query_engine.query(question)
#         answer = response_dict.response  # type: ignore
#         metadata = response_dict.metadata
#         context_sources = response_dict.source_nodes

#         out = {
#             'query': question,
#             'query_source_file': query_source_file,
#             'response': answer,
#             'metadata': metadata,
#             'source_nodes': context_sources,
#         }

#         with open(
#           osp.join(OUTPUT_DIR, f'{query_source_file}_hypo_ol_rag.json'), 'w'
#         ) as f:
#             json.dump(out, f)

"""Encoder and splitter factories."""
from __future__ import annotations

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.node_parser.interface import NodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ragamp.utils import BaseModel


class EncoderConfig(BaseModel):
    """Configuration for the encoder."""

    model_name: str
    tokenizer_name: str
    max_length: int
    embed_batch_size: int
    cache_folder: str


# TODO: Maybe separate the Splitter to its own module later.
class SplitterConfig(BaseModel):
    """Configuration for the splitter."""

    buffer_size: int
    include_metadata: bool


def get_encoder(encoder_cfg: EncoderConfig, device: int = 0) -> BaseEmbedding:
    """Factory for the encoder to build the vector store."""
    return HuggingFaceEmbedding(
        device=f'cuda:{device}', **encoder_cfg.model_dump()
    )


def get_splitter(
    splitter_config: SplitterConfig, encoder: BaseEmbedding
) -> NodeParser:
    """Factory for the splitter to chunk the documents and encode them."""
    # note: embed_model is needed to determine semantic cutoff of chunks.
    return SemanticSplitterNodeParser(
        embed_model=encoder, **splitter_config.model_dump()
    )

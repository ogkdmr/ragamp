from __future__ import annotations

from argparse import ArgumentParser

from ragamp.encoder.get_encoder import EncoderConfig
from ragamp.encoder.get_encoder import get_encoder
from ragamp.encoder.get_encoder import get_splitter
from ragamp.encoder.get_encoder import SplitterConfig
from ragamp.utils import BaseModel


class TestConfig(BaseModel):
    """Test config that encapsulates EncoderConfig and SplitterConfig."""

    encoder_settings: EncoderConfig
    splitter_settings: SplitterConfig


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--config', '-c', type=str, required=True, help='Config file'
    )
    args = parser.parse_args()
    cfg = args.config
    test_cfg = TestConfig.from_yaml(cfg)
    encoder_cfg = test_cfg.encoder_settings
    splitter_cfg = test_cfg.splitter_settings

    # Build the encoder
    encoder = get_encoder(encoder_cfg, device=4)

    # Build the splitter
    splitter = get_splitter(splitter_cfg, encoder)

    print(type(encoder))
    print(type(splitter))

"""Utility functions for parsing RAG queries from HYPO files.

see:
    /rbstor/ac.ogokdemir/ricks_work
    /lus/eagle/projects/LUCID/ogokdemir/ricks_work
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any
from typing import Callable


def parse_hypotheses(hypo_filepath: Path) -> tuple[Path, str] | None:
    """Parse the hypothesis out of an <paper>.hypo_ol file.

    Usage: Choose the 'hypo_ol' --function (-f) in the CLI arguments
    to run this function.

    Args:
        hypo_filepath (Path): The path to the .hypo_ol file

    Returns:
        tuple[Path, str] | None: Filepath and the hypothesis,
        if one could be extracted.
    """
    pattern = re.compile(r'(?i)(hypothesis:? ?)(.*)')

    with open(hypo_filepath) as f:
        content = f.read()
        match = pattern.search(content)
        if match:
            hypothesis = match.group(2).strip()
            return hypo_filepath, hypothesis
        else:
            print(f'Could not find hypothesis in {hypo_filepath}')
            return None


def parallelize_function(
    func: Callable[..., tuple[Path, str] | None],
    func_inputs: list[Path],
    num_workers: int,
    **func_kwargs: dict[str, Any],
) -> list[tuple[Path, str]]:
    """Parallelize a function over a list of arguments.

    Args:
        func (function): The function to parallelize.
        func_inputs (list[Path]): The list of filepaths to process.
        num_workers (int, optional): The number parallel threads.
        func_kwargs: (dict) keyword arguments to pass to the function.

    Returns:
        list[tuple[Path, str]]: The results collected from all function calls.
    """
    partial_func = partial(func, **func_kwargs)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(partial_func, func_inputs))
    return [result for result in results if result is not None]


if __name__ == '__main__':
    parser = ArgumentParser(
        'Utility functions for parsing RAG queries from HYPO files.'
    )
    parser.add_argument(
        '--function',
        '-f',
        type=str,
        required=True,
        choices=['hypo_ol'],
        help='The function to run.',
    )

    parser.add_argument(
        '--input_dir',
        '-i',
        type=Path,
        required=True,
        help='The path to the files to parse.',
    )

    parser.add_argument(
        '--output_dir',
        '-o',
        type=Path,
        required=True,
        help='Directory to write the output jsonl file.',
    )

    parser.add_argument(
        '--num_workers',
        '-n',
        type=int,
        default=64,
        help='Number of parallel threads.',
    )

    parser.add_argument(
        '--test_mode',
        '-t',
        default=False,
        action='store_true',
        help='Dry run on first 100 files.',
    )

    args = parser.parse_args()

    match args.function:
        case 'hypo_ol':
            hypo_files = args.input_dir.glob('*.hypo_ol')
            if args.test_mode:
                hypo_files = list(hypo_files)[:100]
            results = parallelize_function(
                parse_hypotheses, hypo_files, args.num_workers
            )
            with open(
                args.output_dir / f'{args.function}_parsed.jsonl', 'w'
            ) as f:
                # write jsonl lines
                for source, hypothesis in results:
                    f.write(
                        json.dumps(
                            {'source': source.stem, 'output': hypothesis}
                        )
                        + '\n'
                    )

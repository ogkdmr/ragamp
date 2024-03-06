"""Initial code for reading the content of the json formatted RAG response."""

from __future__ import annotations

import json

with open('data/query_responses_strains.json') as f:
    q2r = json.load(f)
    for k, v in q2r.items():
        print('Query AMP: ', k)
        print()
        print('  Response: ', v)
        print()

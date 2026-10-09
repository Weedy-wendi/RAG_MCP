"""Deterministic RAG test-corpus generator.

Builds 100 synthetic PDFs that mimic real-world document families
(academic papers, technical manuals, financial reports, legal texts,
medical documents, news magazines) together with a machine-checkable
answer bank.

Entry points
------------
``build_corpus.py``   generate the corpus + answer bank
``verify_corpus.py``  acceptance gate over a generated corpus
``coverage.py``       coverage matrix computation
``depcheck.py``       dependency guard and API probe
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"

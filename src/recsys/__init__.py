"""Real-time recommendation serving system.

Two-stage retrieval-and-ranking recommender served under a latency budget.
Offline path trains models and builds the ANN index; online path answers
requests. See README.md for the architecture diagram.
"""

__version__ = "0.1.0"

"""Metadata-only alias package.

mlx-Yue commit ab0f058 renamed its distribution from ``lyra-yue2`` to ``mlx-yue`` but
``lyra.pipeline`` and ``lyra.commands`` still call ``importlib.metadata.version("lyra-yue2")``.
Installing this empty distribution makes that lookup succeed without modifying mlx-Yue.
"""

__version__ = "0.1.0"

"""Precompute GSGStain cell graphs.

Example:
    python build_graphs.py --dataroot ./datasets/he2ihc --phase train
"""

from models.graph_construction import main


if __name__ == '__main__':
    main()

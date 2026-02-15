# =============================================================================
# File: ViVit/__init__.py
# Purpose: ViViT module implementation used by legacy/alternative training scripts.
# Prerequisites: Imported by scripts that select ViViT-based temporal modeling.
# Usually Run Next: Used indirectly via training/testing entrypoints.
# Model Components Used: Defines ViViT and its attention/feedforward blocks.
# =============================================================================


from ViVit.vivit import ViViT
# from ViVit.module import Attention, PreNorm, FeedForward, Upsample
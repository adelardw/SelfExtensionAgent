"""Public entry point for the shared mathematics, algorithms and software engine.

The historical src.coding import path stays compatible; this is the same loop,
not another agent pipeline.
"""
from src.coding.harness import CodingHarness as Harness
__all__=['Harness']

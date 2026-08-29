"""IKKEM model package.

The package intentionally does not import solver-backed model modules at import
time. Generic workload and postprocessing helpers can therefore be inspected
without loading a solver or a configuration provider.
"""

__version__ = "1.0.0"

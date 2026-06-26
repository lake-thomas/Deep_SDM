"""Modular Host-NAIP-SDM dataset builder implementation.

The package entrypoint is ``python -m host_dataset_builder``. The builder is
designed to make reproducible presence/background datasets for Deep_SDM from
filtered iNaturalist/GBIF occurrence tables, NAIP chips, WorldClim/GHM scalar
predictors, and optional 3DEP topographic chips or scalar summaries.
"""

from .pipeline import main, process_species, print_plan_only

__all__ = ["main", "process_species", "print_plan_only"]

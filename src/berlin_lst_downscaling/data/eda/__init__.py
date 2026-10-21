"""Exploratory tables for the published Berlin LST training release.

The module reads published artifacts and writes descriptive tables. It does
not change the training release, the feature stacks, or the Stadtstruktur
grouping.
"""

from berlin_lst_downscaling.data.eda.rules import RELEASE_ID

__all__ = ["RELEASE_ID"]

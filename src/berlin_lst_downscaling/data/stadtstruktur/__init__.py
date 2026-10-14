"""Berlin urban-structure evaluation layer.

Source, purpose, grain
----------------------
Source: Umweltatlas Berlin 06.07 / 06.08 on the ISU5 1:5,000 block map,
accessed from the Berlin GDI WFS. This package does not add a model input.
Purpose: publish the official classes for descriptive statistics and
stratified evaluation. One source row is one ISU5 block or block part.
"""

from berlin_lst_downscaling.data.stadtstruktur.sources import RELEASE_ID

__all__ = ["RELEASE_ID"]

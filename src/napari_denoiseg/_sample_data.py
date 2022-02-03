"""
This module is an example of a barebones sample data provider for napari.
It implements the "sample data" specification.
see: https://napari.org/plugins/stable/npe2_manifest_specification.html
Replace code below according to your needs.
"""
from __future__ import annotations

from napari.types import LayerDataTuple, LabelsData


def denoiseg_sample() -> LayerDataTuple:
    return None


def denoiseg_labels() -> LabelsData:
    return None

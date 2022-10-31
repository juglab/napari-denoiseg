import pytest

from napari_denoiseg import (
    denoiseg_data_2D_n0,
    denoiseg_data_2D_n10,
    denoiseg_data_2D_n20,
    denoiseg_data_3D_n10,
    denoiseg_data_3D_n20
)


@pytest.mark.slow
@pytest.mark.parametrize('load_dataset', [denoiseg_data_2D_n0,
                                          denoiseg_data_2D_n10,
                                          denoiseg_data_2D_n20,
                                          denoiseg_data_3D_n10,
                                          denoiseg_data_3D_n20])
def test_load_data(load_dataset):
    data = load_dataset()
    assert len(data) == 2

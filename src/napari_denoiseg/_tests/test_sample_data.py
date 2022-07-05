import pytest

from napari_denoiseg import denoiseg_data_2D_n0, denoiseg_data_2D_n10, denoiseg_data_2D_n20


@pytest.mark.parametrize('load_dataset', [denoiseg_data_2D_n0,
                                     denoiseg_data_2D_n10,
                                     denoiseg_data_2D_n20])
def load_data(load_dataset):
    data = load_dataset()
    assert len(data) == 2
    assert data[0][0].shape == data[1][0].shape
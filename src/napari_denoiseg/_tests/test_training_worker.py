import pytest
import numpy as np

from napari_denoiseg.utils.training_worker import sanity_check_validation_fraction


@pytest.mark.parametrize('fraction', [0.05, 0.1, 0.5, 1])
def test_sanity_check_high_validation_fraction(fraction):
    n = 5
    m = int(n / fraction)
    X_train = np.zeros((m, 8, 8, 1))
    X_val = np.zeros((n, 8, 8, 1))

    # check validation fraction
    sanity_check_validation_fraction(X_train, X_val)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize('fraction', [0.01, 0.02, 0.04, 0.048])
def test_sanity_check_low_validation_fraction(fraction):
    n = 5
    m = int(n / fraction)
    X_train = np.zeros((m, 8, 8, 1))
    X_val = np.zeros((n, 8, 8, 1))

    with pytest.raises(UserWarning):
        sanity_check_validation_fraction(X_train, X_val)



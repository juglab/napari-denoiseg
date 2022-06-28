import pytest

from napari_denoiseg.widgets import AxesWidget


@pytest.mark.parametrize('shape_length', [i for i in range(2, 6)])
def test_axes_widget_no_Z_defaults(shape_length):
    widget = AxesWidget(shape_length, False)

    assert widget.is_valid()


@pytest.mark.parametrize('shape_length', [i for i in range(3, 7)])
def test_axes_widget_Z_defaults(shape_length):
    widget = AxesWidget(shape_length, True)

    assert widget.is_valid()


@pytest.mark.parametrize('shape_length, is_3D', [(2, True), (6, False)])
def test_axes_widget_invalid_defaults(shape_length, is_3D):
    widget = AxesWidget(shape_length, is_3D)

    assert not widget.is_valid()


def test_axes_widget_change_dims():
    widget = AxesWidget(6, False)  # default text is now invalid, regardless of is_3D or n_axes
    assert not widget.is_valid()  # cannot be valid with n=6 and no 3D

    # change is_3D
    widget.update_is_3D(True)

    # change text
    widget.text_field.setText(widget._set_default_text())
    assert widget.is_valid()


def test_axes_widget_change_dims():
    widget = AxesWidget(4, True)
    assert widget.is_valid()  # default is valid

    # change n
    widget.update_axes_number(5)
    assert not widget.is_valid()   # text is not valid anymore

    # change text
    widget.text_field.setText('ZTSYX')  # valid text
    assert widget.is_valid()

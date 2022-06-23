from copy import copy

import numpy as np

from napari_denoiseg import _patch_widget
import pytest


@pytest.fixture
def image2d():
    x_1, y_1 = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
    return x_1


@pytest.fixture
def image3d():
    x_1, y_1 = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
    img = x_1[np.newaxis, :, :]
    return img


def test_sanitize_vertex():
    vertex = np.array([3, 5], np.int8)
    sanitized_vertex = _patch_widget.sanitize_vertex(copy(vertex), 0, 10, 0, 10)
    assert all(np.equal(vertex, sanitized_vertex))
    sanitized_vertex = _patch_widget.sanitize_vertex(copy(vertex), 0, 2, 0, 2)
    expected = np.array([2, 2], np.int8)
    assert all(np.equal(expected, sanitized_vertex))
    sanitized_vertex = _patch_widget.sanitize_vertex(copy(vertex), 4, 10, 6, 10)
    expected = np.array([4, 6], np.int8)
    assert all(np.equal(expected, sanitized_vertex))


def test_sanitize_rectangle():
    rectangle = np.array([[-3, -3], [-3, 50], [50, 50], [50, -3]], np.int8)
    sanitized_rectangle = _patch_widget.sanitize_rectangle(copy(rectangle), (32, 32), 8)
    expected = np.array([[0, 0], [8, 24], [32, 32], [24, 8]], np.int8)
    assert np.array_equal(expected, sanitized_rectangle)
    sanitized_rectangle = _patch_widget.sanitize_rectangle(copy(rectangle), (2, 32, 32), 8)
    assert np.array_equal(expected, sanitized_rectangle)


def test_create_rectangle():
    expected_rectangle = [[0, 0], [8, 0], [8, 8], [0, 8]]
    rectangle = _patch_widget.create_rectangle(np.array([4, 4]), 8, (32, 32), (4, 4))
    assert np.array_equal(expected_rectangle, rectangle)


def test_slice_img_patch2D(image2d):
    rectangle = [[0, 0], [4, 0], [4, 4], [0, 4]]
    x_1, y_1 = np.meshgrid(np.linspace(0, 3, 4), np.linspace(0, 3, 4))
    img = _patch_widget.slice_img_patch(image2d, rectangle, 2)
    assert np.array_equal(x_1, img)


def test_slice_img_patch3D(image3d):
    rectangle = [[0, 0, 0], [0, 4, 0], [0, 4, 4], [0, 0, 4]]
    x_1, y_1 = np.meshgrid(np.linspace(0, 3, 4), np.linspace(0, 3, 4))
    img = _patch_widget.slice_img_patch(image3d, rectangle, 3)
    assert np.array_equal(x_1, img)

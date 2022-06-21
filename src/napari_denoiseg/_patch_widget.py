from copy import copy

import napari
from magicgui import magic_factory, magicgui
import numpy as np


@magicgui(call_button="Export",
          img_layer={'label': 'Layer for ROI picking'},
          patch_size={"widget_type": "Slider", "min": 16, "max": 512, "step": 8, 'value': 16, 'label': 'Patch size'},
          active={'widget_type': 'CheckBox', 'label': 'Start/Stop patch selection', 'visible': True, 'value': False},
          store_path={'widget_type': 'FileEdit', 'mode': 'd', 'label': 'Export path'},
          auto_call=False)
def patch_creation(img_layer: "napari.layers.Image", patch_size: int = 16, active: bool = False,
                   store_path: str = None):
    viewer = napari.current_viewer()
    selection_layer = viewer.layers["2dselection"]
    shapes = selection_layer.data
    patches = [slice_img_patch(x, ndim=len(patch_creation.img_layer.value.data.shape)) for x in shapes]
    from PIL import Image
    for idx, patch in enumerate(patches):
        im = Image.fromarray(patch)
        im.save(str(store_path) + "/" + str(idx) + ".tif", format="TIFF")


@patch_creation.active.changed.connect
def start_stop_selection(state):
    viewer = napari.current_viewer()
    if "highlight" not in viewer.layers:
        highlight_layer = viewer.add_shapes(name="highlight")
        viewer.add_shapes(name="2dselection", ndim=len(patch_creation.img_layer.value.data.shape))
        viewer.layers.selection.select_only(highlight_layer)
    highlight_layer = viewer.layers["highlight"]
    if state:
        highlight_layer.mouse_move_callbacks.append(draw_square)
        highlight_layer.mouse_drag_callbacks.append(create_patch)
    else:
        try:
            highlight_layer.mouse_move_callbacks.remove(draw_square)
            highlight_layer.mouse_drag_callbacks.remove(create_patch)
        except ValueError:
            pass  # do nothing!


def create_patch(layer, event):
    if event.type == "mouse_press" and event.button == 1:
        viewer = napari.current_viewer()
        cords = np.round(layer.world_to_data(viewer.cursor.position)).astype(int)
        upper_left = cords - (patch_creation.patch_size.value / 2)
        upper_right = copy(upper_left)
        upper_right[0] += patch_creation.patch_size.value
        lower_right = cords + (patch_creation.patch_size.value / 2)
        lower_left = copy(lower_right)
        lower_left[0] -= patch_creation.patch_size.value
        rectangle = np.array([upper_left, upper_right, lower_right, lower_left])

        rectangle = sanitize_rectangle(rectangle, patch_creation.img_layer.value.data.shape,
                                       patch_creation.patch_size.value-1)
        if len(patch_creation.img_layer.value.data.shape) == 3:
            current_slice = int(viewer.cursor.position[0])
            rectangle = np.insert(rectangle, 0, current_slice, axis=1)

        selection_layer = viewer.layers["2dselection"]
        selection_layer.add_rectangles(rectangle, edge_width=1, edge_color="cyan", face_color='transparent')


def draw_square(layer, event):
    viewer = napari.current_viewer()
    cords = np.round(layer.world_to_data(viewer.cursor.position)).astype(int)
    upper_left = cords - (patch_creation.patch_size.value / 2)
    lower_right = cords + (patch_creation.patch_size.value / 2)
    rectangle = np.array([upper_left, lower_right])
    layer.selected_data = set(range(layer.nshapes))
    layer.remove_selected()
    layer.add(
        sanitize_rectangle(rectangle, patch_creation.img_layer.value.data.shape, patch_creation.patch_size.value-1),
        shape_type='rectangle',
        edge_width=1,
        edge_color='coral',
        face_color='transparent'
    )


def sanitize_rectangle(rect: np.array, layer_shape: tuple, edge_length: int):
    shape_array = np.array(layer_shape)
    y_dim = len(shape_array)-1
    x_dim = len(shape_array)-2
    if len(rect) == 2:
        rect = sanitize_vertex(rect, 0, shape_array[x_dim]-edge_length, edge_length, shape_array[x_dim])
    if len(rect) == 4:
        rect[0] = sanitize_vertex(rect[0], 0, shape_array[x_dim]-edge_length, 0, shape_array[y_dim]-edge_length)
        rect[1] = sanitize_vertex(rect[1], edge_length, shape_array[y_dim], 0, shape_array[x_dim] - edge_length)
        rect[2] = sanitize_vertex(rect[2], edge_length, shape_array[x_dim], edge_length, shape_array[y_dim])
        rect[3] = sanitize_vertex(rect[3], 0, shape_array[y_dim] - edge_length, edge_length, shape_array[x_dim])
    return rect


def sanitize_vertex(vertex, low_a, high_a, low_b, high_b):
    vertex[0] = np.where(vertex[0] < low_a, low_a, vertex[0])
    vertex[0] = np.where(vertex[0] > high_a, high_a, vertex[0])
    vertex[1] = np.where(vertex[1] < low_b, low_b, vertex[1])
    vertex[1] = np.where(vertex[1] > high_b, high_b, vertex[1])
    return vertex


# Creates an img layer instead of an img
def create_img_layer(rectangle):
    viewer = napari.current_viewer()
    ixgrid = np.ix_(np.arange(rectangle[0][0], rectangle[1][0], dtype=int),
                    np.arange(rectangle[0][1], rectangle[1][1], dtype=int))
    current_slice = int(viewer.cursor.position[0])
    ixgrid = (current_slice,) + ixgrid
    img = patch_creation.img_layer.value.data[ixgrid]
    viewer.add_image(img, name="selection")
    viewer.layers.selection.select_only(viewer.layers["highlight"])


def slice_img_patch(rectangle, ndim: int):
    if ndim == 3:
        ixgrid = np.ix_(np.arange(rectangle[0][1], rectangle[1][1], dtype=int),
                        np.arange(rectangle[0][2], rectangle[2][2], dtype=int))
        ixgrid = (int(rectangle[0][0]),) + ixgrid
        img = patch_creation.img_layer.value.data[ixgrid]
        return img
    if ndim == 2:
        ixgrid = np.ix_(np.arange(rectangle[0][0], rectangle[1][0], dtype=int),
                        np.arange(rectangle[0][1], rectangle[2][1], dtype=int))
        img = patch_creation.img_layer.value.data[ixgrid]
        return img


if __name__ == "__main__":
    from napari_denoiseg._sample_data import denoiseg_data_n0

    data = denoiseg_data_n0()

    # create a Viewer
    viewer = napari.Viewer()

    # add our plugin
    viewer.window.add_dock_widget(patch_creation)

    # add images
    viewer.add_image(data[0][0][0:30], name=data[0][1]['name'])

    napari.run()

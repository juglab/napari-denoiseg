from copy import copy

import napari
from magicgui import magic_factory, magicgui
import numpy as np


@magicgui(call_button="Export",
    img_layer={'label': 'Layer for ROI picking'},
               patch_size={"widget_type": "Slider", "min": 16, "max": 512, "step": 8, 'value': 16, 'label': 'Patch size'},
               active={'widget_type': 'PushButton', 'label': 'Start patch selection', 'visible': True, 'value':True},
               store_path={'widget_type': 'FileEdit', 'mode':'d', 'label': 'Export path'},
               auto_call=False)
def patch_creation(img_layer: "napari.layers.Image", patch_size: int = 16, active: bool = True, store_path: str = None):
    viewer = napari.current_viewer()
    selection_layer = viewer.layers["2dselection"]
    shapes = selection_layer.data
    patches = [slice_img_patch(x) for x in shapes]
    print(shapes)
    print(patches)
    from PIL import Image
    for idx, patch in enumerate(patches):
        im = Image.fromarray(patch)
        im.save(str(store_path) + "/" + str(idx)+".tif", format="TIFF")


@patch_creation.active.clicked.connect
def start_stop_selection(state):
    viewer = napari.current_viewer()
    #patch_creation.active.value = not patch_creation.active.value
    if "highlight" not in viewer.layers:
        highlight_layer = viewer.add_shapes(name="highlight")
        viewer.add_shapes(name="2dselection", ndim=3)
        viewer.layers.selection.select_only(highlight_layer)
    highlight_layer = viewer.layers["highlight"]
    if not patch_creation.active.value:
        highlight_layer.mouse_move_callbacks.append(draw_square)
        highlight_layer.mouse_drag_callbacks.append(create_patch)
        patch_creation.active.label = "Stop patch selection"
    else:
        try:
            highlight_layer.mouse_move_callbacks.remove(draw_square)
            highlight_layer.mouse_drag_callbacks.remove(create_patch)
            patch_creation.active.label = "Start patch selection"
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

        rectangle = sanitize_rectangle(rectangle, patch_creation.img_layer.value.data.shape, patch_creation.patch_size.value)
        #create_img_layer(rectangle)
        current_slice = int(viewer.cursor.position[0])
        rectangle = np.insert(rectangle, 0, current_slice, axis=1)

        selection_layer = viewer.layers["2dselection"]
        selection_layer.add_rectangles(rectangle, edge_width=1, edge_color="cyan", face_color='transparent')


def draw_square(layer, event):
    viewer = napari.current_viewer()
    cords = np.round(layer.world_to_data(viewer.cursor.position)).astype(int)
    upper_left = cords-(patch_creation.patch_size.value/2)
    lower_right = cords+(patch_creation.patch_size.value/2)
    rectangle = np.array([upper_left, lower_right])
    layer.selected_data = set(range(layer.nshapes))
    layer.remove_selected()
    layer.add(
        sanitize_rectangle(rectangle, patch_creation.img_layer.value.data.shape, patch_creation.patch_size.value),
        shape_type='rectangle',
        edge_width=1,
        edge_color='coral',
        face_color='transparent'
    )

def sanitize_rectangle(rect:np.array, layer_shape:tuple, edge_length:int):
    shape_array = np.array(layer_shape)
    rect[0] = np.where(rect[0] < 0, 0, rect[0])
    rect[0] = np.where(rect[0] <= shape_array[1:]-edge_length, rect[0],  shape_array[1:]-edge_length)
    rect[1] = np.where(rect[1] > shape_array[1:], shape_array[1:], rect[1])
    rect[1] = np.where(rect[1] > edge_length, rect[1], edge_length)
    return rect

def create_img_layer(rectangle):
    viewer = napari.current_viewer()
    ixgrid = np.ix_(np.arange(rectangle[0][0], rectangle[1][0], dtype=int),
                    np.arange(rectangle[0][1], rectangle[1][1], dtype=int))
    current_slice = int(viewer.cursor.position[0])
    ixgrid = (current_slice,) + ixgrid
    img = patch_creation.img_layer.value.data[ixgrid]
    image = viewer.add_image(img, name="selection")
    viewer.layers.selection.select_only(viewer.layers["highlight"])

def slice_img_patch(rectangle):
    viewer = napari.current_viewer()
    ixgrid = np.ix_(np.arange(rectangle[0][1], rectangle[1][1], dtype=int),
                    np.arange(rectangle[0][2], rectangle[2][2], dtype=int))
    ixgrid = (int(rectangle[0][0]),) + ixgrid
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

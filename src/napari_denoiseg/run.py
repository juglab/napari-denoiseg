import napari

# create Qt GUI context
from src.napari_denoiseg import DenoiSegQWidget, example_magic_widget

with napari.gui_qt():
    # create a Viewer and add an image here
    viewer = napari.Viewer()

    # custom code to add data here
    viewer.window.add_dock_widget(example_magic_widget())

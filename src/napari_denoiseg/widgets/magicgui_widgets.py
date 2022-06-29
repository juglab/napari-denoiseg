from pathlib import Path

import napari
from magicgui import magic_factory
from magicgui.widgets import create_widget, Container, Widget


@magic_factory(auto_call=True,
               labels=False,
               slider={"widget_type": "Slider", "min": 0, "max": 100, "step": 5, 'value': 60})
def percentage_slider(slider: int):
    pass


@magic_factory(auto_call=True,
               Threshold={"widget_type": "FloatSpinBox", "min": 0, "max": 1., "step": 0.1, 'value': 0.6})
def threshold_spin(Threshold: int):
    pass


@magic_factory(auto_call=True, Model={'mode': 'r', 'filter': '*.h5 *.zip'})
def load_button(Model: Path):
    pass


@magic_factory(auto_call=True, use3d={'label': ' ', 'widget_type': 'Checkbox', 'visible': True})
def enable_3d(use3d: bool = False) -> bool:
    return use3d


def layer_choice(annotation, **kwargs) -> Widget:
    widget = create_widget(annotation=annotation, **kwargs)
    widget.reset_choices()
    viewer = napari.current_viewer()
    viewer.layers.events.inserted.connect(widget.reset_choices)
    viewer.layers.events.removed.connect(widget.reset_choices)
    viewer.layers.events.changed.connect(widget.reset_choices)
    return widget


def two_layers_choice():
    """
    Returns a container with two drop-down widgets to select images and masks.
    :return:
    """
    img = layer_choice(annotation=napari.layers.Image, name="Images")
    lbl = layer_choice(annotation=napari.layers.Labels, name="Labels")

    return Container(widgets=[img, lbl])

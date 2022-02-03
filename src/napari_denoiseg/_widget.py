"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton
from magicgui import magic_factory
import time


class DenoiSegQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton('Click me!')
        btn.clicked.connect(self._on_click)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print('napari has', len(self.viewer.layers), 'layers')


@magic_factory(perc_train_labels={"widget_type": "FloatSlider", "min": 0.1, "max": 1., "step": 0.05},
               n_epochs={"widget_type": "SpinBox", "step": 1},
               n_steps={"widget_type": "SpinBox", "step": 1},
               patch_shape={"widget_type": "Slider", "min": 16, "max": 512, "step": 16},
               batch_size={"widget_type": "Slider", "min": 16, "max": 512, "step": 16},
               neighborhood_radius={"widget_type": "Slider", "min": 1, "max": 16},
               progress={"widget_type": "ProgressBar", "min": 0, "max": 100})
def example_magic_widget(Data: 'napari.layers.Image',
                         Ground_truth: 'napari.layers.Labels',
                         perc_train_labels: float = 0.2,
                         n_epochs: int = 10,
                         n_steps: int = 200,
                         patch_shape: int = 64,
                         batch_size: int = 64,
                         neighborhood_radius: int = 5,
                         progress: int = 0):
    print('Will run lengthy calculation')
    worker = __process()
    worker.yielded.connect(__show_progress)

    worker.start()


def __show_progress(progress):
    pass


@thread_worker
def __process():
    n_pt = 10
    t_s = 2
    for i in range(n_pt):
        # update progress bar
        yield int(i * 100 / n_pt + 0.5)

        time.sleep(t_s)

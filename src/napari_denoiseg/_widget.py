"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton
from magicgui import magic_factory, widgets
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


@magic_factory(perc_train_labels={"widget_type": "FloatSlider", "min": 0.1, "max": 1., "step": 0.05, 'value': 0.6},
               n_epochs={"widget_type": "SpinBox", "step": 1, 'value': 10},
               n_steps={"widget_type": "SpinBox", "step": 1, 'value': 200},
               patch_shape={"widget_type": "Slider", "min": 16, "max": 512, "step": 16, 'value': 64},
               batch_size={"widget_type": "Slider", "min": 16, "max": 512, "step": 16, 'value': 64},
               neighborhood_radius={"widget_type": "Slider", "min": 1, "max": 16, 'value': 16},
               epoch_prog={'visible': True, 'min': 0, 'max': 100, 'step': 1, 'value': 0, 'label': 'epochs'},
               step_prog={'visible': True, 'min': 0, 'max': 100, 'step': 1, 'value': 0, 'label': 'steps'})
def example_magic_widget(Data: 'napari.layers.Image',
                         Ground_truth: 'napari.layers.Labels',
                         perc_train_labels: float,
                         n_epochs: int,
                         n_steps: int,
                         patch_shape: int,
                         batch_size: int,
                         neighborhood_radius: int,
                         epoch_prog: widgets.ProgressBar,
                         step_prog: widgets.ProgressBar):
    def update_progress(update):
        epoch_prog.native.setValue(update[0])
        step_prog.native.setValue(update[1])

    # ProgressBar.native.setValue avoids bug in magicgui ProgressBar.increment(val)
    # @thread_worker(connect={'yielded': epoch_prog.native.setValue})
    @thread_worker(connect={'yielded': update_progress})
    def process():
        n_pt = 5
        n_sp = 10
        t_s = 0.1
        for i in range(n_pt):
            p_epoch = int(i * 100 / (n_pt - 1) + 0.5)

            for j in range(n_sp):
                p_step = int(j * 100 / (n_sp - 1) + 0.5)

                # update progress bar
                yield p_epoch, p_step

                # sleep
                time.sleep(t_s)

    config = {'n_epochs': n_epochs,
              'n_steps': n_steps,
              'patch_shape': patch_shape,
              'batch_size': batch_size,
              'neighborhood_radius': neighborhood_radius}
    print(config)

    print('Will run lengthy calculation')
    process()


from qtpy.QtWidgets import (
    QDialog,
    QFormLayout
)
from napari_denoiseg.widgets import create_int_spinbox, create_double_spinbox


class TrainingSettingsWidget(QDialog):

    def __init__(self, parent):
        super().__init__(parent)

        # create widgets
        self.unet_depth = create_int_spinbox(value=3, min_value=3, max_value=5)
        self.unet_kernelsize = create_int_spinbox(value=3, min_value=3, max_value=7, step=2)
        self.unet_n_first = create_int_spinbox(value=32, min_value=8, step=8)
        self.train_learning_rate = create_double_spinbox(value=0.0004, step=0.0001)
        self.n2v_perc_pix = create_double_spinbox(value=1.5, step=0.1, max_value=100)
        self.n2v_neighborhood_radius = create_int_spinbox(value=5, min_value=3, max_value=16)
        self.denoiseg_alpha = create_double_spinbox(value=0.5, step=0.01)

        # arrange form layout
        form = QFormLayout()
        form.addRow('UNet depth', self.unet_depth)
        form.addRow('UNet kernel size', self.unet_kernelsize)
        form.addRow('UNet first filters', self.unet_n_first)
        form.addRow('Learning rate', self.train_learning_rate)
        form.addRow('N2V pixel %', self.n2v_perc_pix)
        form.addRow('N2V radius', self.n2v_neighborhood_radius)
        form.addRow('DenoiSeg \u03B1', self.denoiseg_alpha)

        self.setLayout(form)

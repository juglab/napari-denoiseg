
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QScrollArea, QWidget, QGridLayout, QLabel,QDialog

from napari_denoiseg.widgets.widget_provider import create_qspinbox, create_qdoublespinbox

class TrainingSettingsWidget(QDialog):

    def __init__(self, parent, viewer):
        super().__init__(parent)

        internal_widget = QWidget()
        layout = QGridLayout()
        layout.addWidget(QLabel("UNet Depth"), 0, 0)
        self.unet_depth = create_qspinbox(val=4)
        layout.addWidget(self.unet_depth, 0, 1)
        layout.addWidget(QLabel("UNet Kernel Size"), 0, 2)
        self.unet_kernelsize = create_qspinbox(val=3)
        layout.addWidget(self.unet_kernelsize, 0, 3)
        layout.addWidget(QLabel("unet_n_first"), 0, 4)
        self.unet_n_first = create_qspinbox(val=32)
        layout.addWidget(self.unet_n_first, 0, 5)
        layout.addWidget(QLabel("train_learning_rate"), 1, 0)
        self.train_learning_rate = create_qdoublespinbox(val=0.0004, max=1, step=0.0001)
        layout.addWidget(self.train_learning_rate, 1, 1)
        layout.addWidget(QLabel("n2v_perc_pix"), 1, 2)
        self.n2v_perc_pix = create_qdoublespinbox(val=1.5, step=0.1, max=100)
        layout.addWidget(self.n2v_perc_pix, 1, 3)
        layout.addWidget(QLabel("n2v_neighborhood_radius"), 1, 4)
        self.n2v_neighborhood_radius = create_qspinbox(val=5)
        layout.addWidget(self.n2v_neighborhood_radius, 1, 5)
        layout.addWidget(QLabel("denoiseg_alpha"), 2, 0)
        self.denoiseg_alpha = create_qdoublespinbox(max=1, val=0.5, step=0.01)
        layout.addWidget(self.denoiseg_alpha, 2, 1)

        internal_widget.setLayout(layout)
        self.setLayout(layout)

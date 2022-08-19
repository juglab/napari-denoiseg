
from qtpy.QtWidgets import (
    QDialog,
    QFormLayout,
    QLabel
)
from napari_denoiseg.widgets import create_int_spinbox, create_double_spinbox


class TrainingSettingsWidget(QDialog):

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('Expert settings')

        # create widgets
        label_unet_depth = QLabel('U-Net depth')
        desc_unet_depth = 'Number of resolution levels of the U-Net architecture'
        self.unet_depth = create_int_spinbox(value=3, min_value=3, max_value=5)
        label_unet_depth.setToolTip(desc_unet_depth)
        self.unet_depth.setToolTip(desc_unet_depth)

        label_unet_kernelsize = QLabel('U-Net kernel size')
        desc_unet_kernelsize = 'Size of the convolution filters in all image dimensions'
        self.unet_kernelsize = create_int_spinbox(value=3, min_value=3, max_value=7, step=2)
        label_unet_kernelsize.setToolTip(desc_unet_kernelsize)
        self.unet_kernelsize.setToolTip(desc_unet_kernelsize)

        label_unet_n_first = QLabel('U-Net n filters')
        desc_unet_n_first = 'Number of convolution filters for first U-Net resolution level (value is doubled after ' \
                            'each down-sampling operation) '
        self.unet_n_first = create_int_spinbox(value=32, min_value=8, step=8)
        label_unet_n_first.setToolTip(desc_unet_n_first)
        self.unet_n_first.setToolTip(desc_unet_n_first)

        label_train_learning_rate = QLabel('Learning rate')
        desc_train_learning_rate = 'Fixed learning rate'
        self.train_learning_rate = create_double_spinbox(step=0.0001)
        self.train_learning_rate.setDecimals(4)
        self.train_learning_rate.setValue(0.0004)  # TODO: bug? cannot be set in create_double_spinbox.
        label_train_learning_rate.setToolTip(desc_train_learning_rate)
        self.train_learning_rate.setToolTip(desc_train_learning_rate)

        label_n2v_perc_pix = QLabel('N2V pixel %')
        desc_n2v_perc_pix = 'Percentage of pixel to mask per patch'
        self.n2v_perc_pix = create_double_spinbox(value=1.5, step=0.1, max_value=100)
        self.n2v_perc_pix.setDecimals(1)
        self.n2v_perc_pix.setToolTip(desc_n2v_perc_pix)
        label_n2v_perc_pix.setToolTip(desc_n2v_perc_pix)

        label_n2v_neighborhood_radius = QLabel('N2V radius')
        desc_n2v_neighborhood_radius = 'Neighborhood radius for n2v manipulator'
        self.n2v_neighborhood_radius = create_int_spinbox(value=5, min_value=3, max_value=16)
        self.n2v_neighborhood_radius.setToolTip(desc_n2v_neighborhood_radius)
        label_n2v_neighborhood_radius.setToolTip(desc_n2v_neighborhood_radius)

        label_denoiseg_alpha = QLabel('DenoiSeg \u03B1')
        desc_denoiseg_alpha = 'Denoising relative weight in the total loss function'
        self.denoiseg_alpha = create_double_spinbox(value=0.5, step=0.01)
        self.denoiseg_alpha.setToolTip(desc_denoiseg_alpha)
        label_denoiseg_alpha.setToolTip(desc_denoiseg_alpha)

        # arrange form layout
        form = QFormLayout()
        form.addRow(label_unet_depth, self.unet_depth)
        form.addRow(label_unet_kernelsize, self.unet_kernelsize)
        form.addRow(label_unet_n_first, self.unet_n_first)
        form.addRow(label_train_learning_rate, self.train_learning_rate)
        form.addRow(label_n2v_perc_pix, self.n2v_perc_pix)
        form.addRow(label_n2v_neighborhood_radius, self.n2v_neighborhood_radius)
        form.addRow(label_denoiseg_alpha, self.denoiseg_alpha)

        self.setLayout(form)

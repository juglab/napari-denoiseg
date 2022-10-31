from qtpy.QtWidgets import (
    QDialog,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QCheckBox,
    QGroupBox,
    QLineEdit
)

from .axes_widget import LettersValidator
from .qt_widgets import create_int_spinbox, create_double_spinbox
from .magicgui_widgets import load_button


class TrainingSettingsWidget(QDialog):

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('Expert settings')
        self.setLayout(QVBoxLayout())

        # defaults values
        unet_n_depth = 4
        unet_kern_size = 3
        unet_n_first = 32
        train_learning_rate = 0.0004
        n2v_perc_pix = 1.5
        n2v_neighborhood_radius = 5
        denoiseg_alpha = 0.5
        unet_residuals = False
        relative_weights = '1.0, 1.0, 5.0'

        # groups
        self.retraining = QGroupBox()
        self.retraining.setTitle("Retrain model")

        self.expert_settings = QGroupBox()
        self.expert_settings.setTitle("Expert settings")

        ####################################################
        # create widgets for expert settings
        label_unet_depth = QLabel('U-Net depth')
        desc_unet_depth = 'Number of resolution levels of the U-Net architecture'
        self.unet_depth = create_int_spinbox(value=unet_n_depth, min_value=2, max_value=5)
        label_unet_depth.setToolTip(desc_unet_depth)
        self.unet_depth.setToolTip(desc_unet_depth)

        label_unet_kernelsize = QLabel('U-Net kernel size')
        desc_unet_kernelsize = 'Size of the convolution filters in all image dimensions'
        self.unet_kernelsize = create_int_spinbox(value=unet_kern_size, min_value=3, max_value=9, step=2)
        label_unet_kernelsize.setToolTip(desc_unet_kernelsize)
        self.unet_kernelsize.setToolTip(desc_unet_kernelsize)

        label_unet_n_first = QLabel('U-Net n filters')
        desc_unet_n_first = 'Number of convolution filters for first U-Net resolution level\n' \
                            '(value is doubled after each down-sampling operation)'
        self.unet_n_first = create_int_spinbox(value=unet_n_first, min_value=8, step=8)
        label_unet_n_first.setToolTip(desc_unet_n_first)
        self.unet_n_first.setToolTip(desc_unet_n_first)

        label_unet_residuals = QLabel('U-Net residuals')
        desc_unet_residuals = 'If checked, model will internally predict the residual w.r.t.\n' \
                              'the input (typically better), this requires the number of input\n' \
                              'and output image channels to be equal'
        self.unet_residuals = QCheckBox()
        self.unet_residuals.setChecked(unet_residuals)
        label_unet_residuals.setToolTip(desc_unet_residuals)
        self.unet_residuals.setToolTip(desc_unet_residuals)

        label_train_learning_rate = QLabel('Learning rate')
        desc_train_learning_rate = 'Starting learning rate'
        self.train_learning_rate = create_double_spinbox(step=0.0001, n_decimal=4)
        self.train_learning_rate.setValue(train_learning_rate)  # TODO: bug? cannot be set in create_double_spinbox.
        label_train_learning_rate.setToolTip(desc_train_learning_rate)
        self.train_learning_rate.setToolTip(desc_train_learning_rate)

        label_n2v_perc_pix = QLabel('N2V pixel %')
        desc_n2v_perc_pix = 'Percentage of pixel to mask per patch'
        self.n2v_perc_pix = create_double_spinbox(value=n2v_perc_pix, step=0.001, max_value=100, n_decimal=3)
        self.n2v_perc_pix.setToolTip(desc_n2v_perc_pix)
        label_n2v_perc_pix.setToolTip(desc_n2v_perc_pix)

        label_n2v_neighborhood_radius = QLabel('N2V radius')
        desc_n2v_neighborhood_radius = 'Neighborhood radius for n2v manipulator'
        self.n2v_neighborhood_radius = create_int_spinbox(value=n2v_neighborhood_radius,
                                                          min_value=2,
                                                          step=1,
                                                          max_value=17)
        self.n2v_neighborhood_radius.setToolTip(desc_n2v_neighborhood_radius)
        label_n2v_neighborhood_radius.setToolTip(desc_n2v_neighborhood_radius)

        label_denoiseg_alpha = QLabel('DenoiSeg \u03B1')
        desc_denoiseg_alpha = 'Denoising relative weight in the total loss function (as opposed to segmentation)'
        self.denoiseg_alpha = create_double_spinbox(value=denoiseg_alpha, step=0.01, n_decimal=2)
        self.denoiseg_alpha.setToolTip(desc_denoiseg_alpha)
        label_denoiseg_alpha.setToolTip(desc_denoiseg_alpha)

        label_relative_weights = QLabel('Relative weights')
        desc_relative_weights = 'Relative weights of the three segmentation classes (background, foreground, border).' \
                                '\ne.g. 1.0, 2.0, 5.0'
        self.relative_weights = QLineEdit(relative_weights)
        self.relative_weights.setValidator(LettersValidator('0123456789., '))
        label_relative_weights.setToolTip(desc_relative_weights)
        self.relative_weights.setToolTip(desc_relative_weights)

        # arrange form layout
        form = QFormLayout()
        form.addRow(label_unet_depth, self.unet_depth)
        form.addRow(label_unet_kernelsize, self.unet_kernelsize)
        form.addRow(label_unet_n_first, self.unet_n_first)
        form.addRow(label_unet_residuals, self.unet_residuals)
        form.addRow(label_train_learning_rate, self.train_learning_rate)
        form.addRow(label_n2v_perc_pix, self.n2v_perc_pix)
        form.addRow(label_n2v_neighborhood_radius, self.n2v_neighborhood_radius)
        form.addRow(label_denoiseg_alpha, self.denoiseg_alpha)
        form.addRow(label_relative_weights, self.relative_weights)

        self.expert_settings.setLayout(form)

        ####################################################
        # create widgets for load model
        self.load_model_button = load_button()
        self.load_model_button.native.setToolTip('Load a pre-trained model (weights and configuration)')

        self.retraining.setLayout(QVBoxLayout())
        self.retraining.layout().addWidget(self.load_model_button.native)

        ####################################################
        # assemble expert settings
        self.layout().addWidget(self.retraining)
        self.layout().addWidget(self.expert_settings)

    def get_model_path(self):
        return self.load_model_button.Model.value

    def has_model(self):
        return self.get_model_path().exists() and self.get_model_path().is_file()

    def _get_relative_weights(self):
        def is_float(element: str) -> bool:
            try:
                float(element)
                return True
            except ValueError:
                return False

        if self.relative_weights.text() != '':
            weights = self.relative_weights.text()

            # remove spaces
            weights = weights.replace(' ', '')

            # split by commas
            weights = weights.split(',')

            # create float array
            weights = [float(s) for s in weights if is_float(s)]

            # if there are only 3 entries, return
            if len(weights) == 3:
                return weights

        return [1.0, 1.0, 5.0]

    def get_settings(self):
        return {'unet_kern_size': self.unet_kernelsize.value(),
                'unet_n_first': self.unet_n_first.value(),
                'unet_n_depth': self.unet_depth.value(),
                'unet_residual': self.unet_residuals.isChecked(),
                'train_learning_rate': self.train_learning_rate.value(),
                'n2v_perc_pix': self.n2v_perc_pix.value(),
                'n2v_neighborhood_radius': self.n2v_neighborhood_radius.value(),
                'denoiseg_alpha': self.denoiseg_alpha.value(),
                'relative_weights': self._get_relative_weights()}

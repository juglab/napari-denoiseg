from qtpy.QtWidgets import QSpinBox, QProgressBar


def create_qspinbox(min: int = 1, max: int = 1000, val: int = 2, step: int = 1, visible: bool = True,
                    tooltip: str = None) -> QSpinBox:
    qspinbox = QSpinBox()
    qspinbox.setMinimum(min)
    qspinbox.setMaximum(max)
    qspinbox.setSingleStep(step)
    qspinbox.setValue(val)
    qspinbox.setVisible(visible)
    qspinbox.setToolTip(tooltip)
    return qspinbox


def create_progressbar(min: int = 0, max: int = 100, val: int = 0, text_visible: bool = True,
                       visible: bool = True, format:str = f'Epoch ?/{100}', tooltip: str = None) -> QProgressBar:
    qprogressbar = QProgressBar()
    qprogressbar.setMinimum(min)
    qprogressbar.setMaximum(max)
    qprogressbar.setValue(val)
    qprogressbar.setVisible(visible)
    qprogressbar.setTextVisible(text_visible)
    qprogressbar.setFormat(format)
    qprogressbar.setToolTip(tooltip)
    return qprogressbar

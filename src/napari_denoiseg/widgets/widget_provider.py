from qtpy.QtWidgets import QSpinBox, QProgressBar, QDoubleSpinBox


def create_qdoublespinbox(min: float = 0, max: float = 1, val: float = 0.5, step: float = 0.1, visible: bool = True,
                    tooltip: str = None) -> QSpinBox:
    qspinbox = QDoubleSpinBox()
    qspinbox.setMinimum(min)
    qspinbox.setMaximum(max)
    qspinbox.setSingleStep(step)
    qspinbox.setValue(val)
    qspinbox.setVisible(visible)
    qspinbox.setToolTip(tooltip)
    qspinbox.setMinimumHeight(50)
    qspinbox.setContentsMargins(0,3,0,3)
    return qspinbox


def create_qspinbox(min: int = 1, max: int = 1000, val: int = 2, step: int = 1, visible: bool = True,
                    tooltip: str = None) -> QSpinBox:
    qspinbox = QSpinBox()
    qspinbox.setMinimum(min)
    qspinbox.setMaximum(max)
    qspinbox.setSingleStep(step)
    qspinbox.setValue(val)
    qspinbox.setVisible(visible)
    qspinbox.setToolTip(tooltip)
    qspinbox.setMinimumHeight(50)
    qspinbox.setContentsMargins(0,3,0,3)
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
    qprogressbar.setMinimumHeight(30)
    return qprogressbar

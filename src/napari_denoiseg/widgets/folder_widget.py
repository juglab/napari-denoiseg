from PyQt5.QtWidgets import QLabel
from qtpy.QtWidgets import (
    QWidget,
    QPushButton,
    QLineEdit,
    QHBoxLayout,
    QFileDialog
)


class FileEditWidget(QWidget):
    """
    A widget used for selecting an existing folder.
    """

    def __init__(self, text: str, select_directory: bool = True, label: str = None):
        super().__init__()

        self.select_directory = select_directory
        self.setLayout(QHBoxLayout())
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0, 0, 0, 0)

        if label is not None:
            self.label = QLabel(label);
            self.layout().addWidget(self.label)

        # text field
        self.text_field = QLineEdit('')
        self.layout().addWidget(self.text_field)

        # folder selection button
        self.button = QPushButton(text)
        self.layout().addWidget(self.button)
        self.button.clicked.connect(self._open_dialog)

    def _open_dialog(self):
        path = None
        if self.select_directory:
            path = QFileDialog.getExistingDirectory(self, 'Select Folder')
            print(path)
        else:
            path = QFileDialog.getOpenFileName(self, 'Select File')[0]
        self.text_field.setText(path)

        # set text in the text field

    def get_path(self):
        return self.text_field.text()

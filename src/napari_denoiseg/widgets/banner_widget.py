from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from qtpy.QtGui import QIcon, QPixmap, QImage
from qtpy import QtCore

class QBannerWidget(QWidget):

    def __init__(self, img_path: str, short_desc: str,  wiki_link: str, github_link: str):
        super().__init__()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        link_layout = QVBoxLayout()
        link_layout.setContentsMargins(5, 0, 0, 0)
        link_widget = QWidget()
        link_widget.setLayout(link_layout)
        link_layout.addWidget(QLabel(short_desc))
        link_layout.addWidget(self._create_link(wiki_link, "Documentation"))
        link_layout.addWidget(self._create_link(github_link, "GitHub"))
        icon = QPixmap(img_path)
        img_widget = QLabel()
        img_widget.setPixmap(icon)
        #img_widget.setIconSize(QtCore.QSize(200, 200))
        img_widget.setFixedSize(200, 200)
        #img_widget.setIcon(icon)
        #img_widget.setEnabled(False)
        layout.addWidget(img_widget)
        layout.addWidget(link_widget)

    def _create_link(self, link: str, text: str) -> QLabel:
        """

        :param link: the string this label should link to
        :return: returns a QLabel object with a hyperlink
        :rtype: object
        """
        label = QLabel()
        label.setContentsMargins(0, 5, 0, 5)
        label.setText("<a href=\"{}\">{}</a>".format(link, text))
        label.setOpenExternalLinks(True)
        label.setStyleSheet("font-weight: bold; color: green; text-decoration: underline")
        return label

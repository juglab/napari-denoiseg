import webbrowser

from qtpy import QtCore
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit
)
from qtpy.QtGui import QPixmap, QCursor


def _create_link(link: str, text: str) -> QLabel:
    """

    :param link: the string this label should link to
    :return: returns a QLabel object with a hyperlink
    :rtype: object
    """
    label = QLabel()
    label.setContentsMargins(0, 5, 0, 5)
    # TODO: is there a non-dark mode in napari?
    label.setText("<a href=\'{}\' style=\'color:white\'>{}</a>".format(link, text))
    label.setOpenExternalLinks(True)
    # label.setStyleSheet("font-weight: bold; color: green; text-decoration: underline")
    return label

def _open_link(link: str):
    def link_opener(event):
        webbrowser.open(link)

    return link_opener

class QBannerWidget(QWidget):

    def __init__(self, img_path: str, short_desc: str, wiki_link: str, github_link: str):
        super().__init__()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # logo
        icon = QPixmap(img_path)
        img_widget = QLabel()
        img_widget.setPixmap(icon)
        img_widget.setFixedSize(128, 128)
        # img_widget.setIconSize(QtCore.QSize(200, 200))
        # img_widget.setIcon(icon)
        # img_widget.setEnabled(False)

        # right panel
        right_layout = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # title
        title = QLabel('DenoiSeg')
        title.setStyleSheet("font-weight: bold;")

        # description
        description_widget = QPlainTextEdit()
        description_widget.setPlainText(short_desc)
        description_widget.setFixedSize(256, 80)

        # bottom widget
        bottom_widget = QWidget()
        bottom_widget.setLayout(QHBoxLayout())

        # github logo
        gh_icon = QPixmap('../resources/icons/GitHub-Mark-Light-32px.png')
        gh_widget = QLabel()
        gh_widget.setPixmap(gh_icon)
        gh_widget.mousePressEvent = _open_link(github_link)
        gh_widget.setCursor(QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

        # add widgets
        bottom_widget.layout().addWidget(_create_link(wiki_link, "Documentation"))
        bottom_widget.layout().addWidget(gh_widget)

        right_widget.layout().addWidget(title)
        right_widget.layout().addWidget(description_widget)
        right_widget.layout().addWidget(bottom_widget)

        # right_widget.setLayout(right_layout)
        # right_layout.addWidget(QLabel(short_desc))
        # right_layout.addWidget(_create_link(wiki_link, "Documentation"))
        # right_layout.addWidget(_create_link(github_link, "GitHub"))

        # add widgets
        layout.addWidget(img_widget)
        layout.addWidget(right_widget)

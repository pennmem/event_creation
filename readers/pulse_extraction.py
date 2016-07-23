import sys
from PyQt4.QtGui import *
from PyQt4 import QtCore

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import RectangleSelector

from pylab import draw

import random

import numpy as np

import os

DATA_ROOT = '/Volumes/rhino_mount/data/eeg/'

class LabeledEditLayout(QHBoxLayout):

    def __init__(self, label):
        super(LabeledEditLayout, self).__init__()

        self.label = QLabel(label)
        self.label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.edit = QLineEdit()
        self.edit.setMaximumWidth(150)
        self.addWidget(self.label)
        self.addWidget(self.edit)

    @property
    def text(self):
        return str(self.edit.text())

    def set_text(self, text):
        self.edit.setText(text)

    def not_empty(self):
        return len(str(self.text).strip()) > 0

    @property
    def textChanged(self):
        return self.edit.textChanged

    @property
    def editingFinished(self):
        return self.edit.editingFinished

    @property
    def returnPressed(self):
        return self.edit.returnPressed

class SyncPulseExtractor(QWidget):

    def __init__(self, model=None, parent=None):

        super(SyncPulseExtractor, self).__init__(parent)

        self.model = model or SyncPulseExtractionModel()
        self.setMinimumWidth(2000)

        layout = QVBoxLayout(self)

        self.load_button = QPushButton("Load EEG")
        self.save_button = QPushButton("Save Pulses")

        self.load_button.setMaximumWidth(150)
        self.save_button.setMaximumWidth(150)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)

        self.subject_edit = LabeledEditLayout("Subject")
        self.task_edit = LabeledEditLayout("Task")
        self.session_edit = LabeledEditLayout("Session")

        info_layout = QVBoxLayout()
        info_layout.addLayout(self.subject_edit)
        info_layout.addLayout(self.task_edit)
        info_layout.addLayout(self.session_edit)

        top_layout = QHBoxLayout()
        top_layout.addLayout(button_layout)
        top_layout.addLayout(info_layout)

        layout.addLayout(top_layout)


        self.figure = plt.figure()

        self.canvas = FigureCanvas(self.figure)

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.clear_selection_button = QPushButton("Clear selection")
        self.clear_selection_button.clicked.connect(self.clear_peaks)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.clear_selection_button)

        self.gs = gridspec.GridSpec(2, 1, height_ratios=[1,5])

        self.full_ax = plt.subplot(self.gs[1])
        self.zoom_ax = plt.subplot(self.gs[0])

        self.rectangle_selector = RectangleSelector(self.full_ax, self.selection_callback,
                                                    drawtype='box', useblit=True,
                                                    button=[1,3],
                                                    minspanx=5, minspany=5,
                                                    spancoords='pixels',
                                                    interactive=False,
                                                    rectprops={'fill':False}
                                                    )

        self.model.load_eeg_file(os.path.join(DATA_ROOT, 'R1008J', 'eeg.noreref', 'R1008J_21Nov14_1037.121'))

        self.figure.set_tight_layout(True)
        self.plot()

    def selection_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        try:
            peaks_x, peaks_y = self.model.find_and_add_peaks(x1, y1, x2, y2)

            self.zoom_ax.plot(peaks_x, peaks_y, 'r*')
            self.full_ax.plot(peaks_x, peaks_y, 'r*')
            self.figure.canvas.draw()

        except UnFindablePeaksException:
            print 'BOO!'
            pass

    def clear_peaks(self):
        self.model.clear_peaks()
        self.plot()

    def plot(self):
        ''' plot data '''
        self.plot_full()
        self.plot_zoom()

    def plot_full(self):
        # discards the old graph
        self.full_ax.hold(False)

        # plot data
        self.full_ax.plot(self.model.data, '-')

        self.full_ax.hold(True)

    def plot_zoom(self):
        self.zoom_ax.hold(False)
        self.zoom_ax.plot(self.model.data, '-')
        self.zoom_ax.hold(True)
        mid = len(self.model.data)/2
        data_range = [mid-20000, mid+20000]
        mid_data = self.model.data[data_range[0]:data_range[1]]
        min_data = min(mid_data)
        max_data = max(mid_data)
        self.zoom_ax.set_xlim(*data_range)
        self.zoom_ax.set_ylim(min_data, max_data)
        self.zoom_ax.get_xaxis().set_ticks(data_range)
        self.zoom_ax.get_yaxis().set_ticks([])


class UnFindablePeaksException(Exception):
    pass

class SyncPulseExtractionModel(object):

    def __init__(self):
        self.eeg_files = None
        self.selected_x_peaks = None
        self._data = None


    def load_eeg_file(self, eeg_file_1, eeg_file_2=None):
        params_file = os.path.splitext(eeg_file_1)[0] + '.params.txt'
        params = dict([line.split() for line in open(params_file, 'r').readlines()])
        data_format = params['dataformat'].strip('\'')

        data = np.fromfile(open(eeg_file_1), data_format)

        if eeg_file_2:
            data = np.fromfile(open(eeg_file_2), data_format) - data
            self.eeg_files = (eeg_file_1, eeg_file_2)
        else:
            self.eeg_files = (eeg_file_1,)

        self._data = data * float(params['gain'])

    @property
    def data_loaded(self):
        return not self._data is None

    @property
    def data(self):
        return self._data

    def clear_peaks(self):
        self.selected_x_peaks = None

    def find_and_add_peaks(self, x1, y1, x2, y2):

        min_x = int(min(x1, x2))
        max_x = int(max(x1, x2))

        data_in_x_range = self._data[min_x:max_x]

        peaks_x, peaks_y = self.find_peaks_in_selection(data_in_x_range, y1, y2)
        self.selected_x_peaks = min_x + np.array(peaks_x)
        return self.selected_x_peaks, np.array(peaks_y)


    @staticmethod
    def find_peaks_in_selection(data, y1, y2):
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        if (data < min_y).any() and (data > max_y).any():
            raise UnFindablePeaksException()

        if (data < min_y).any():
            in_border = min_y
            sign = +1
        else:
            in_border = max_y
            sign = -1

        crossings = np.diff(((sign * data) > (sign * in_border)).astype(int))

        up_crossings = np.where(crossings == 1)[0]
        dn_crossings = np.where(crossings == -1)[0] + 1

        peaks = []

        for up in up_crossings:
            following_downs = dn_crossings[np.where(dn_crossings >= up)[0]]
            if len(following_downs) > 0:
                down = following_downs[0]
                if down == up:
                    peaks.append((up, data[up]))
                else:
                    max_ind = np.argmax(sign * (data[up:down])) + up
                    peaks.append((max_ind, data[max_ind]))

        if peaks:
            return zip(*peaks)
        else:
            return [], []






if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = SyncPulseExtractor()
    main.show()

    sys.exit(app.exec_())
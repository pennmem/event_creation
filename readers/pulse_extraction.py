import sys
from PyQt4.QtGui import *
from PyQt4 import QtCore
from threading import Timer

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
        self.setMinimumWidth(1000)

        layout = QVBoxLayout(self)

        self.load_button = QPushButton("Load EEG")
        self.save_button = QPushButton("Save Pulses")

        self.load_button.setMaximumWidth(150)
        self.save_button.setMaximumWidth(150)
        self.save_button.setEnabled(False)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)

        self.eeg_text = QLabel("No EEG Loaded")
        self.status_text = QLabel("")
        self.eeg_text.setAlignment(QtCore.Qt.AlignCenter)
        self.eeg_text.setFont(QFont('', 12))

        status_layout = QVBoxLayout()
        status_layout.addWidget(self.eeg_text, alignment=QtCore.Qt.AlignHCenter)
        status_layout.addWidget(self.status_text, alignment=QtCore.Qt.AlignHCenter)

        self.subject_edit = LabeledEditLayout("Subject")
        self.elec1_edit = LabeledEditLayout("Elec 1")
        self.elec2_edit = LabeledEditLayout("Elec 2")

        self.elec1_edit.label.setMaximumWidth(45)
        self.elec1_edit.edit.setMaximumWidth(45)
        self.elec2_edit.label.setMaximumWidth(45)
        self.elec2_edit.edit.setMaximumWidth(45)


        elec_layout = QHBoxLayout()
        elec_layout.setAlignment(QtCore.Qt.AlignRight)
        elec_layout.addLayout(self.elec1_edit)
        elec_layout.addLayout(self.elec2_edit)
        elec_layout.setSpacing(10)

        info_layout = QVBoxLayout()
        info_layout.addLayout(self.subject_edit)
        info_layout.addLayout(elec_layout)

        top_layout = QHBoxLayout()
        top_layout.addLayout(button_layout)
        top_layout.addStretch(1)
        top_layout.addLayout(status_layout)
        top_layout.addStretch(1)
        top_layout.addLayout(info_layout)

        layout.addLayout(top_layout)


        self.figure = plt.figure()

        self.canvas = FigureCanvas(self.figure)

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.clear_selection_button = QPushButton("Clear selection")

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
        self.figure.set_tight_layout(True)
        self.assign_callbacks()
        self.apply_edits()

    def assign_callbacks(self):
        self.clear_selection_button.clicked.connect(self.clear_peaks)
        self.subject_edit.editingFinished.connect(self.subject_edit_callback)
        self.elec1_edit.editingFinished.connect(self.elec1_edit_callback)
        self.elec2_edit.editingFinished.connect(self.elec2_edit_callback)

        self.load_button.clicked.connect(self.load_eeg_pressed)
        self.save_button.clicked.connect(self.save_pulses_pressed)

    def load_eeg_pressed(self):
        self.subject_edit_callback()
        self.elec1_edit_callback()
        self.elec2_edit_callback()

        eeg_directory = self.model.get_eeg_dir()

        filenames = QFileDialog.getOpenFileNames(self, "Select EEG Files", eeg_directory)

        if filenames:
            if len(filenames) > 2:
                msg = QMessageBox(self)
                msg.setText("Cannot select more than two files. Select again.")
                msg.exec_()
                return
            filenames = [str(filename) for filename in filenames]
            filenames.sort()
            self.model.load_eeg_file(*filenames)
            self.plot()
            self.figure.canvas.draw()
            shortened_filenames = \
                [fname if len(fname) < 50 else '...' + fname[-50:] for fname in filenames]
            self.eeg_text.setText('\n'.join(['EEG Loaded: %s' % filename for filename in shortened_filenames]))
            self.apply_edits()
            self.save_button.setEnabled(True)

    def save_pulses_pressed(self):
        save_name = self.model.get_save_name()

        filename = QFileDialog.getSaveFileName(self, 'Save pulses',
                                               save_name,
                                               '.txt')

        if filename:
            self.model.save_peaks(filename)
            self.status_text.setText("%s saved successfully!" % os.path.basename(str(filename)))
        else:
            self.status_text.setText("Save cancelled...")

    def subject_edit_callback(self):
        self.model.subject = self.subject_edit.text

    def task_edit_callback(self):
        self.model.task = self.task_edit.text

    def session_edit_callback(self):
        self.model.session = self.session_edit.text

    def elec1_edit_callback(self):
        self.model.elec1 = self.elec1_edit.text

    def elec2_edit_callback(self):
        self.model.elec2 = self.elec2_edit.text

    def apply_edits(self):
        self.subject_edit.set_text(self.model.subject)
        self.elec1_edit.set_text(self.model.elec1)
        self.elec2_edit.set_text(self.model.elec2)

    def enable_blit(self):
        self.rectangle_selector.useblit = True

    def selection_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        try:
            peaks_x, peaks_y = self.model.find_and_add_peaks(x1, y1, x2, y2)

            self.status_text.setText("%d pulses selected..." % len(peaks_x))

            # TODO: THIS IS TERRIBLE, but it's the only way I could figure out to update the Zoom plot
            self.rectangle_selector.useblit = False
            self.zoom_ax.plot(peaks_x, peaks_y, 'r*')
            self.full_ax.plot(peaks_x, peaks_y, 'r*')

            self.figure.canvas.draw()
            Timer(.1, self.enable_blit).start()

        except UnFindablePeaksException:
            self.status_text.setText("Cannot properly locate peaks! Select again!")

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
        min_data = min(mid_data) - 200
        max_data = max(mid_data) + 200
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
        self.subject = 'R1008J'
        self.elec1 = ""
        self.elec2 = ""

    def get_save_name(self):
        base_name, _ = os.path.splitext(self.eeg_files[0])
        return '%s.%s.%s.sync.txt' % (base_name, self.elec1, self.elec2)

    def save_peaks(self, filename):
        with open(filename, 'w') as f:
            f.write('\n'.join([str(x) for x in self.selected_x_peaks]))


    def get_eeg_dir(self):
        return os.path.join(DATA_ROOT, self.subject, 'eeg.noreref')

    def load_eeg_file(self, eeg_file_1, eeg_file_2=None):
        params_file = os.path.splitext(eeg_file_1)[0] + '.params.txt'
        params = dict([line.split() for line in open(params_file, 'r').readlines()])
        data_format = params['dataformat'].strip('\'')

        data = np.fromfile(open(eeg_file_1), data_format)
        self.elec1 = os.path.splitext(eeg_file_1)[-1][1:]
        basename = os.path.basename(eeg_file_1)
        self.subject = basename.split('_')[0]
        if eeg_file_2:
            data = np.fromfile(open(eeg_file_2), data_format) - data
            self.eeg_files = (eeg_file_1, eeg_file_2)
            self.elec2 = os.path.splitext(eeg_file_2)[-1][1:]
        else:
            self.eeg_files = (eeg_file_1,)
            self.elec2 = ''

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
    main.raise_()
    sys.exit(app.exec_())
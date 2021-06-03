import sys
from threading import Timer
import os
import glob
import numpy as np

from PyQt4.QtGui import *
from PyQt4 import QtCore

import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.widgets import RectangleSelector
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000

from .eeg_reader import NK_reader, EDF_reader
from ..exc import PeakFindingError

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

        subject_with_spacer = QHBoxLayout()

        self.subject_edit = LabeledEditLayout("Subject: ")
        self.subject_edit.edit.setMaximumWidth(100)
        self.subject_edit.addStretch(1)

        subject_with_spacer.addSpacing(8)
        subject_with_spacer.addLayout(self.subject_edit)

        self.data_root_button = QPushButton("Data Root")
        self.data_root_contents = QLabel("")

        self.data_root_contents.setMinimumWidth(200)
        self.data_root_contents.setFont(QFont('', 8))

        data_root_layout = QHBoxLayout()
        data_root_layout.addWidget(self.data_root_button)
        data_root_layout.addWidget(self.data_root_contents)
        data_root_layout.addStretch(1)

        subject_and_mount_layout = QVBoxLayout()
        subject_and_mount_layout.addLayout(subject_with_spacer)
        subject_and_mount_layout.addLayout(data_root_layout)
        subject_and_mount_layout.setSpacing(5)
        subject_and_mount_layout.setAlignment(QtCore.Qt.AlignLeft)

        info_layout = QHBoxLayout()
        info_layout.addLayout(subject_and_mount_layout)

        layout = QVBoxLayout(self)

        self.elec1_drop_down = QComboBox()
        self.elec2_drop_down = QComboBox()
        self.elec1_drop_down.setMaximumWidth(75)
        self.elec2_drop_down.setMaximumWidth(75)

        self.load_button = QPushButton("Load")
        self.load_button.setMaximumWidth(150)

        elec_layout = QHBoxLayout()
        elec_layout.setAlignment(QtCore.Qt.AlignRight)
        elec_layout.addWidget(self.elec1_drop_down)
        elec_layout.addWidget(self.elec2_drop_down)
        elec_layout.setSpacing(10)

        load_layout = QVBoxLayout()
        load_layout.addLayout(elec_layout)
        load_layout.addWidget(self.load_button)


        self.load_drop_down = QComboBox()
        self.load_drop_down.setMinimumWidth(200)

        self.save_button = QPushButton("Save Pulses")
        self.save_button.setMaximumWidth(400)
        self.save_button.setEnabled(False)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.load_drop_down)
        button_layout.addWidget(self.save_button)

        self.eeg_text = QLabel("No EEG Loaded")
        self.status_text = QLabel("")
        self.eeg_text.setAlignment(QtCore.Qt.AlignCenter)
        self.eeg_text.setMaximumWidth(400)
        self.eeg_text.setFont(QFont('', 8))

        status_layout = QVBoxLayout()
        status_layout.addWidget(self.eeg_text, alignment=QtCore.Qt.AlignHCenter)
        status_layout.addWidget(self.status_text, alignment=QtCore.Qt.AlignHCenter)

        top_layout = QHBoxLayout()
        top_layout.addLayout(info_layout)
        top_layout.addStretch(1)
        top_layout.addLayout(status_layout)
        top_layout.addStretch(1)
        top_layout.addLayout(load_layout)
        top_layout.addLayout(button_layout)

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
        self.zoom_ax.get_xaxis().set_ticks([])
        self.zoom_ax.get_yaxis().set_ticks([])

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
        self.data_root_button.clicked.connect(self.set_data_root_pressed)

        self.elec1_drop_down.activated.connect(self.elec1_changed)
        self.elec2_drop_down.activated.connect(self.elec2_changed)
        self.load_drop_down.activated.connect(self.eeg_file_changed)
        self.load_button.clicked.connect(self.load_eeg)
        self.save_button.clicked.connect(self.save_pulses_pressed)

    def eeg_file_changed(self):
        if self.load_drop_down.currentIndex() != 0:
            self.model.set_eeg_file(self.load_drop_down.currentText())
            self.update_possible_elecs()

    def unload_eeg(self):
        self.model.clear_data()
        self.eeg_text.setText("Nothing loaded...")
        self.save_button.setEnabled(False)
        self.plot()
        self.figure.canvas.draw()
        self.apply_edits()

    def load_eeg(self):
        self.show_message("Loading...")
        try:
            self.model.load_data()
        except:
            self.show_message("Error! Check console!")
            raise

        self.plot()
        self.figure.canvas.draw()
        self.apply_edits()

    @staticmethod
    def shorten_filenames(filename):
        return filename if len(filename) < 35 else '...' + filename[-35:]

    def set_data_root_pressed(self):
        filename = QFileDialog.getExistingDirectory(self, 'Set Data Root')

        if filename:
            self.model.data_root = str(filename)
            self.apply_edits()

    def save_pulses_pressed(self):
        save_name = self.model.get_save_name()

        filename = QFileDialog.getSaveFileName(self, 'Save pulses',
                                               save_name,
                                               '.txt')

        if filename:
            self.model.save_peaks(filename)
            self.status_text.setText("%s saved successfully!" % os.path.basename(str(filename)))
        else:
            self.status_text.setText("Save cancelled :( ..")

    def subject_edit_callback(self):
        self.model.set_subject(self.subject_edit.text)
        self.update_possible_eeg_files()
        self.eeg_file_changed()

    def update_possible_elecs(self):
        self.elec1_drop_down.clear()
        self.elec2_drop_down.clear()
        if self.model.possible_elecs:
            self.elec1_drop_down.addItems(['Select Elec 1'] + self.model.possible_elecs)
            self.elec2_drop_down.addItems(['Select Elec 2'] + self.model.possible_elecs)
        else:
            self.elec1_drop_down.addItem('--')
            self.elec2_drop_down.addItem('--')

    def elec1_changed(self):
        if self.elec1_drop_down.currentIndex() != 0:
            self.model.set_elec1(self.elec1_drop_down.currentText())
        else:
            self.model.set_elec1(None)

    def elec2_changed(self):
        if self.elec2_drop_down.currentIndex() != 0:
            self.model.set_elec2(self.elec2_drop_down.currentText())
        else:
            self.model.set_elec2(None)

    @staticmethod
    def change_drop_down_index(drop_down, text):
        items = [drop_down.itemText(i) for i in range(drop_down.count())]
        drop_down.setCurrentIndex(items.index(text))

    def apply_edits(self):
        self.subject_edit.set_text(self.model.subject)
        if self.model.elec1:
            self.change_drop_down_index(self.elec1_drop_down, self.model.elec1)
        if self.model.elec2:
            self.change_drop_down_index(self.elec2_drop_down, self.model.elec2)
        self.data_root_contents.setText(self.model.data_root)

    def enable_blit(self):
        self.rectangle_selector.useblit = True

    def update_possible_eeg_files(self):
        self.load_drop_down.clear()
        if self.model.possible_eeg_files:
            self.load_drop_down.addItems(['Select EEG'] + self.model.possible_eeg_names)
        else:
            self.load_drop_down.addItem("No eeg file found")

    def selection_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        try:
            peaks_x, peaks_y = self.model.find_and_add_peaks(x1, y1, x2, y2)

            self.status_text.setText("%d pulses selected..." % len(peaks_x))

            if len(peaks_x) > 0:
                self.save_button.setEnabled(True)

            self.zoom_ax.plot(peaks_x, peaks_y, 'r*')
            self.full_ax.plot(peaks_x, peaks_y, 'r*')

            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

        except PeakFindingError:
            self.status_text.setText("Cannot properly locate peaks! Select again!")

    def clear_peaks(self):
        self.model.clear_peaks()
        self.zoom_ax.clear()
        self.full_ax.clear()
        self.plot()
        self.save_button.setEnabled(False)

    def clear_axes(self):
        self.zoom_ax.clear()
        self.zoom_ax.get_xaxis().set_ticks([])
        self.zoom_ax.get_yaxis().set_ticks([])
        self.full_ax.clear()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def plot(self):
        ''' plot data '''
        if not self.model.data is None:
            self.plot_full()
            self.plot_zoom()
            self.figure.canvas.flush_events()
        else:
            self.clear_axes()

    def show_message(self, message):
        #self.full_ax.hold(False)
        self.full_ax.clear()
        self.full_ax.text(.5, .5, message, fontsize=30,
                          horizontalalignment='center',
                          verticalalignment='center')
        self.canvas.draw()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def plot_full(self):
        # discards the old graph
        #self.full_ax.hold(False)
        # plot data
        self.full_ax.plot(self.model.data, '-')

        #self.full_ax.hold(True)

    def plot_zoom(self):
        #self.zoom_ax.hold(False)
        self.zoom_ax.plot(self.model.data, '-')
        #self.zoom_ax.hold(True)
        mid = len(self.model.data)/2
        data_range = [mid-20000, mid+20000]
        mid_data = self.model.data[data_range[0]:data_range[1]]
        min_data = min(mid_data) - 200
        max_data = max(mid_data) + 200
        self.zoom_ax.set_xlim(*data_range)
        self.zoom_ax.set_ylim(min_data, max_data)
        self.zoom_ax.get_xaxis().set_ticks(data_range)
        self.zoom_ax.get_yaxis().set_ticks([])


class SyncPulseExtractionModel(object):

    DEFAULT_DATA_ROOT = '/data/eeg'

    def __init__(self):
        self.eeg_file = []
        self.possible_eeg_files = []
        self.possible_eeg_names = []
        self.selected_x_peaks = np.array([])
        self.basename = None
        self._data = None
        self._subject = ''
        self._elec1 = None
        self._elec2 = None
        self._elec1_num = None
        self._elec2_num = None
        self._reader = None
        self.data_root = self.DEFAULT_DATA_ROOT

    @property
    def subject(self):
        return self._subject

    @property
    def elec1(self):
        return self._elec1

    @property
    def elec2(self):
        return self._elec2

    def set_subject(self, subject):
        self._subject = subject
        self.find_eeg_files()
        self.eeg_file = None

    def set_elec1(self, elec1):
        self._elec1 = elec1
        if elec1:
            self._elec1_num = list(self._labels.keys())[list(self._labels.values()).index(self._elec1)]

    def set_elec2(self, elec2):
        self._elec2 = elec2
        if elec2:
            self._elec2_num = list(self._labels.keys())[list(self._labels.values()).index(self._elec2)]

    def set_eeg_file(self, eeg_file):
        self.eeg_file = os.path.join(self.raw_dir(), str(eeg_file))
        self.set_elec1(None)
        self.set_elec2(None)
        self.load_eeg_file()

    @property
    def is_nk(self):
        return os.path.splitext(self.eeg_file)[-1].lower() == '.eeg'

    @property
    def possible_elecs(self):
        return list(self._labels.values())

    def raw_dir(self):
        return os.path.join(self.data_root, self.subject, 'raw')

    def noreref_dir(self):
        return os.path.join(self.data_root, self.subject, 'eeg.noreref')

    def get_eeg_file(self):
        return self.eeg_file

    def find_eeg_files(self):
        if not self.subject:
            return

        edf_files = glob.glob(os.path.join(self.raw_dir(), '*', '*.edf')) + \
                    glob.glob(os.path.join(self.raw_dir(), '*', '*.EDF'))
        nk_files  = glob.glob(os.path.join(self.raw_dir(), '*', '*.eeg')) + \
                    glob.glob(os.path.join(self.raw_dir(), '*', '*.EEG'))

        self.possible_eeg_files = edf_files + nk_files
        self.possible_eeg_names = [os.path.join(os.path.basename(os.path.dirname(f)),os.path.basename(f))
                                   for f in self.possible_eeg_files]

    def get_save_name(self):
        base_name = os.path.basename(os.path.dirname(self.eeg_file))
        dir_name = os.path.dirname(self.eeg_file)
        # FIXME
        return os.path.join(dir_name, '%s.sync.txt' % (base_name))

    def save_peaks(self, filename):
        with open(filename, 'w') as f:
            f.write('\n'.join([str(x) for x in self.selected_x_peaks]))

    def get_eeg_dir(self):
        return os.path.join(self.data_root, self.subject, 'eeg.noreref')

    def clear_data(self):
        self.eeg_file = None
        self._data = None
        self.selected_x_peaks = None

    def load_eeg_file(self):
        if self.is_nk:
            self._reader = NK_reader(self.eeg_file)
        else:
            self._reader = EDF_reader(self.eeg_file)

        jacksheet_filename = os.path.join(self.data_root, self.subject, 'docs', 'jacksheet.txt')
        if os.path.exists(jacksheet_filename):
            self._reader.set_jacksheet(jacksheet_filename)
            self._labels = {v:k for k,v in list(self._reader.jacksheet.items())}
        else:
            self._labels = self._reader.labels

    def load_data(self):
        data_1 = 0
        data_2 = 0
        if self.elec1:
            data_1 = self._reader.channel_data(self._elec1_num)
        if self.elec2:
            data_2 = self._reader.channel_data(self._elec2_num)
        self._data = data_1 - data_2

    @property
    def data_loaded(self):
        return not self._data is None

    @property
    def data(self):
        return self._data

    def clear_peaks(self):
        self.selected_x_peaks = np.array([])

    def find_and_add_peaks(self, x1, y1, x2, y2):
        if self.data is None:
            return

        min_x = int(min(x1, x2))
        max_x = int(max(x1, x2))

        data_in_x_range = self._data[min_x:max_x]

        peaks_x, peaks_y = self.find_peaks_in_selection(data_in_x_range, y1, y2)
        new_peaks = min_x + np.array(peaks_x)
        self.selected_x_peaks = np.concatenate([self.selected_x_peaks, new_peaks])
        return new_peaks, np.array(peaks_y)

    @staticmethod
    def find_peaks_in_selection(data, y1, y2):
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        if (data < min_y).any() and (data > max_y).any():
            raise PeakFindingError()

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
            return list(zip(*peaks))
        else:
            return [], []


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = SyncPulseExtractor()
    main.show()
    main.raise_()
    sys.exit(app.exec_())

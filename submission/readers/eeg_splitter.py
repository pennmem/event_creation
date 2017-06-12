import sys
from traceback import print_exc
import re
import json
import os
import glob

from ..configuration import config, paths
if __name__ == '__main__':
    config.parse_args()
    import matplotlib
    if not config.show_plots:
        matplotlib.use('agg')
    else:
        matplotlib.use('Qt4Agg')

from PyQt4 import QtCore
from PyQt4.QtGui import *

from .eeg_reader import EDF_reader, NK_reader, NSx_reader


class DeleteableListWidget(QListWidget):
    delete_pressed = QtCore.pyqtSignal()

    def __init__(self):
        super(QListWidget, self).__init__()

    def keyPressEvent(self, QKeyEvent):
        super(DeleteableListWidget, self).keyPressEvent(QKeyEvent)
        if QKeyEvent.key() == QtCore.Qt.Key_Delete or QKeyEvent.key() == QtCore.Qt.Key_Backspace:
            self.delete_pressed.emit()

class LabeledListboxLayout(QVBoxLayout):

    def __init__(self, label):
        super(LabeledListboxLayout, self).__init__()
        self.setSpacing(5)
        self.label = QLabel(label)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.addWidget(self.label)
        self._items = []
        self.list_box = DeleteableListWidget()
        self.addWidget(self.list_box)

    @property
    def items(self):
        return self._items

    def set_strings(self, items):
        self.list_box.clear()
        self._items = items
        for item in items:
            q_item = QListWidgetItem(item)
            self.list_box.addItem(q_item)

    def add_string(self, item):
        self._items.append(item)
        self.list_box.addItem(QListWidgetItem(item))

    def remove_selected_item(self):
        self.remove_item_by_index(self.list_box.currentRow())

    def remove_item_by_index(self, index):
        if index < len(self._items) and index >= 0:
            self.list_box.takeItem(index)
            del self._items[index]

    def remove_item_by_name(self, item):
        index = self._items.index(item)
        self.remove_item_by_index(index)

    @property
    def deletePressed(self):
        return self.list_box.delete_pressed

    @property
    def currentRow(self):
        return self.list_box.currentRow()

class LabeledEditLayout(QHBoxLayout):

    def __init__(self, label):
        super(LabeledEditLayout, self).__init__()

        self.label = QLabel(label)
        self.label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.edit = QLineEdit()
        self.edit.setMaximumWidth(150)
        self.addWidget(self.label, alignment=QtCore.Qt.AlignCenter)
        self.addWidget(self.edit, alignment=QtCore.Qt.AlignCenter)

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

class EntryListboxLayout(LabeledListboxLayout):

    def __init__(self, label):
        super(LabeledListboxLayout, self).__init__()
        self.setSpacing(5)
        self.top_row = LabeledEditLayout(label)
        self.addLayout(self.top_row)
        self._items = []
        self.list_box = DeleteableListWidget()
        self.addWidget(self.list_box)

    @property
    def text(self):
        return self.top_row.text

    def add_current_string(self):
        self.add_string(self.text)
        self.top_row.edit.setText("")

    @property
    def textChanged(self):
        return self.top_row.textChanged

    @property
    def editingFinished(self):
        return self.top_row.editingFinished

    @property
    def returnPressed(self):
        return self.top_row.returnPressed

    def not_empty(self):
        return self.top_row.not_empty()


class EEG_splitter_gui(QWidget):

    # ************************** CALLBACKS **************************

    # ******* Small Edits *******

    def subject_edit_callback(self):
        self.model.subject = self.subject_layout.text
        self.enable_buttons()

    def task_edit_callback(self):
        self.model.task = self.task_layout.text
        self.enable_buttons()

    def session_edit_callback(self):
        self.model.session = self.session_layout.text
        self.enable_buttons()

    # ******* Regex Edits *******

    def non_split_edit_callback(self):
        if self.non_split_entry.not_empty():
            self.non_split_entry.add_current_string()
            self.model.set_non_split_regex(self.non_split_entry.items)
            self.update_channels()

    def non_neural_edit_callback(self):
        if self.non_neural_entry.not_empty():
            self.non_neural_entry.add_current_string()
            self.model.set_non_neural_regex(self.non_neural_entry.items)
            self.update_channels()

    def non_reref_edit_callback(self):
        if self.non_reref_entry.not_empty():
            self.non_reref_entry.add_current_string()
            self.model.set_non_reref_regex(self.non_reref_entry.items)
            self.update_channels()

    # ******* Deletions *******

    def non_split_delete_callback(self):
        self.non_split_entry.remove_selected_item()
        self.model.set_non_split_regex(self.non_split_entry.items)
        self.update_channels()

    def non_neural_delete_callback(self):
        self.non_neural_entry.remove_selected_item()
        self.model.set_non_neural_regex(self.non_neural_entry.items)
        self.update_channels()

    def non_reref_delete_callback(self):
        self.non_reref_entry.remove_selected_item()
        self.model.set_non_reref_regex(self.non_reref_entry.items)
        self.update_channels()

    # ******* Checkboxes ******

    def remove_duplicates_callback(self):
        self.model.set_remove_duplicates(self.remove_duplicates_checkbox.isChecked())
        self.update_channels()

    def enable_channel_0_callback(self):
        self.model.set_enable_channel_0(self.enable_zero_checkbox.isChecked())
        self.update_channels()

    def remove_ref_callback(self):
        self.model.set_remove_ref(self.remove_ref_checkbox.isChecked())
        self.update_channels()

    # ******* Buttons *******

    def select_eeg_file(self):
        # Force the subject, task, and session to update
        self.subject_edit_callback()
        self.task_edit_callback()
        self.session_edit_callback()

        default_dir = self.model.default_eeg_directory
        if not os.path.exists(default_dir):
            default_dir = self.model.root_directory


        filename = str(QFileDialog.getOpenFileName(self, 'Select EEG File',
                                                   default_dir,
                                                   self.model.EEG_FILE_EXTENSIONS
                                                   ))
        if filename:
            filename = self.model.load_eeg_file(filename)
            if not filename:
                msg = QMessageBox(self)
                msg.setText("Could not load "
                            ""
                            "file...")
                msg.exec_()
            if len(filename) > 35:
                disp_filename = '...%s' % filename[-35:]
            else:
                disp_filename = filename
            self.loaded_eeg_text.setText('Loaded EEG: %s' % disp_filename)
            self.update_channels()
            self.enable_buttons()
            self.enable_checkboxes()

        self.load_eeg_button.setDown(False)


    def select_root(self):
        dirname = QFileDialog.getExistingDirectory(self, "Select root directory",
                                                   '/', QFileDialog.ShowDirsOnly)
        if dirname:
            self.model.root_directory = str(dirname)
            self.update_root()
            self.root_dir_button.setDown(False)

    def save_config(self):
        if not os.path.exists(self.model.CONFIG_DIRECTORY):
            os.mkdir(self.model.CONFIG_DIRECTORY)
        filename = QFileDialog.getSaveFileName(self, 'Save file',
                                               self.model.CONFIG_DIRECTORY,
                                               self.model.CONFIG_FILE_EXTENSION)
        if filename:
            self.model.save_config(str(filename))
        self.save_config_button.setDown(False)

    def load_config(self):
        filename = QFileDialog.getOpenFileName(self, 'Select Config File',
                                               self.model.CONFIG_DIRECTORY,
                                               self.model.CONFIG_FILE_EXTENSION)
        if filename:
            try:
                self.model.load_config(filename)
            except (IOError, KeyError):
                msg = QMessageBox(self)
                msg.setText("Could not load configuration file...")
                msg.exec_()
                return
            self.update_regexes()
            self.update_checkboxes()
            self.update_channels()
            self.update_edits()
            self.update_root()
            self.enable_checkboxes()
        self.load_config_button.setDown(False)

    def make_jacksheet(self, is_json):
        if self.model.jacksheet_exists(is_json):
            comparison = self.model.compare_jacksheets(is_json)
            if comparison:
                message = '%s\n\n%s' % (comparison, 'Do you want to continue?')
                reply = QMessageBox.question(self, 'Continue?', message,
                                             QMessageBox.Yes, QMessageBox.No)
                if reply == QMessageBox.No:
                    return False
            else:
                message = 'Matching jacksheet already exists:\n%s' % self.model.jacksheet_filename(is_json)
                msg = QMessageBox(self)
                msg.setText(message)
                msg.exec_()
                return True
        elif not self.model.jacksheet_directory_exists():
            message = 'Containing directory for %s does not exist. Permission to create?' % \
                      self.model.jacksheet_dirname
            reply = QMessageBox.question(self, 'Continue?', message,
                                         QMessageBox.Yes, QMessageBox.No)
            if reply == QMessageBox.No:
                return False
            os.makedirs(self.model.jacksheet_dirname)
        try:
            open(self.model.jacksheet_filename(is_json), 'a')
        except:
            print_exc()
            message = 'Could not open file %s for writing. Check permissions...' % self.model.jacksheet_filename(is_json)
            msg = QMessageBox(self)
            msg.setText(message)
            msg.exec_()
            return False
        message = 'Jacksheet will be placed in the file: %s\nContinue?' % self.model.jacksheet_filename(is_json)
        reply = QMessageBox.question(self, 'Continue?', message,
                                     QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.model.create_jacksheet(is_json)
            msg = QMessageBox(self)
            msg.setText("Jacksheet created successfully!")
            msg.exec_()
            return True
        else:
            return False


    def make_jacksheets(self):
        success = True
        if self.model.is_jacksheet_made:
            msg = QMessageBox(self)
            msg.setText("Nothing to do!")
            msg.exec_()
            self.jacksheet_button.setDown(False)
            return
        if self.model.OUTPUT_JSON_JACKSHEET:
            success = success and self.make_jacksheet(True)
        if success and self.model.OUTPUT_TXT_JACKSHEET:
            success = success and self.make_jacksheet(False)
        self.jacksheet_button.setDown(False)
        return success

    def show_non_blocking_status(self, text):
        self.status_message = QMessageBox(QMessageBox.Information, "", text)
        self.status_message.setModal(False)
        self.status_message.setStandardButtons(QMessageBox.NoButton)
        self.status_message.show()
        self.app.processEvents()

    def close_status(self):
        if self.status_message:
            self.status_message.close()
            self.status_message = None

    def split_channels(self):
        if self.model.is_eeg_split:
            msg = QMessageBox(self)
            msg.setText("Nothing to do!")
            msg.exec_()
            self.split_button.setDown(False)
            return

        if not self.model.is_jacksheet_made:
            if not self.make_jacksheets():
                self.split_button.setDown(False)
                return

        if self.model.was_previously_split():
            reply = QMessageBox.question(self, "Continue?", "EEG was previously split under the same name." +
                                         "Would you like to delete all split files before continuing?",
                                         QMessageBox.Yes, QMessageBox.No, QMessageBox.Cancel)
            if reply == QMessageBox.Cancel:
                self.split_button.setDown(False)
                return
            elif reply == QMessageBox.Yes:
                self.model.remove_split_eeg()

        self.show_non_blocking_status("Splitting in progress. Please wait - Window will close automatically")
        try:
            self.model.split_channels(self.model.OUTPUT_JSON_JACKSHEET)
            self.close_status()
            msg2 = QMessageBox(self)
            msg2.setText("Success! Huzzah!")
            msg2.exec_()
        except Exception as e:
            self.close_status()
            msg2 = QMessageBox(self)
            print_exc()
            msg2.setText("Error occurred! Booooo.\n%s"%e)
            msg2.exec_()
        self.split_button.setDown(False)

    def add_callbacks(self):
        self.subject_layout.editingFinished.connect(self.subject_edit_callback)
        self.task_layout.editingFinished.connect(self.task_edit_callback)
        self.session_layout.editingFinished.connect(self.session_edit_callback)

        self.non_split_entry.returnPressed.connect(self.non_split_edit_callback)
        self.non_neural_entry.returnPressed.connect(self.non_neural_edit_callback)
        self.non_reref_entry.returnPressed.connect(self.non_reref_edit_callback)

        self.non_split_entry.deletePressed.connect(self.non_split_delete_callback)
        self.non_neural_entry.deletePressed.connect(self.non_neural_delete_callback)
        self.non_reref_entry.deletePressed.connect(self.non_reref_delete_callback)

        self.remove_duplicates_checkbox.stateChanged.connect(self.remove_duplicates_callback)
        self.enable_zero_checkbox.stateChanged.connect(self.enable_channel_0_callback)
        self.remove_ref_checkbox.stateChanged.connect(self.remove_ref_callback)

        self.load_eeg_button.clicked.connect(self.select_eeg_file)
        self.save_config_button.clicked.connect(self.save_config)
        self.load_config_button.clicked.connect(self.load_config)
        self.root_dir_button.clicked.connect(self.select_root)
        self.jacksheet_button.clicked.connect(self.make_jacksheets)
        self.split_button.clicked.connect(self.split_channels)

    @staticmethod
    def strings_from_dict(d, separator=': '):
        keys = d.keys()
        keys.sort()
        return ['%d%s%s' % (k, separator, d[k]) for k in keys]

    def update_channels(self):
        self.update_list(self.non_split_display, self.model.non_split_channels)
        self.update_list(self.non_reref_display, self.model.non_reref_channels)
        self.update_list(self.non_neural_display, self.model.non_neural_channels)
        self.update_list(self.neural_display, self.model.neural_channels)

    def update_regexes(self):
        self.non_split_entry.set_strings(self.model.non_split_regex)
        self.non_neural_entry.set_strings(self.model.non_neural_regex)
        self.non_reref_entry.set_strings(self.model.non_reref_regex)

    def update_checkboxes(self):
        self.remove_duplicates_checkbox.setChecked(self.model.duplicates_removed)
        self.enable_zero_checkbox.setChecked(self.model.channel_0_enabled)
        self.remove_ref_checkbox.setChecked(self.model.ref_removed)

    def update_edits(self):
        self.subject_layout.set_text(self.model.subject)
        self.session_layout.set_text(self.model.session)
        self.task_layout.set_text(self.model.task)

    def update_root(self):
        self.root_dir_text.setText('Root: %s' % self.model.root_directory)

    def enable_buttons(self):
        if self.model.subject and self.model.reader:
            self.jacksheet_button.setEnabled(True)
            if self.model.session and self.model.task and not self.model.file_ext=='.ns2':
                self.split_button.setEnabled(True)
            else:
                self.split_button.setEnabled(False)
        else:
            self.jacksheet_button.setEnabled(False)
            self.split_button.setEnabled(False)

    def enable_checkboxes(self):
        enabled = not (self.model.file_ext == '.ns2' or self.model.file_ext == '.eeg')
        self.remove_duplicates_checkbox.setEnabled(True)
        self.enable_zero_checkbox.setEnabled(enabled)
        self.remove_ref_checkbox.setEnabled(enabled)
        if not enabled:
            self.enable_zero_checkbox.setChecked(False)
            self.remove_ref_checkbox.setChecked(False)

    @classmethod
    def update_list(cls, display, channels):
        display.set_strings(cls.strings_from_dict(channels))


    def __init__(self, model=None, load_default=True, app=None):
        super(EEG_splitter_gui, self).__init__()
        self.app = app
        self.model = model or EEG_splitter_model()
        if load_default:
            try:
                self.model.load_config(os.path.join(self.model.CONFIG_DIRECTORY, self.model.DEFAULT_CONFIG))
            except (IOError, ValueError, KeyError):
                msg = QMessageBox(self)
                msg.setText("Could not load default configuration. Creating now...")
                msg.exec_()
                if not os.path.exists(self.model.CONFIG_DIRECTORY):
                    os.mkdir(self.model.CONFIG_DIRECTORY)
                self.model.save_config(os.path.join(self.model.CONFIG_DIRECTORY, self.model.DEFAULT_CONFIG))
        self.setWindowTitle('EEG Splitter')
        self.status_message = None
        self.grid_layout = QGridLayout(self)

        # ------- COLUMN 1 ---------

        notes_label = QLabel(
r'''
Notes:
    ^       Beginning of string
    $       End of string
    \s      Whitespace
    \d+     One or more numbers
    [A-D]   Capital letters A-D
'''
        )
        notes_label.setFont(QFont("courier", 10))

        self.non_split_entry = EntryListboxLayout("Don't Split:")
        self.non_neural_entry = EntryListboxLayout("Non-Neural:")
        self.non_reref_entry = EntryListboxLayout("Don't Reref:")

        checkbox_layout = QVBoxLayout()
        checkbox_layout.setSpacing(10)
        self.remove_duplicates_checkbox = QCheckBox("Remove Duplicates")
        self.enable_zero_checkbox = QCheckBox("Enable Channel \"0\"")
        self.remove_ref_checkbox = QCheckBox('Remove "-REF"')

        self.root_dir_button = QPushButton("Select Root Directory")
        self.root_dir_button.setMinimumHeight(25)
        self.root_dir_text = QLabel("Root directory: ---")
        self.root_dir_text.setFont(QFont("", 10))

        checkbox_layout.addWidget(self.remove_duplicates_checkbox)
        checkbox_layout.addWidget(self.enable_zero_checkbox)
        checkbox_layout.addWidget(self.remove_ref_checkbox)
        checkbox_layout.addWidget(self.root_dir_button)
        checkbox_layout.addWidget(self.root_dir_text)

        self.grid_layout.addWidget(notes_label, 0, 0)
        self.grid_layout.addLayout(self.non_split_entry, 1, 0)
        self.grid_layout.addLayout(self.non_neural_entry, 2, 0)
        self.grid_layout.addLayout(self.non_reref_entry, 3, 0)
        self.grid_layout.addLayout(checkbox_layout, 4, 0)

        # ------- COLUMN 2 ---------

        title_and_load_layout = QVBoxLayout()
        title_and_load_layout.setSpacing(5)
        title_label = QLabel("PySplit!")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setFont(QFont("", 24))
        self.load_eeg_button = QPushButton("Load EEG")
        self.load_eeg_button.setMinimumHeight(50)
        self.loaded_eeg_text = QLabel("Loaded EEG: NONE")
        self.loaded_eeg_text.setFont(QFont("", 10))
        self.loaded_eeg_text.setAlignment(QtCore.Qt.AlignCenter)
        title_and_load_layout.addWidget(title_label)
        title_and_load_layout.addWidget(self.load_eeg_button)
        title_and_load_layout.addWidget(self.loaded_eeg_text)
        self.non_split_display = LabeledListboxLayout("Channels not split")
        self.non_neural_display = LabeledListboxLayout("Non-Neural Channels")
        self.non_reref_display = LabeledListboxLayout("Neural Channels not in reref")

        config_layout = QVBoxLayout()
        config_layout.setSpacing(5)

        self.save_config_button = QPushButton("Save Configuration")
        self.save_config_button.setMinimumHeight(50)
        self.load_config_button = QPushButton("Load Configuration")
        self.load_config_button.setMinimumHeight(50)
        config_layout.addWidget(self.load_config_button, alignment=QtCore.Qt.AlignBottom)
        config_layout.addWidget(self.save_config_button, alignment=QtCore.Qt.AlignBottom)

        self.grid_layout.addLayout(title_and_load_layout, 0, 1)
        self.grid_layout.addLayout(self.non_split_display, 1, 1)
        self.grid_layout.addLayout(self.non_neural_display, 2, 1)
        self.grid_layout.addLayout(self.non_reref_display, 3, 1)
        self.grid_layout.addLayout(config_layout, 4, 1)
        self.grid_layout.setAlignment(QtCore.Qt.AlignBottom)

        # ------- COLUMN 3 ---------

        info_layout = QVBoxLayout()
        info_layout.setSpacing(5)
        self.subject_layout = LabeledEditLayout("Subject:")
        self.task_layout = LabeledEditLayout("Task:")
        self.session_layout = LabeledEditLayout("Session:")
        info_layout.addLayout(self.subject_layout)
        info_layout.addLayout(self.task_layout)
        info_layout.addLayout(self.session_layout)

        self.neural_display = LabeledListboxLayout("Neural Channels")

        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(10)
        self.split_button = QPushButton("Split EEG")
        self.split_button.setMinimumHeight(50)
        self.jacksheet_button = QPushButton("Create Jacksheet")
        self.jacksheet_button.setMinimumHeight(50)
        actions_layout.addWidget(self.split_button, alignment=QtCore.Qt.AlignBottom)
        actions_layout.addWidget(self.jacksheet_button, alignment=QtCore.Qt.AlignBottom)

        self.grid_layout.addLayout(info_layout, 0, 2)
        self.grid_layout.addLayout(self.neural_display, 1, 2, 3, 1)
        self.grid_layout.addLayout(actions_layout, 4, 2)

        self.grid_layout.setSpacing(30)

        self.add_callbacks()
        self.update_regexes()
        self.update_checkboxes()
        self.update_edits()
        self.update_root()
        self.enable_buttons()
        self.enable_checkboxes()

class EEG_splitter_model():

    SUBJECT_DIRECTORY = '/data/eeg'

    CONFIG_DIRECTORY = os.path.join(os.environ['HOME'], 'pysplit_config')
    DEFAULT_CONFIG = 'default_config.json'

    EEG_FILE_EXTENSIONS = 'EEG files (*.EEG *.EDF *.NS2)'
    CONFIG_FILE_EXTENSION = 'JSON files (*.json)'

    DEFAULT_NON_SPLIT = [
        r'Event',
        r'PUSH',
        r'^E$',
        r'\$',
        r'^C',
        r'^A',
        r'\s',
        r'^B'
    ]

    DEFAULT_NON_NEURAL = [
        'Stim-2',
        'Stim-2-Stim-1',
        'EKG',
        'ECG',
        'Stim-1',
    ]

    DEFAULT_NON_REREF = []

    DEFAULT_REMOVE_DUPLICATES = True
    DEFAULT_ENABLE_CHANNEL_0 = False
    DEFAULT_REMOVE_REF = True

    ATTRS_TO_SAVE = [
        '_non_split_regex_list',
        '_non_neural_regex_list',
        '_non_reref_regex_list',
        '_should_remove_duplicates',
        '_should_enable_channel_0',
        '_should_remove_ref',
        'root_directory',
        'subject',
        'task',
        'session'
    ]

    OUTPUT_JSON_JACKSHEET = False
    OUTPUT_TXT_JACKSHEET = True

    JACKSHEET_JSON = 'jacksheet.json'
    JACKSHEET_TXT = 'jacksheet.txt'

    DEBUG = False

    def __init__(self):
        self._non_split_regex_list = self.DEFAULT_NON_SPLIT[:]
        self._non_neural_regex_list = self.DEFAULT_NON_NEURAL[:]
        self._non_reref_regex_list = self.DEFAULT_NON_REREF[:]
        self._non_split_channel_list = {}
        self._non_neural_channel_list = {}
        self._non_reref_channel_list = {}
        self._neural_channel_list = {}
        self._raw_channel_list = {}
        self._complete_channel_list = {}
        self._should_remove_duplicates = self.DEFAULT_REMOVE_DUPLICATES
        self._should_enable_channel_0 = self.DEFAULT_ENABLE_CHANNEL_0
        self._should_remove_ref = self.DEFAULT_REMOVE_REF
        self.root_directory = self.SUBJECT_DIRECTORY
        self.subject = ''
        self.task = ''
        self.session = ''
        self.loaded_eeg_file = ''
        self.reader = None
        self.is_jacksheet_made = False
        self.is_eeg_split = False

    @property
    def file_ext(self):
        [_, ext] = os.path.splitext(self.loaded_eeg_file)
        return ext.lower()

    def jacksheet_filename(self, is_json):
        if is_json:
            return self.json_jacksheet_filename
        else:
            return self.txt_jacksheet_filename

    @property
    def eeg_basename(self):
        return '%s_%s_%s_%s' % (self.subject, self.task, self.session, self.reader.get_start_time_string())

    @property
    def noreref_dir(self):
        if not self.DEBUG:
            return os.path.join(self.root_directory, self.subject, 'eeg.noreref')
        else:
            return os.path.join('./tests', self.subject, 'eeg.noreref')

    @property
    def jacksheet_dirname(self):
        if not self.DEBUG:
            return os.path.join(self.root_directory, self.subject, 'docs')
        else:
            return os.path.join('./tests', self.subject, 'docs')

    @property
    def json_jacksheet_filename(self):
        return os.path.join(self.jacksheet_dirname, self.JACKSHEET_JSON)

    @property
    def txt_jacksheet_filename(self):
        return os.path.join(self.jacksheet_dirname, self.JACKSHEET_TXT)

    @property
    def leads_filename(self):
        if not self.DEBUG:
            return os.path.join(self.root_directory, self.subject, 'tal', 'leads.txt')
        else:
            return os.path.join('./tests', self.subject, 'tal', 'leads.txt')

    @property
    def good_leads_filename(self):
        if not self.DEBUG:
            return os.path.join(self.root_directory, self.subject, 'tal', 'good_leads.txt')
        else:
            return os.path.join('./tests', self.subject, 'tal', 'good_leads.txt')


    @property
    def default_eeg_directory(self):
        path = self.SUBJECT_DIRECTORY
        if self.subject:
            path = os.path.join(path, self.subject)
            if self.task and self.session:
                path = os.path.join(path, 'raw', '%s_%s' % (self.task, self.session))
        return path

    @property
    def non_split_regex(self):
        return self._non_split_regex_list

    @property
    def non_split_channels(self):
        return self._non_split_channel_list

    @property
    def non_neural_channels(self):
        return self._non_neural_channel_list

    @property
    def non_neural_regex(self):
        return self._non_neural_regex_list

    @property
    def non_reref_channels(self):
        return self._non_reref_channel_list

    @property
    def non_reref_regex(self):
        return self._non_reref_regex_list

    @property
    def neural_channels(self):
        return self._neural_channel_list

    @property
    def duplicates_removed(self):
        return self._should_remove_duplicates

    @property
    def channel_0_enabled(self):
        return self._should_enable_channel_0

    @property
    def ref_removed(self):
        return self._should_remove_ref

    def set_non_split_regex(self, non_split_regex_list):
        self._non_split_regex_list = non_split_regex_list
        self.recalculate_lists()


    def set_non_neural_regex(self, non_neural_regex_list):
        self._non_neural_regex_list = non_neural_regex_list
        self.recalculate_lists()

    def set_non_reref_regex(self, non_reref_regex_list):
        self._non_reref_regex_list = non_reref_regex_list
        self.recalculate_lists()

    def set_remove_duplicates(self, value):
        self._should_remove_duplicates = value
        self.recalculate_lists()

    def set_enable_channel_0(self, value):
        self._should_enable_channel_0 = value
        self.recalculate_lists()

    def set_remove_ref(self, value):
        self._should_remove_ref = value
        self.recalculate_lists()

    def recalculate_lists(self):
        self.apply_filters()
        self._neural_channel_list = self._complete_channel_list.copy()
        self.set_channel_list(self._non_split_channel_list, self._non_split_regex_list)
        self.set_channel_list(self._non_neural_channel_list, self._non_neural_regex_list)
        self.set_channel_list(self._non_reref_channel_list, self._non_reref_regex_list)
        self.is_jacksheet_made = False
        self.is_eeg_split = False

    def set_channel_list(self, list_to_set, regex_list):
        included_nums = self.get_matching_channels(self._neural_channel_list, regex_list)
        list_to_set.clear()
        for num in included_nums:
            list_to_set[num] = self._neural_channel_list[num]
            if num in self._neural_channel_list:
                del self._neural_channel_list[num]

    @classmethod
    def get_matching_channels(cls, channel_dict, include_regexes):
        included_numbers = set()
        for number, channel in channel_dict.items():
            for include_regex in include_regexes:
                if re.search(include_regex, channel):
                    included_numbers.add(number)
        return included_numbers

    def load_config(self, config_file):
        config = json.load(open(config_file, 'r'))
        for attr in self.ATTRS_TO_SAVE:
            setattr(self, attr, config[attr])
        self.recalculate_lists()

    def save_config(self, config_file):
        [_, ext] = os.path.splitext(config_file)
        if ext.lower() != '.json':
            config_file += '.json'
        config = {attr: getattr(self, attr) for attr in self.ATTRS_TO_SAVE}
        json.dump(config, open(config_file, 'w'), indent=4)

    READER_BY_EXT = {
        '.edf' : EDF_reader,
        '.eeg' : NK_reader,
        '.ns2' : NSx_reader
    }

    @staticmethod
    def load_edf_file(eeg_file):
        return EDF_reader(eeg_file)

    @staticmethod
    def is_edf(eeg_file):
        return os.path.splitext(eeg_file)[-1].lower() == '.edf'

    @staticmethod
    def is_nk(eeg_file):
        return os.path.splitext(eeg_file)[-1].lower() == '.eeg'

    def load_eeg_file(self, eeg_file):
        self.loaded_eeg_file = eeg_file
        try:
            self.reader = self.READER_BY_EXT[self.file_ext](eeg_file)
        except IOError:
            return None
        self._raw_channel_list = self.reader.labels
        self.recalculate_lists()
        return self.loaded_eeg_file

    def apply_filters(self):
        self._complete_channel_list = self._raw_channel_list.copy()
        if self._should_remove_ref:
            self.remove_ref()
        if self._should_remove_duplicates:
            self.remove_duplicates()
        if self._should_enable_channel_0:
            self.enable_channel_0()

    def remove_ref(self):
        self._complete_channel_list = {k: re.sub('-REF$', '', v) for k, v in self._complete_channel_list.items()}

    def enable_channel_0(self):
        self._complete_channel_list = {k-1: v for k, v in self._complete_channel_list.items()}

    def remove_duplicates(self):
        keys = self._complete_channel_list.keys()
        keys.sort()
        unique_keys = []
        unique_values = []
        skipped = 0
        for key in keys:
            if self._complete_channel_list[key] not in unique_values:
                unique_keys.append(key - skipped)
                unique_values.append(self._complete_channel_list[key])
            else:
                skipped += 1
        self._complete_channel_list = dict(zip(unique_keys, unique_values))

    def jacksheet_exists(self, is_json):
        if is_json:
            return os.path.exists(self.json_jacksheet_filename)
        else:
            return os.path.exists(self.txt_jacksheet_filename)

    def jacksheet_directory_exists(self):
        return os.path.exists(self.jacksheet_dirname)

    @staticmethod
    def read_json_jacksheet(filename):
        jacksheet_str = json.load(open(filename))
        return {int(k):v for k,v in jacksheet_str.items()}
    
    @staticmethod
    def read_txt_jacksheet(filename):
        jacksheet_str = [line.strip().split() for line in open(filename, 'r')]
        return {int(line[0]):{'label': line[1]} for line in jacksheet_str}


    def compare_jacksheets(self, is_json):
        if is_json:
            try:
                jacksheet = self.read_json_jacksheet(self.json_jacksheet_filename)
            except:
                print_exc()
                return 'Jacksheet could not be read'
        else:
            try:
                jacksheet = self.construct_jacksheet_dict_from_txt()
            except:
                print_exc()
                return 'Jacksheet, leads, or good leads could not be read'

        only_old = {}
        for number, lead in jacksheet.items():
            number = int(number)
            label = lead['label']
            if lead['reref']:
                if number not in self.neural_channels or label != self.neural_channels[number]:
                    only_old[int(number)] = '%s (Neural, reref)' % label
            elif lead['neural']:
                if number not in self.non_reref_channels or label!=self.non_reref_channels[number]:
                    only_old[int(number)] = '%s (Neural, non-reref)' % label
            elif number not in self.non_neural_channels or label != self.non_neural_channels[number]:
                only_old[int(number)] = '%s (Non-neural)' % label

        only_new_neural = {}
        for number, lead in self.neural_channels.items():
            if number not in jacksheet or lead != jacksheet[number]['label'] or not jacksheet[number]['neural']\
                    or not jacksheet[number]['reref']:
                only_new_neural[number] = '%s (Neural, reref)' % lead

        only_new_non_neural = {}
        for number, lead in self.non_neural_channels.items():
            if number not in jacksheet or lead != jacksheet[number]['label'] or jacksheet[number]['neural']\
                    or jacksheet[number]['reref']:
                only_new_non_neural[number] = '%s (Non-Neural)' % lead

        only_new_non_reref = {}
        for number, lead in self.non_reref_channels.items():
            if number not in jacksheet or lead != jacksheet[number]['label'] or not jacksheet[number]['neural']\
                    or jacksheet[number]['reref']:
                only_new_non_reref[number] = '%s (Neural, non-reref)' % lead

        error = ''
        if only_old:
            print 'Only old: \n\t' + '\n\t'.join(EEG_splitter_gui.strings_from_dict(only_old))
            error += self.format_leads_for_message(only_old, 'old jacksheet')
        if only_new_neural:
            print 'Only new neural: \n\t' + '\n\t'.join(EEG_splitter_gui.strings_from_dict(only_new_neural))
            error += self.format_leads_for_message(only_new_neural, 'new neural channels')
        if only_new_non_reref:
            print 'Only new non-reref: \n\t' + '\n\t'.join(EEG_splitter_gui.strings_from_dict(only_new_non_reref))
            error += self.format_leads_for_message(only_new_non_reref, 'new non-reref channels')
        if only_new_non_neural:
            print 'Only new non-neural: \n\t' + '\n\t'.join(EEG_splitter_gui.strings_from_dict(only_new_non_neural))
            error += self.format_leads_for_message(only_new_non_neural, 'new non-neural channels')
        return error

    def format_leads_for_message(self, lead_dict, label):
        error = 'The following channels appear only in %s:\n' % label
        error_list = EEG_splitter_gui.strings_from_dict(lead_dict)
        if len(error_list) > 10:
            error_list = error_list[:10] + ['[clipped]']
        error += '\n'.join(error_list) + '\n\n'
        return error

    def create_leads(self):
        if not os.path.exists(os.path.dirname(self.leads_filename)):
            os.makedirs(os.path.dirname(self.leads_filename))
        current_channels = self.neural_channels.copy()
        current_channels.update(self.non_reref_channels)
        numbers = current_channels.keys()
        numbers.sort()
        with open(self.leads_filename, 'w') as f:
            f.write('\n'.join([str(n) for n in numbers]))

    def create_good_leads(self):
        if not os.path.exists(os.path.dirname(self.good_leads_filename)):
            os.makedirs(os.path.dirname(self.good_leads_filename))
        numbers = self.neural_channels.keys()
        numbers.sort()
        with open(self.good_leads_filename, 'w') as f:
            f.write('\n'.join([str(n) for n in numbers]))

    def build_jacksheet_dict(self):
        output = {}
        for k, v in self.neural_channels.items():
            output[k] = {'label': v, 'neural': True, 'reref': True}
        for k, v in self.non_neural_channels.items():
            output[k] = {'label': v, 'neural': False, 'reref': False}
        for k, v in self.non_reref_channels.items():
            output[k] = {'label': v, 'neural': True, 'reref': False}
        return output

    def construct_jacksheet_dict_from_txt(self):
        jacksheet = self.read_txt_jacksheet(self.txt_jacksheet_filename)
        for key in jacksheet:
            jacksheet[key]['neural'] = False
            jacksheet[key]['reref'] = False
        with open(self.leads_filename) as f:
            leads = [int(line.strip()) for line in f.readlines()]
            for lead in leads:
                jacksheet[lead]['neural'] = True
        with open(self.good_leads_filename) as f:
            leads = [int(line.strip()) for line in f.readlines()]
            for lead in leads:
                jacksheet[lead]['reref'] = True
        return jacksheet


    def create_jacksheet(self, is_json):
        with open(self.jacksheet_filename(is_json), 'w') as f:
            if is_json:
                json.dump(self.build_jacksheet_dict(), f, sort_keys=True, indent=4)
            else:
                current_channels = self.neural_channels.copy()
                current_channels.update(self.non_neural_channels)
                current_channels.update(self.non_reref_channels)
                f.write('\n'.join(EEG_splitter_gui.strings_from_dict(current_channels, ' ')))
                self.create_leads()
                self.create_good_leads()
            self.is_jacksheet_made = True

    def was_previously_split(self):
        if not os.path.exists(self.noreref_dir):
            return False
        files = glob.glob(os.path.join(self.noreref_dir, '%s*' % self.eeg_basename))
        return len(files) > 0

    def remove_split_eeg(self):
        files = glob.glob(os.path.join(self.noreref_dir, '%s*' % self.eeg_basename))
        for file in files:
            os.remove(file)

    def split_channels(self, is_json):
        if not os.path.exists(self.jacksheet_filename(is_json)):
            raise Exception("Cannot split EEG without jacksheet")
        self.reader.set_jacksheet(self.jacksheet_filename(is_json))
        if not os.path.exists(self.noreref_dir):
            os.mkdir(self.noreref_dir)
        self.reader.split_data(self.noreref_dir,
                               '%s_%s_%s_%s' % (self.subject, self.task, self.session, self.reader.get_start_time_string()))
        self.is_eeg_split = True

# create our window
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = EEG_splitter_gui(app=app)
    w.show()
    w.raise_()
    app.exec_()

def test_split_eeg():
    app = QApplication(sys.argv)
    model = EEG_splitter_model()
    model.subject = 'R1008J'
    model.task = 'YC1'
    model.session = '0'
    #model.load_eeg_file('/Volumes/rhino_mount/data/eeg/R1001P/raw/FR1_0/R1001P_2014-10-12A.edf')
    w = EEG_splitter_gui(model, load_default=False, app=app)
    w.show()
    w.raise_()
    app.exec_()

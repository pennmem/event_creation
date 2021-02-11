from .base_log_parser import BaseSessionLogParser, UnknownExperimentError
from .system2_log_parser import System2LogParser
from ..viewers.recarray import strip_accents
from copy import deepcopy
import numpy as np
import os
from .. import fileutil


class THSessionLogParser(BaseSessionLogParser):

    @staticmethod
    def _th_fields():
        """
        Returns the template for a new th field
        :return:
        """
        return (
            ('trial', -999, 'int16'),
            ('item_name', '', 'S64'),
            ('resp_word', '', 'S64'),
            ('serialpos', -999, 'int16'),
            ('probepos', -999, 'int16'),
            ('block', -999, 'int16'),
            ('locationX', -999, 'float64'),
            ('locationY', -999, 'float64'),
            ('navStartLocationX', -999, 'float64'),
            ('navStartLocationY', -999, 'float64'),
            ('recStartLocationX', -999, 'float64'),
            ('recStartLocationY', -999, 'float64'),
            ('reactionTime', -999, 'float64'),
            ('list_length', -999, 'int16'),
            ('recalled', False, 'b1'),
            ('exp_version', '', 'S64'),
            ('stim_list', False, 'b1'),
            ('is_stim', False, 'b1'),
        )


    TH_RADIUS_SIZE = 13.0
    _MSTIME_INDEX = 0
    _TYPE_INDEX = 1

    def __init__(self, protocol, subject, montage, experiment, session, files):

        # create parsed log file treasure.par from original log file before BaseSessionLogParser
        # is initialized
        files['treasure_par'] = os.path.join(os.path.dirname(files['session_log']), 'treasure.par')
        self.parse_raw_log(files['session_log'], files['treasure_par'])
        super(THSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files,
                                                 primary_log='treasure_par',
                                                 include_stim_params=True, allow_unparsed_events=True)

        self._log_header = ''
        self._trial = -999
        self._serialpos = -999
        self._probepos = -999
        self._stim_list = False
        self._item_name = ''
        self._block = -999
        self._locationX = -999
        self._locationY = -999
        self._navStartLocationX = -999
        self._navStartLocationY = -999
        self._recStartLocationX = -999
        self._recStartLocationY = -999
        self._reactionTime = -999
        self._list_length = -999
        self._resp_word = ''
        self._recalled = False

        # self._version = ''
        # kind of hacky, 'type' is the second entry in the header
        self._add_fields(*self._th_fields())
        self._add_type_to_new_event(
            type=self.event_header,
            CHEST=self.event_line,
            PROBE=self.event_line
        )
        self._add_type_to_modify_events(
            type=self.modify_header,
            PROBE=self.modify_rec
        )

    # Any reason to not have all the fields persist in TH?
    @staticmethod
    def persist_fields_during_stim(event):
        return [field[0] for field in THSessionLogParser._th_fields()]


    @property
    def _empty_event(self):
        """
        Overiding BaseSessionLogParser._empty_event because we don't have msoffset field
        """
        event = self.event_from_template(self._fields)
        event.protocol = self._protocol
        event.subject = self._subject
        event.montage = self._montage
        event.experiment = self._experiment
        event.session = self._session
        return event

    def event_default(self, split_line):
        """
        Override base class's default event to TH specific events.
        """
        event = self._empty_event
        event.mstime = int(split_line[self._MSTIME_INDEX])
        event.type = split_line[self._TYPE_INDEX]
        event.session = self._session
        event.trial = self._trial
        event.stim_list = self._stim_list
        event.item_name = self._item_name
        event.block = self._block
        event.serialpos = self._serialpos
        event.probepos = self._probepos
        event.locationX = self._locationX
        event.locationY = self._locationY
        event.navStartLocationX = self._navStartLocationX
        event.navStartLocationY = self._navStartLocationY
        event.recStartLocationX = self._recStartLocationX
        event.recStartLocationY = self._recStartLocationY
        event.reactionTime = self._reactionTime
        event.list_length = self._list_length
        event.resp_word = self._resp_word
        event.recalled = self._recalled
        return event

    def event_header(self, split_line):
        """I don't really want an event for this line, I'm just doing it to get the header. Getting the header because
        some old log files don't have the stimList column, and we need to know if that exists. Could do it based on the
        column number, but I feel like this is safer in case the log file changes."""
        self._log_header = split_line
        split_line[0] = -999
        split_line[1] = 'dummy'
        return self.event_default(split_line)

    def modify_header(self, events):
        """Remove dummy event"""
        events = events[events['type'] != 'dummy']
        return events

    def set_event_properties(self, split_line):

        ind = self._log_header.index('item')
        self._item_name = split_line[ind].upper() if split_line[ind] != 'None' else ''

        ind = self._log_header.index('trial')
        self._trial = int(split_line[ind])

        ind = self._log_header.index('block')
        self._block = int(split_line[ind])

        ind = self._log_header.index('list_length')
        self._list_length = int(split_line[ind])

        ind = self._log_header.index('serialpos')
        self._serialpos = int(split_line[ind])

        ind = self._log_header.index('probepos')
        self._probepos = int(split_line[ind])

        ind = self._log_header.index('locationX')
        self._locationX = float(split_line[ind])

        ind = self._log_header.index('locationY')
        self._locationY = float(split_line[ind])

        ind = self._log_header.index('navStartLocationX')
        self._navStartLocationX = float(split_line[ind])

        ind = self._log_header.index('navStartLocationY')
        self._navStartLocationY = float(split_line[ind])

        ind = self._log_header.index('recStartLocationX')
        self._recStartLocationX = float(split_line[ind]) if split_line[ind] != 'None' else -999

        ind = self._log_header.index('recStartLocationY')
        self._recStartLocationY = float(split_line[ind]) if split_line[ind] != 'None' else -999

        if 'stimList' in self._log_header:
            ind = self._log_header.index('stimList')
            self._stim_list = bool(int(split_line[ind]))

    def event_line(self, split_line):

        # set all the values in the line
        self.set_event_properties(split_line)
        # self._resp_word = ''
        # self._recalled = False
        event = self.event_default(split_line)
        return event

    def modify_rec(self, events):
        """"""

        probe_event_ind = len(events)-1
        ann_file = str(self._trial) + '_' + str(self._probepos-1)
        ann_outputs = self._parse_ann_file(ann_file)
        for i, recall in enumerate(ann_outputs):

            word = recall[-1]
            new_event = deepcopy(events[probe_event_ind])
            is_correct = new_event.item_name.lower() == word.lower()
            new_event.type = 'REC_EVENT'
            new_event.recalled = is_correct
            new_event.resp_word = word
            new_event.mstime += recall[0]

            modify_events_mask = np.logical_and.reduce((events.serialpos == self._serialpos,
                                            events.trial == self._trial,
                                            events.type != 'REC_EVENT'))

            # if not a vocalization, add the response word to the corresponding
            # presenation and probe events. If they say multiple words, this
            # will keep overwriting until the final word
            if word != '<>' and word != 'v' and word != '!':
                events.resp_word[modify_events_mask] = word
                events.recalled[modify_events_mask] = is_correct

        # don't forget about pass
        # don't forget about list_length

        # pres_events = (events['trial'] == self._trial) & (events['type'] == 'CHEST') & (events['item_name'] != '')
        # pres_events_inc_empty = (events['trial'] == self._trial) & (events['type'] == 'CHEST')
        # list_length = np.sum(pres_events)
        # for ind in np.where(pres_events_inc_empty)[0]:
        #     events[ind].list_length = list_length
        # events[-1].list_length = list_length
            events = np.append(events, new_event).view(np.recarray)
        return events

    @staticmethod
    def parse_raw_log(raw_log_file, out_file_path):
        def writeToFile(f,data,subject):
            columnOrder = ['mstime','type','item','trial','block','list_length','serialpos','probepos','locationX','locationY','navStartLocationX','navStartLocationY','recStartLocationX','recStartLocationY','stimList'];
            strToWrite = ''
            for col in columnOrder:
                line = data[col]
                if col != columnOrder[-1]:
                    strToWrite += '%s\t'%(line)
                else:
                    strToWrite += '%s\t%s\n'%(line,subject)
            f.write(strToWrite)

        def makeEmptyDict(mstime=None,eventType=None,item=None,trial=None,block=None,list_length=None,serialpos=None,probepos=None,locationX=None,locationY=None,navStartLocationX=None,navStartLocationY=None,recStartLocationX=None,recStartLocationY=None,stimList=None):
            fields = ['mstime','type','item','trial','block','list_length','serialpos','probepos','locationX','locationY','navStartLocationX','navStartLocationY','recStartLocationX','recStartLocationY','stimList'];
            vals = [mstime,eventType,item,trial,block,list_length,serialpos,probepos,locationX,locationY,navStartLocationX,navStartLocationY,recStartLocationX,recStartLocationY,stimList]
            emptyDict = dict(list(zip(fields,vals)))
            return emptyDict

        def getPresDictKey(data,recItem,trialNum):
            for key in data:
                if data[key]['item'] == recItem and data[key]['type'] == 'CHEST' and data[key]['trial'] == trialNum:
                    return key

        def getPresDictKey(data,x,y,trialNum):
            for key in data:
                if data[key]['locationX'] == x and data[key]['locationY'] == y and data[key]['type'] == 'CHEST' and data[key]['trial'] == trialNum:
                    return key

        # open raw log file to read and treasure.par to write new abridged log
        sess_dir, log_file = os.path.split(raw_log_file)
        in_file = open(raw_log_file, 'r')
        out_file = open(out_file_path, 'w')

        # file to keep track of player pather
        playerPathsFile = open(os.path.join(sess_dir,"playerPaths.par"), 'w')

        # write log header
        columnOrder = ['mstime','type','item','trial','block','list_length','serialpos','probepos','locationX','locationY','navStartLocationX','navStartLocationY','recStartLocationX','recStartLocationY','stimList'];
        subject = log_file[:-4]
        out_file.write('\t'.join(columnOrder) + '\tsubject\n')

        # initial values
        treasureInfo = {}
        data = {}
        phase = None
        env_center = None
        block = 0
        radius = None
        startMS = None
        totalScore = 0
        pathMS = None
        chest = None
        x = None
        y = None

        # loop over all lines in raw log
        for s in in_file.readlines():

            s = s.replace('\r','')
            tokens = s[:-1].split('\t')
            if len(tokens)>1:

                # log beginning of session and selector radius
                if tokens[2] == 'EnvironmentPositionSelector' and tokens[4] == 'DIAMETER' and startMS is None:
                    startMS = tokens[0]
                    radius = float(tokens[5])/2.0
                    out_file.write('SESSION_START\t%s\n'%(startMS))
                    out_file.write('RADIUS\t%s\n'%(radius))

                # to keep track of running score total
                elif tokens[2] == 'Total Score' and tokens[3] == 'TEXT_MESH':
                    totalScore += int(tokens[4])

                # the beginning of a trial
                elif tokens[2] == 'Trial Info':
                    trialNum = tokens[4]
                    list_length = tokens[8]
                    isStimList = 0
                    if len(tokens)==13 and tokens[12] == 'True':
                        isStimList = 1

                # keep a dictionary of treasure chest locations
                if 'TreasureChest' in tokens[2] and tokens[3] == 'POSITION':
                    treasureInfo[tokens[2]] = {}
                    treasureInfo[tokens[2]]['pos'] = [tokens[4],tokens[6]]

                # Need environment center to later figure out if the object is on
                # same half of environment as player
                if tokens[2] == 'Experiment Info' and tokens[3] == 'ENV_CENTER':
                    env_center = [tokens[4],tokens[6]]

                # keep track of most current player position
                if tokens[2] == 'Player' and tokens[3] == 'POSITION':
                    playerPosition = (tokens[4],tokens[5],tokens[6])

                # keep track of most current player position
                if tokens[2] == 'Player' and tokens[3] == 'ROTATION':
                    playerRotation = tokens[5]

                # keep track of most current experiment phase (navigation or recall)
                elif tokens[2] == 'Trial Event':
                    if tokens[3] == 'TRIAL_NAVIGATION_STARTED':
                        phase = 'nav'
                        serialPos = 0
                        item = ''
                        navStartX = playerPosition[0]
                        navStartY = playerPosition[2]
                        itemStartMS = tokens[0]
                        pathCount = 0
                    elif tokens[3] == 'RECALL_PHASE_STARTED':
                        phase = 'rec'
                        recPos = 0
                        recItem = ''

                # navigation specific
                if phase == 'nav':

                    # when running into a treasure chest
                    if tokens[3] == 'TREASURE_OPEN':
                        chest = tokens[2]
                        presX = treasureInfo[chest]['pos'][0]
                        presY = treasureInfo[chest]['pos'][1]
                        serialPos += 1
                        mstime = tokens[0]
                        item = ''

                        # item chest or filled chest
                        if tokens[5] == 'True':
                            isItemPres = 1
                        else:
                            isItemPres = 0
                            data[mstime] = makeEmptyDict(mstime,'CHEST',None,trialNum,block,list_length,serialPos,None,presX,presY,navStartX,navStartY,stimList=isStimList)

                    # filled chest item identity
                    elif tokens[3] == 'TREASURE_LABEL':
                        item = tokens[4]
                        treasureInfo[chest]['item'] = item

                    # when the item appears from the chest
                    elif tokens[2] == item and tokens[3] == 'SPAWNED':
                        mstime = tokens[0]
                        data[mstime] = makeEmptyDict(mstime,'CHEST',item,trialNum,block,list_length,serialPos,None,presX,presY,navStartX,navStartY,stimList=isStimList)

                    # when the item is removed from screen
                    elif tokens[2] == chest and tokens[3] == 'DESTROYED':
                        itemStartMS = tokens[0]
                        pathCount += 1

                    # logging player position
                    if tokens[0] != pathMS and pathCount <= 3:
                        pathMS = tokens[0]
                        delta = int(pathMS) - int(itemStartMS)
                        playerPathsFile.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (pathMS, itemStartMS, delta, trialNum, pathCount, playerPosition[0], playerPosition[2], playerRotation))

                # recall specific
                elif phase == 'rec':

                    # start of recall phase
                    if tokens[2] == 'Trial Event' and tokens[3] == 'OBJECT_RECALL_CHOICE_STARTED':
                        recPos += 1
                        x = None
                        y = None

                    elif x is None and tokens[2] == 'ObjectRecallIndicator' and tokens[3] == 'POSITION':
                        x = tokens[4]
                        y = tokens[6]

                        isRecFromNearSide = None
                        recStartTime = tokens[0]
                        mstime = tokens[0]
                        reactionTime = None

                        key = getPresDictKey(data,x,y,trialNum)
                        if key is not None:
                            this_item = data[key]['item'].lower()
                            data[key]['recStartLocationX'] = playerPosition[0]
                            data[key]['recStartLocationY'] = playerPosition[2]
                            data[key]['probepos'] = recPos
                            pres_serialPos = data[key]['serialpos']

                        else:
                            this_item = 'trick'
                            presX = 'NaN'
                            presY = 'NaN'
                            pres_serialPos = -1
                        data[mstime] = makeEmptyDict(mstime,'PROBE',this_item,trialNum,block,list_length,pres_serialPos,recPos,x,y,navStartX,navStartY,playerPosition[0],playerPosition[2],stimList=isStimList)

                    # done with block
                    elif tokens[2] == 'Completed Block UI' and tokens[4] == 'True':
                        block += 1

        # make sure all the events are in order, and write to new file
        sortedKeys = sorted(data)
        for key in sortedKeys:
            writeToFile(out_file,data[key],subject)

        # close files
        in_file.close()
        out_file.close()
        os.chmod(out_file_path, 0o644)
        playerPathsFile.close()

        # save out total score
        scoreFile = open(os.path.join(sess_dir,"totalScore.txt"), 'w')
        scoreFile.write('%d'%(totalScore))
        scoreFile.close()

def th_test(protocol, subject, montage, experiment, session, base_dir='/data/eeg/'):
    import glob

    exp_path = os.path.join(base_dir, subject, 'behavioral', experiment, 'session_' + str(session))
    # files = {'session_log': os.path.join(exp_path, 'session_%d' % session, 'treasure.par'),
             # 'annotations': ''}
    files = {'treasure_par': os.path.join(exp_path,'treasure.par'),
             'session_log': os.path.join(exp_path, subject+'Log.txt'),
             'annotations': glob.glob(os.path.join(exp_path,'audio/*.ann'))}

    parser = THSessionLogParser(protocol, subject, montage, experiment, session, files)
    return parser

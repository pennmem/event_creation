from .base_log_parser import BaseSessionLogParser, UnknownExperimentError
from .system2_log_parser import System2LogParser
from ..viewers.recarray import strip_accents
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
            ('chestNum', -999, 'int16'),
            ('block', -999, 'int16'),
            ('locationX', -999, 'float64'),
            ('locationY', -999, 'float64'),
            ('chosenLocationX', -999, 'float64'),
            ('chosenLocationY', -999, 'float64'),
            ('navStartLocationX', -999, 'float64'),
            ('navStartLocationY', -999, 'float64'),
            ('recStartLocationX', -999, 'float64'),
            ('recStartLocationY', -999, 'float64'),
            ('isRecFromNearSide', False, 'b1'),
            ('isRecFromStartSide', False, 'b1'),
            ('reactionTime', -999, 'float64'),
            ('confidence', -999, 'int16'),
            ('radius_size', -999, 'float64'),
            ('listLength', -999, 'int16'),
            ('distErr', -999, 'float64'),
            ('normErr', -999, 'float64'),
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

        # remove msoffset field because it does not exist in the TH log
        self._fields = tuple([x for x in self._fields if x[0] != 'msoffset'])                
        self._log_header = ''
        self._radius_size = self.TH_RADIUS_SIZE        
        self._trial = -999
        self._chestNum = -999
        self._stim_list = False
        self._item_name = ''
        self._block = -999
        self._locationX = -999
        self._locationY = -999
        self._chosenLocationX = -999
        self._chosenLocationY = -999
        self._navStartLocationX = -999
        self._navStartLocationY = -999
        self._recStartLocationX = -999
        self._recStartLocationY = -999
        self._isRecFromNearSide = False
        self._isRecFromStartSide = False
        self._reactionTime = -999
        self._confidence = -999
        self._distErr = -999
        self._normErr = -999
        self._recalled = False

        # self._version = ''
        # kind of hacky, 'type' is the second entry in the header
        self._add_fields(*self._th_fields())
        self._add_type_to_new_event(
            type=self.event_header,
            CHEST=self.event_line,
            REC=self.event_line
        )
        self._add_type_to_modify_events(
            type=self.modify_header,
            REC=self.modify_rec
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
        event.session = self._session
        event.stim_list = self._stim_list
        event.radius_size = self._radius_size
        event.item_name = self._item_name
        event.block = self._block
        event.chestNum = self._chestNum
        event.locationX = self._locationX
        event.locationY = self._locationY
        event.chosenLocationX = self._chosenLocationX
        event.chosenLocationY = self._chosenLocationY
        event.navStartLocationX = self._navStartLocationX
        event.navStartLocationY = self._navStartLocationY
        event.recStartLocationX = self._recStartLocationX
        event.recStartLocationY = self._recStartLocationY
        event.isRecFromNearSide = self._isRecFromNearSide
        event.isRecFromStartSide = self._isRecFromStartSide
        event.reactionTime = self._reactionTime
        event.confidence = self._confidence
        event.distErr = self._distErr
        event.normErr = self._normErr
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
        self._item_name = split_line[ind] if split_line[ind] != 'None' else ''

        ind = self._log_header.index('trial')
        self._trial = int(split_line[ind])

        ind = self._log_header.index('block')
        self._block = int(split_line[ind])

        ind = self._log_header.index('chestNum')
        self._chestNum = int(split_line[ind])

        ind = self._log_header.index('locationX')
        self._locationX = float(split_line[ind])

        ind = self._log_header.index('locationY')
        self._locationY = float(split_line[ind])

        ind = self._log_header.index('chosenLocationX')
        self._chosenLocationX = float(split_line[ind]) if split_line[ind] != 'None' else -999

        ind = self._log_header.index('chosenLocationY')
        self._chosenLocationY = float(split_line[ind]) if split_line[ind] != 'None' else -999

        ind = self._log_header.index('navStartLocationX')
        self._navStartLocationX = float(split_line[ind])

        ind = self._log_header.index('navStartLocationY')
        self._navStartLocationY = float(split_line[ind])

        ind = self._log_header.index('recStartLocationX')
        self._recStartLocationX = float(split_line[ind]) if split_line[ind] != 'None' else -999

        ind = self._log_header.index('recStartLocationY')
        self._recStartLocationY = float(split_line[ind]) if split_line[ind] != 'None' else -999

        ind = self._log_header.index('isRecFromNearSide')
        self._isRecFromNearSide = bool(int(split_line[ind])) if split_line[ind] != 'None' else False

        ind = self._log_header.index('isRecFromStartSide')
        self._isRecFromStartSide = bool(int(split_line[ind])) if split_line[ind] != 'None' else False

        ind = self._log_header.index('reactionTime')
        self._reactionTime = float(split_line[ind]) if split_line[ind] != 'None' else -999

        ind = self._log_header.index('confidence')
        self._confidence = int(split_line[ind]) if split_line[ind] != 'None' else -999

        if 'stimList' in self._log_header:
            ind = self._log_header.index('stimList')
            self._stim_list = bool(int(split_line[ind]))

    def event_line(self, split_line):

        # set all the values in the line
        self.set_event_properties(split_line)

        # calculate distance error and normalized distance error if this is not an empty chest
        if self._item_name != '':
            # calc distance error
            xy_resp = np.array([self._chosenLocationX, self._chosenLocationY], dtype=float)
            xy_act = np.array([self._locationX, self._locationY], dtype=float)
            self._distErr = np.linalg.norm(xy_resp-xy_act)

            # calc normalized distance error
            rand_x = np.random.uniform(359.9, 409.9, 100000)
            rand_y = np.random.uniform(318.0, 399.3, 100000)
            possible_errors = np.sqrt((rand_x - xy_resp[0]) ** 2 + (rand_y - xy_resp[1]) ** 2)
            self._normErr = np.mean(possible_errors < self._distErr)

            # label recalled True if the distance error is less than the radius
            self._recalled = self._distErr < THSessionLogParser.TH_RADIUS_SIZE
        else:
            self._recalled = False
            self._distErr = -999
            self._normErr = -999

        event = self.event_default(split_line)
        return event

    def modify_rec(self, events):
        """This adds list length field to current rec event and all prior CHEST events of the current trial.
        No current way to know this ahead of time."""

        pres_events = (events['trial'] == self._trial) & (events['type'] == 'CHEST') & (events['item_name'] != '')
        pres_events_inc_empty = (events['trial'] == self._trial) & (events['type'] == 'CHEST')
        listLength = np.sum(pres_events)
        for ind in np.where(pres_events_inc_empty)[0]:
            events[ind].listLength = listLength
        events[-1].listLength = listLength
        return events

    @staticmethod
    def parse_raw_log(raw_log_file, out_file_path):
        def writeToFile(f,data,subject):
            columnOrder = ['mstime','type','item','trial','block','chestNum','locationX','locationY','chosenLocationX','chosenLocationY','navStartLocationX','navStartLocationY','recStartLocationX','recStartLocationY','isRecFromNearSide','isRecFromStartSide','reactionTime','confidence','stimList'];
            strToWrite = ''
            for col in columnOrder:
                line = data[col]
                if col != columnOrder[-1]:
                    strToWrite += '%s\t'%(line)
                else:
                    strToWrite += '%s\t%s\n'%(line,subject)
            f.write(strToWrite)

        def makeEmptyDict(mstime=None,eventType=None,item=None,trial=None,block=None,chestNum=None,locationX=None,locationY=None,chosenLocationX=None,chosenLocationY=None,navStartLocationX=None,navStartLocationY=None,recStartLocationX=None,recStartLocationY=None,isRecFromNearSide=None,isRecFromStartSide=None,reactionTime=None,confidence=None,stimList=None):
            fields = ['mstime','type','item','trial','block','chestNum','locationX','locationY','chosenLocationX','chosenLocationY','navStartLocationX','navStartLocationY','recStartLocationX','recStartLocationY','isRecFromNearSide','isRecFromStartSide','reactionTime','confidence','stimList'];
            vals = [mstime,eventType,item,trial,block,chestNum,locationX,locationY,chosenLocationX,chosenLocationY,navStartLocationX,navStartLocationY,recStartLocationX,recStartLocationY,isRecFromNearSide,isRecFromStartSide,reactionTime,confidence,stimList]
            emptyDict = dict(zip(fields,vals))
            return emptyDict

        def getPresDictKey(data,recItem,trialNum):
            for key in data:
                if data[key]['item'] == recItem and data[key]['type'] == 'CHEST' and data[key]['trial'] == trialNum:
                    return key

        # open raw log file to read and treasure.par to write new abridged log
        sess_dir, log_file = os.path.split(raw_log_file)
        in_file = open(raw_log_file, 'r')
        out_file = open(out_file_path, 'w')

        # file to keep track of player pather
        playerPathsFile = open(os.path.join(sess_dir,"playerPaths.par"), 'w')
        
        # write log header
        columnOrder = ['mstime','type','item','trial','block','chestNum','locationX','locationY','chosenLocationX','chosenLocationY','navStartLocationX','navStartLocationY','recStartLocationX','recStartLocationY','isRecFromNearSide','isRecFromStartSide','reactionTime','confidence','stimList'];
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
                            data[mstime] = makeEmptyDict(mstime,'CHEST',None,trialNum,block,serialPos,presX,presY,None,None,navStartX,navStartY,stimList=isStimList)

                    # filled chest item identity
                    elif tokens[3] == 'TREASURE_LABEL':
                        item = tokens[4]
                        treasureInfo[chest]['item'] = item                        

                    # when the item appears from the chest
                    elif tokens[2] == item and tokens[3] == 'SPAWNED':
                        mstime = tokens[0]
                        data[mstime] = makeEmptyDict(mstime,'CHEST',item,trialNum,block,serialPos,presX,presY,None,None,navStartX,navStartY,stimList=isStimList)

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
                    if tokens[2] == 'Trial Event' and tokens[3] == 'RECALL_SPECIAL':
                        recPos += 1
                        recItem = tokens[4]
                        x = None
                        y = None
                        presX = None
                        presY = None
                        isRecFromNearSide = None
                        recStartTime = tokens[0]
                        mstime = tokens[0]
                        reactionTime = None

                        key = getPresDictKey(data,recItem,trialNum)
                        data[key]['recStartLocationX'] = playerPosition[0]
                        data[key]['recStartLocationY'] = playerPosition[2]
                        presX = data[key]['locationX']
                        presY = data[key]['locationY']                        

                        # determine if navigation and recall started from the same side
                        isRecFromStartSide = 0
                        if ((float(navStartY) >= float(env_center[1]) and float(data[key]['recStartLocationY']) >= float(env_center[1])) or
                            (float(navStartY) < float(env_center[1]) and float(data[key]['recStartLocationY']) < float(env_center[1]))):
                            isRecFromStartSide = 1
                        data[key]['isRecFromStartSide'] = isRecFromStartSide

                        # determine if the target location is the same half of the environment
                        # as the player test location
                        isRecFromNearSide = 0
                        if ((float(presY) >= float(env_center[1]) and float(data[key]['recStartLocationY']) >= float(env_center[1])) or
                            (float(presY) < float(env_center[1]) and float(data[key]['recStartLocationY']) < float(env_center[1]))):
                            isRecFromNearSide = 1
                        data[key]['isRecFromNearSide'] = isRecFromNearSide

                    # confidence judgement
                    elif tokens[2] == 'Experiment' and tokens[3] == 'REMEMBER_RESPONSE':
                        confidence = 0
                        if tokens[4]=='maybe':
                            confidence = 1
                        elif tokens[4]=='yes':
                            confidence = 2

                        key = getPresDictKey(data,recItem,trialNum)
                        data[key]['confidence'] = confidence

                    # # confidence judgement, for compatabilty with old log format
                    elif tokens[2] == 'Experiment' and tokens[3] == 'DOUBLE_DOWN_RESPONSE':
                        confidence = 1
                        if tokens[4] == 'True':
                            confidence = 2
                        key = getPresDictKey(data,recItem,trialNum)
                        data[mstime]['confidence'] = confidence
                        data[key]['confidence'] = confidence

                    # resose location
                    elif tokens[2] == 'EnvironmentPositionSelector' and tokens[3] == 'CHOSEN_TEST_POSITION':
                        x = tokens[4]
                        y = tokens[6]
                        reactionTime = int(tokens[0]) - int(recStartTime)

                    # correct location
                    elif tokens[2] == 'EnvironmentPositionSelector' and tokens[3] == 'CORRECT_TEST_POSITION':
                        presX = tokens[4]
                        presY = tokens[6]

                        data[mstime] = makeEmptyDict(mstime,'REC',recItem,trialNum,block,'NaN',presX,presY,x,y,navStartX,navStartY,playerPosition[0],playerPosition[2],reactionTime=reactionTime,confidence=confidence,stimList=isStimList)
                        # fill in the presentaiton event with recall info
                        # there is probably/definitely a more efficient way to do this
                        key = getPresDictKey(data,recItem,trialNum)
                        data[key]['chosenLocationX'] = x
                        data[key]['chosenLocationY'] = y
                        data[key]['reactionTime'] = reactionTime
                        data[key]['confidence'] = confidence
                        data[mstime]['chestNum'] = data[key]['chestNum']
                        data[mstime]['isRecFromNearSide'] = isRecFromNearSide
                        data[mstime]['isRecFromStartSide'] = isRecFromStartSide

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
        os.chmod(out_file_path, 0644)
        playerPathsFile.close()

        # save out total score
        scoreFile = open(os.path.join(sess_dir,"totalScore.txt"), 'w')
        scoreFile.write('%d'%(totalScore))
        scoreFile.close()

def th_test(protocol, subject, montage, experiment, session, base_dir='/data/eeg/'):
    exp_path = os.path.join(base_dir, subject, 'behavioral', experiment)
    # files = {'session_log': os.path.join(exp_path, 'session_%d' % session, 'treasure.par'),
             # 'annotations': ''}
    files = {'treasure_par': os.path.join(exp_path,'treasure.par'),
             'session_log': os.path.join(exp_path, subject+'Log.txt')}
 
    parser = THSessionLogParser(protocol, subject, montage, experiment, session, files)    
    return parser


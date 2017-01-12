# NOTE: MODIFY THE FOLLOWING LINE 
# (and ONLY the following line)
# TO SWITCH LANGUAGE BETWEEN ENGLISH ('EN')
# AND SPANISH ('SP')
LANGUAGE='EN'

EXPERIMENT_NAME = 'FR1'
VERSION_NUM = '2.04'

# NEW IN VERSION 1.02
require_labjack = True

# Overridden in sconfig files
do_stim = False

# Control PC
control_pc = True

# Number of sessions per subject
numSessions = 5

# Number of trials/lists per session
numTrials = 25

# Number of words per trial/list
listLen = 12

# Pause+Jitter after orienting stim before first word
PauseBeforeWords = 1000
JitterBeforeWords = 400

PauseBeforeRecall = 500
JitterBeforeRecall = 200

# Word Font size (percentage of vertical screen)
wordHeight = .1

# Duration word is on the screen
wordDuration = 1600

# ISI+Jitter after word is cleared from the screen
ISI = 750
Jitter = 250

# Duration of recall in ms
recallDuration = 30000

# Beep at start and end of recording (freq,dur,rise/fall)
startBeepFreq = 800
startBeepDur = 500
startBeepRiseFall = 100
stopBeepFreq = 400
stopBeepDur = 500
stopBeepRiseFall = 100

# Orienting Stimulus text
orientText = '+'
recallStartText = '*******'

countdownMovie = 'video_%s/countdown.mpg'%LANGUAGE
introMovie = 'video_%s/instructions.mpg' #LANGUAGE WILL BE PLACED HERE BY PLAY_INTRO.py


# Math distractor options
doMathDistract = True
continuousDistract = False
MATH_numVars = 3
MATH_maxNum = 9
MATH_minNum = 1
MATH_maxProbs = 50
MATH_plusAndMinus = False
MATH_minDuration_Practice = 30000
MATH_minDuration = 20000
MATH_textSize = .1
MATH_correctBeepDur = 500
MATH_correctBeepFreq = 400
MATH_correctBeepRF = 50
MATH_correctSndFile = None
MATH_incorrectBeepDur = 500
MATH_incorrectBeepFreq = 200
MATH_incorrectBeepRF = 50
MATH_incorrectSndFile = None

# Word pool to use
wp = 'pools_%s/RAM_wordpool.txt' % LANGUAGE
noAcc_wp = 'RAM_wordpool_noAcc.txt'
practice_wordList = 'pools_%s/word-pool_PRACTICE.txt'  # LANGUAGE PLACED BY FR.PY
presentationType = 'text'  # image, sound, text
presentationAttribute = 'name'  # attribute to use to create the text

# Instructions text file
pre_practiceList = 'text_%s/pre_practiceList.txt'  # LANGUAGE PLACED BY FR.PY
post_practiceList = 'text_%s/post_practiceList.txt'  # LANGUAGE PLACED BY FR.PY

# make stim form
makeStimForm = False
trialsPerPage = 7

# Default font
defaultFont = 'fonts/Verdana.ttf'

# Realtime configuration
# ONLY MODIFY IF YOU KNOW WHAT YOU ARE DOING!
# HOWEVER, IT SHOULD BE TWEAKED FOR EACH MACHINE
doRealtime = True
rtPeriod = 120
rtComputation = 9600
rtConstraint = 1200

# Used to find the file for constructing word lists with similar words
similarityFile = 'wordpool_generation/RAM_FR_LSA.csv'

fastConfig = False

if fastConfig:
    ISI = 10
    Jitter = 0

    PauseBeforeWords = 10;
    JitterBeforeWords = 10;

    wordDuration = 20;
    recallDuration = 10;

    MATH_minDuration_Practice = 10;
    MATH_minDuration = 10;

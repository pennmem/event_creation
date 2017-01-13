# event_creation

This repository contains python utilities for the submission of 
data to centralized location, and the subsequent processing and QA of
the data.

The term "database" will be used throughout. Note that the "database" is
actually a collection of directories and is not a true database. 

## Installation

The easiest way to install the necessary python packages is with conda.

The file conda_environment.yml specifies the packages necessary for 
execution. Create the new environment with
`conda env create -f conda_environment.yml` .

Note that this will create a new environment titled `event_creation`. 

The only additional package necessary for execution not installed by 
conda is ptsa: https://github.com/maciekswat/ptsa_new. 


## Use

There are four top-level executable scripts to use for the processing 
of data:
 - submit: Submits source data to the database and performs processing, 
   creating, for example, events or a montage file
 - split: GUI to split data or extract channel labels from a raw eeg
   file (creating a jacksheet)
 - extract_raw: GUI to extract sync pulses from a raw eeg recording. 
   Can also be used to view single-channel data without splitting a file
 - extract_split: Same as extract_raw but works on already-split data

Each of these files activates the `event_creation` environment before
executing the appropriate python script.

The remainder of this documentation discusses only the `submit` 
functionality and design.

Submission accepts a number of command-line options, viewable with 
`./submit -h`. At present, these options are as follows:
```
usage: convenience.py [-h] [--debug] [--montage-only] [--change-experiment]
                      [--change-session] [--allow-convert] [--force-convert]
                      [--force-events] [--force-eeg] [--force-montage]
                      [--clean-only] [--aggregate-only] [--do-compare]
                      [--json JSON_FILE] [--build-db DB_NAME] [--view-only]
                      [--show-plots] [--set-input INPUTS] [--path PATHS]

optional arguments:
  -h, --help           show this help message and exit
  --debug              Prints debug information to terminal during execution
  --montage-only       Imports a localization or montage instead of importing
                       events
  --change-experiment  Signals that the name of the experiment changes on
                       import. Defaults to true for PS* behavioral
  --change-session     Signals that the session number changes on import
  --allow-convert      Attempt to convert events from matlab if standard
                       events creation fails
  --force-convert      ONLY attempt conversion of events from matlab and skips
                       standard import
  --force-events       Forces events creation even if no changes have occurred
  --force-eeg          Forces eeg splitting even if no changes have occurred
  --force-montage      Forces montage change even if no changes have occurred
  --clean-only         ONLY cleans the database. Removes empty folders and
                       folders without processed equivalent
  --aggregate-only     ONLY aggreate index files. Run if any paths have been
                       manually changed
  --do-compare         Compare created JSON events to MATLAB events and undo
                       import if they do mot match
  --json JSON_FILE     Imports all sessions from specified JSON file. Build
                       JSON file with option --build-db
  --build-db DB_NAME   Builds a JSON database which can later be imported.
                       Current options are "sharing" or "ram"
  --view-only          View information about already-submitted subjects
  --show-plots         Show plots of fit and residuals when aligning data (not
                       available when running with sudo)
  --set-input INPUTS   Set inputs for subsequent submission (KEY=VALUE). Will
                       not prompt for these inputs if provided. Available
                       options are: code, protocol, experiment, localization,
                       montage, original_experiment, session,
                       original_session, subject
  --path PATHS         Override the path set in config file (KEY=VALUE).
                       Available options are: events_root, rhino_root,
                       loc_db_root, db_root, data_root
```
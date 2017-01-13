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
 - **submit**: Submits source data to the database and performs processing, 
   creating, for example, events or a montage file
 - **split**: GUI to split data or extract channel labels from a raw eeg
   file (creating a jacksheet)
 - **extract_raw**: GUI to extract sync pulses from a raw eeg recording. 
   Can also be used to view single-channel data without splitting a file
 - **extract_split**: Same as extract_raw but works on already-split data

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

The options `--set-input` and `--paths` accept arguments of the form 
`KEY1=VALUE1:KEY2=VALUE2` such that one can set the path to rhino's root
and the path to the root of the database with:

`./submit --path rhino_root=/path/to/rhino:db_root=/path/to/db`


In the most basic use case, creation of events can be executed with just 

`./submit`
 
(you will be prompted for the details of submission).
 
 Creation of a montage can be executed with 

`./submit --montage`

#### Use cases for additional arguments:
- **`--debug`**: More verbose output. Useful when something goes wrong
- **`--change-session`**: Two very common uses are:
    - A session has been skipped, but should imported as subsequent sessions
    - A subject's montage has changed (e.g. R1001P -> R1001P_1). 
       If sessions 0 and 1 were run under R1001P, and then a third session was
       run under R1001P_1, the "original session" of the new session would be
       0, while the "new session" would be 1
- **`--allow-convert`**: If events creation from source files fails, but matlab 
    events exist, this flag will allow those events to be converted
    directly from python to json
- **`--force-*`**: If events creation code has changed, this will allow files
    to be re-processed, even if no changes have occurred in the source files
- **`--clean`**: If there are empty folders in the database due to deletions 
    or processing failures, this will prune those directories
- **`--aggregate`**: If a processed directory has been deleted manually, this will
    force re-indexing such that `r1.json` (or the equivalent index) will be up 
    to date 
- **`--do-compare`**: If doing a mass-import, this should be enabled! It requires
    that the created JSON events match their matlab equivalent.
- **`--json <FILE>`**: Imports multiple subjects with specifications defined
    in a json file. Useful for mass imports (such as data sharing)
- **`--build-db <NAME>`**: Builds a json file which can subsequently be imported.
    The only (useful) currently-implemented name is `sharing`, which will generate a 
    file with all sessions meant to be shared in the RAM Phase I Data sharing.
- **`--view`**: Useful if you want to know what sessions have been imported for a 
     given subject
- **`--show-plots`**: If alignment fails, enable this option for a graphical view 
     of the details of alignment. *NOTE: This cannot be run with "sudo", so 
     a user-writable database location must be specified*
- **`--set-input`**: If running a same or similar set of inputs, this can save time. 
    Could also be useful  for automation, calling from crontabs, etc. Use syntax such as:
    - `./submit --set subject=R1001P:session=0:experiment=FR1`
- **`--paths`**: Useful for testing (export to non-official location), if rhino is mounted,
    or if creating a database for export
    
    
## Description of code

The execution of submission can be broken up into two parts 
(***transferring*** and ***processing***), which in combination make up the process of 
***submitting***.

### Transferring

#### Transfer config files
At the heart of the transfer process are the transfer configuration files 
(stored in `submission/transfer_inputs`). Following the definition of paths
and defaults at the top of the file, each config contains a list of files,
links, and directories that will be used in processing. 

The following is a sample file definition, specifying the location of pyepl's
`session.log` file:
```
  - name: session_log
    << : *FILE
    groups: [ transfer, verbal ]
    origin_directory: *BEH_SESS_DIR
    origin_file: session.log
    destination: logs/session.log
```
File definitions can contain the following fields:
- **`name`**: A key by which a processing pipeline can refer to the transferred file
- **`type`**: Must be one of [file, link, directory]. This field is not set manually,
            but rather inherited from the default_file entry, via the \<\< yaml operator
- **`groups`**: Different experiments require that different files are transferred. 
            The `groups` field allows this to happen. When creating events, each pipeline is
            put into the groups with its protocol name, experiment name, and experiment 
            type. (e.g., a session of FR1 would automatically have groups `r1`, `FR` and `FR1`).
            Additional groups can also be assigned, such as `verbal` for pyepl tasks, or 
            `system_#` specifying the RAM system number. A transferer must belong to all
            of a file's specified groups in order for it to transfer that file.
- **`origin_directory`**: Specifies the path to the directory containing the file(s) to
            transfer. The python formatting syntax can be used to insert variables into
            the path. ***NOTE**: wildcard expansion does NOT work here*. 
- **`origin_file`**: Specifies the path to the file to be transferred relative to the 
            origin directory. Again, python formatting can be used to substitute variables.
            Wildcard substitution (UNIX style) is acceptable in this field.
- **`destination`**: Specifies where the file should be transferred to, relative to the 
            root of the *`current_source`* folder.
            
The following fields can also be specified, but are optional:
- **`multiple`**: Set to `True` if multiple files can be transferred. If specified, the 
            `destination` field sets a directory rather than a filename, and files
            are transferred with their original name.
- **`required`**: Set to `False` if import of the file is optional for the current process
- **`checksum_contents`**: `True` by default. If `False`, only the filename will be checked
            when determining if changes have occurred. Useful for large files.
- **`files`**: Only for items of type `directory`. Specifies a list of the containing
            files which are to be transferred.

#### Transferer object and associated classes
Initialization of a file transfer begins with the creation of a `Transeferer` 
(`submission.transferer.Transferer`), which accepts a transfer configuration
yaml file, a list of the groups that the transferer belongs to, the destination
where the transferer should place the files, and keyword arguments to replace 
in the origin directories and filenames specified in yaml.

The `Transferer` initializes an instance of a `TransferConfig` 
(`submission.transfer_config.TransferConfig`), representing the yaml file, which
in turn contains `TransferFile`s, representing each file to transfer.

When the transfer is initiated, the `Transferer` will copy each required item to 
their destination directories, and will roll back the transfer entirely if it
fails at any point.
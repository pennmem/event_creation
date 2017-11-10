# Submission Utility (event creation)

This repository contains python utilities for the submission of 
data to centralized location, and the subsequent processing and QA of
the data.

The term "database" will be used throughout. Note that the "database" is
actually a collection of directories and is not a true database. 


## Installation

After cloning the repository, execute the following two commands:
```
git submodule init
git submodule update
```
This will initialize and check out the neurorad submodule.

After installation, the necessary dependencies must be installed.
The easiest way to install the necessary python packages is with conda.

The file conda_environment.yml specifies the packages necessary for 
execution. Create the new environment with
`conda env create -f conda_environment.yml` .

Note that this will create a new environment titled `event_creation`. 

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
                      [--force-localization] [--force-dykstra]
                      [--clean-only] [--aggregate-only] [--do-compare]
                      [--json JSON_FILE] [--build-db DB_NAME] [--view-only]
                      [--show-plots] [--set-input INPUTS] [--path PATHS]

optional arguments:
  -h, --help           show this help message and exit
  --debug              Prints debug information to terminal during execution
  --montage-only       Imports a montage instead of importing
                       events
  --localization-only  Imports a localization instead of importing events
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
  --force-localization Forces localization change even if no changes have occurred
                       (Note: will *NOT* rerun Dykstra correction)
  --force-dykstra      When re-running localization, also rerun Dykstra correction
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
     - example: `./submit --show-plots --paths db_root='.'`
     - Note that this will **not** actually submit to the real database
- **`--set-input`**: If running a same or similar set of inputs, this can save time. 
    Could also be useful  for automation, calling from crontabs, etc. Use syntax such as:
    - `./submit --set subject=R1001P:session=0:experiment=FR1`
- **`--paths`**: Useful for testing (export to non-official location), if rhino is mounted,
    or if creating a database for export
    
    
## Description of code

The execution of submission can be broken up into two parts 
(***transferring*** and ***processing***), which in combination make up the process of 
***submitting***.

Submission consists of the creation of a `TransferPipeline` 
(`submission.pipelines.TransferPipeline`), which consists of a single `Transferer`, 
and any number of `PipelineTask`s (which perform the processing). Pipeline tasks can 
be marked as non-critical, in which case a failure is allowed. By default, however,
failure of an individual task rolls back the entire transfer process.

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

Functions that instantiate different types of Transferers are located in 
 `submission.transferer` (`generate_ephys_transferer()`, `generate_import_montage_transferer()`,
 `genereate_session_transferer()`)

### Processing

At the present time, there are two types of processing pipelines that have been 
implemented: montage and events creation. Event creation can be further broken down
into "building" and "converting" events, each of which consist of an EEG Splitting
pipeline, and an Events Creation pipeline

#### Montage creation
Montage creation simply copies the information from the `loc_db` (localization database,
currently located in RAM_maint's home folder) into the submission database. 

#### Building events
As EEG splitting and events creation are separable tasks, they are organized into 
two separate pipelines which are run successively. These pipelines are instantiated
with functions in `subbmission.pipelines` (`build_split_pipeline()`, `build_events_pipeline()`).

Both pipelines start by determining the groups to which the transfer belongs
(`submission.pipelines.determine_groups()`), then proceeds with the creation
of the processed files.

##### EEG splitting
Splitting is initialized from the `SplitEEGTask` (`submission.tasks.SplitEEGTask`) object, which
instantiates an `EEGReader` (`readers.eeg_reader.EEGReader`) specific to the file format 
that is being split (EDF, Nihon-Kohdon, HDF, and others).

The splitting task first groups simultaneous recordings (only applicable to Blackrock ns2 files),
then splits each of the files into the processed "noreref" directory. 

**TODO:** ns[3-6] files are recorded at the same time as ns2 files, and contain micro recordings.
Some way of splitting these files and generating a secondary eegoffset field has to be created.

Finally, the splitting task creats a `sources.json` file, which lists the basename and
meta-information for each of the split files, so they can be later aligned with behavioral events. 

##### Event creation
The `EventCreationTask` (`submission.tasks.EventCreationTask`) orchestrates the creation
and aligning of events. It performs any or all of the following steps:
- **parsing**: creates a child of `BaseLogParser` (`parsers.base_log_parser.BaseLogParser`), which
  parses the primary log file for the experiment and creates a set of `np.recarray` events
  
- **alignment**: creates an aligner which fills in the `eegoffset` field of the events. There is no
  common parent class for alignment, as the methods differed widely enough I found it difficult
  to find common code.
  
- **adding stim**: For System 2 and System 3, the aligner can add stimulation events to the recarray.
  In theory, adding stim has nothing to do with alignment and should be moved elsewhere. In practice,
  they use the same log file.
  
- **artifact detection**: *LTP only* 

- **cleaning**: Performs any post-alignment removal or changing of events, if necessary

###### Parsing

The `BaseLogParser` defines the steps for parsing a log file. It defines a template for the fields
that are to be created in the record array (`_BASE_FIELDS`), as well as the fields relevant to
stimulation (`_STIM_FIELDS`). The parser functions by splitting the log file into units (lines
for epl, list elements for JSON) and determining a type for each log unit. Each type is then 
mapped to a function that defines the corresponding event to be created for that unit.

For example, the `PSSys3LogParser` (which extends `BaseSys3LogParser`), registers the following
event creation handlers:

```
self._add_type_to_new_event(
    STIM=self._event_skip, # Skip because it is added later with alignment
    SHAM=self.event_default,
    PAUSED=self.event_paused,
)
```

`STIM` events are skipped entirely, as they are handled by the Aligner class (for aforementioned 
bad reasons). `SHAM` events create a new event of type `SHAM` with no other modifications, and
`PAUSED` events call the event_paused() constructor:

```
def event_paused(self, event_json):
    if event_json['event_value'] == 'OFF':
        return False
    event = self.event_default(event_json)
    event.type = 'AD_CHECK'
    return event
```
`event_paused()` returns False to signal that the event should be skipped in the case that the
"pause" turned off (experiment is resumed), and otherwise returns the default event, but 
modifies the event type to be `AD_CHECK`.

The file `parsers.base_log_parser` is heavily commented and will be of great use if extending 
this functionality

###### Alignment
The `Aligner` set of classes add the eegoffset and eegfile fields to a set of record-array events.
While they share common outputs, the methods for alignment different widely enough that extracting common
functionality didn't seem worth it, so each file implements the call to `align()` separately.

The call to `align()` returns a new copy of the events with eegoffset and eegfile filled in.

All aligners rely on the `eeg_sources` file (created during eeg splitting) to determine the start
times of the files. 

The code in `alignment.system2` and `alignment.system1` are well commented and can be refereced
if interested in the specifics of the alignment method. The following are a few notes on alignment
in each system

*System 1:* 
* Alignment depends on the files `eeg_log` and `sync_pulses`, the former of which 
    lists the times at which the sync pulse was sent on the laptop, and the latter is a list of
    the samples at which eeg pulses were extracted from the raw EEG file.
* To extract sync pulses, use the `pulse_extraction` tools.
* The general method is to find a "start window" and "end window" - periods of time at the start
    and end of the `eeg_log` that can be found (via diffs) in the `sync_pulses` file. Once those
    start and end windows are located, it fits a straight line between them. There is no real error
    checking for good alignment.
* **TODO**: The MATLAB aligntool had the ability to align multiple source files. This does not.

*System 2*:
* System 2 alignment is done with either `System2TaskAligner` (for most events) or 
    `System2HostAligner` (for PS2(.1), because it doesn't use the task laptop
* The call to `align()` takes a `start_type`: events that occur before this type 
    appears in the events file do not have to be aligned
* The `System2TaskAligner` gets coefficients between the task and host, and 
    host and neuroport, then aligns from task to host.
* If `--show-plot` is specified, execution will pause until the plot of the fit is closed.
     
*System 3*:
* System 3 alignment reuses much of the code from system 2. 

###### Adding stim
The System 2 and System 3 alignment can add stimulation into the events, as stim is specified
in the events on the host PC, not on the task laptop. As such, the regressions used
to calculate `eegoffset` must be applied backwards to find the appropriate `mstime` at which 
stimulation took place.

The parameter `persistent_fields` specifies those fields in the recarray which persist across
stim events. For example, if stim occurs during a word, the stim event should still specify
that that word was onscreen. 

###### Artifact detection
(Not covered in this document)

###### Cleaning
After events have been aligned, the parser can optionally implement a `clean_events()` function
which takes in the events, and returns a copy with some events modified. This can be useful 
for post-hoc marking of, e.g., which words received stim.


#### Converting events
EEG and events creation are again broken into two pipelines, built with 
`submission.pipelines.build_convert_eeg_pipeline` and `submission.pipelines.build_convert_events_pipeline`.

These pipelines read in the initial MATLAB events, then convert these events directly to 
recarrays, using one of the classes in `parsers.mat_converter`.

### Comparing
If events are built from scratch and MATLAB events exist, the submission utility has the 
ability to compare the existing MATLAB events to the recarray events that are generated.


The file `tests.test_event_creation` is terrible, and contains all of the exceptions that 
were necessary to get the matlab events to agree with the recarray events. Most of these 
are individual differences due to the reprocessing of data, or due to mistakes in the original 
data. 

## Additional notes

* Some reorganization of the files and directories into more sensible units would be nice. The
    directory `readers` should probably just be `eeg`, and `submission.convenience` should probably
    just be `submission.submission`.
    
* pulse_extraction and pulse_extraction_2 contain almost identical codebases, and should be 
    combined into a single file. They differ in that one reads split eeg and one reads raw eeg.
* Additional command line options can be specified in `config.config.yml`
* The python `logging` class is at the root of the `loggers.logger`. It simply sets up defaults
    so that information will be printed in the right way, and will be logged to a rotating file
    in the `protocols` database
* There is an extra layer of abstraction in the input process through the file `submission.automation`.
    This file specifies the builders for different pipelines, and helps to keep track of errors
    and successes in the case in which multiple pipelines are created and processed at the same time.
  

# The Neuroradiology Localization Pipeline
#### (A Quick Guide)

# Installation
1. Clone the `event_creation` repository:
```
git clone https://github.com/pennmem/event_creation.git
```
2. Initialize the `neurorad` submodule:
```
cd event_creation
git submodule init
git submodule update
```
3. Create the necessary conda environment:
```
conda env create -f conda_environment.yml
```

# Usage
To run the localization pipeline:
```
./submit --localization-only
```
If the pipeline has been run on a subject, and none of the input files
have changed, then this command will not rerun the pipeline. To force
the pipeline to rerun, use the following options:
```
   --force-localization: Force the pipeline to rerun in the absence
                         of a change in the underlying files. Note that
                         this will NOT redo the Dykstra correction

   --force-dykstra:      Force the pipeline to rerun a previously
                         computed Dykstra correction.
```
# Settings
By default, the pipeline script writes everything to `/protocols`, a directory
only writable by the `RAM_maint` user. If you do not have access to
the `RAM_maint` account and wish to run the pipeline, you will need to
change the database directory, which is set in the file
`submission/configuration/config.yml` and is controlled by the
`db_root` setting.

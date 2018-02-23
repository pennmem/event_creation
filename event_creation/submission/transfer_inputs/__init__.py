import os.path

__all__ = ['TRANSFER_INPUTS_DIR','TRANSFER_INPUTS']

TRANSFER_INPUTS_DIR = __path__[0]

TRANSFER_INPUTS = {
    'behavioral': os.path.join(TRANSFER_INPUTS_DIR, 'behavioral_inputs.yml'),
    'ephys': os.path.join(TRANSFER_INPUTS_DIR, 'ephys_inputs.yml'),
    'montage': os.path.join(TRANSFER_INPUTS_DIR, 'montage_inputs.yml'),
    'localization': os.path.join(TRANSFER_INPUTS_DIR, 'localization_inputs.yml'),
}

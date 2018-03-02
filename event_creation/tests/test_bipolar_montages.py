from ..submission.convenience import run_montage_import
from ..submission.configuration import config
import os
osp = os.path
from ptsa.data.readers import TalReader,CMLEventReader,JsonIndexReader,EEGReader
import numpy as np
import re
from ..submission.log import logger


bipolar_subjects = ['R1370E','R1364C','R1377M']
test_index_root = osp.join(config.paths.rhino_root,'scratch','leond')
reference_index_root = config.paths.rhino_root
pairs_path_template = '{protocols_root}/protocols/r1/subjects/{subject}/localizations/0/montages/0/neuroradiology/current_processed/pairs.json'


def make_bipolar_montage(code,):
    config.paths.db_root = test_index_root
    inputs = {}
    subject = re.sub(r'_.*', '', code)
    inputs['code'] = code
    inputs['subject']=subject
    inputs['montage'] = '0.0'
    inputs['reference_scheme'] = 'bipolar'
    inputs['protocol'] = 'r1'
    try:
        os.symlink(osp.join(reference_index_root,'protocols','r1','subjects',subject,'localizations','0','neuroradiology'),
                   osp.join(test_index_root,'protocols','r1','subjects',subject,'localizations','0','neuroradiology'))
    except OSError as ose:
        if ose.errno != 17:
            raise ose
    return run_montage_import(inputs,force=True)




if __name__ == '__main__':
    for subject in bipolar_subjects:
        logger.set_subject(subject,'R1')
        sucess, _ =make_bipolar_montage(subject)
        if sucess:
            tal = TalReader(filename=pairs_path_template.format(protocols_root=test_index_root,
                                                                subject= subject))
            new_bipolar_pairs = tal.get_bipolar_pairs()
            jr = JsonIndexReader(osp.join(reference_index_root,'protocols','r1.json'))
            task_event_files = list(jr.aggregate_values('task_events',subject=subject,montage=0))
            events = CMLEventReader(filename=task_event_files[0]).read()
            eeg = EEGReader(events=events[:1],channels = np.array([]),start_time=0.0,end_time=0.1).read()
            hdf_bipolar_pairs = eeg.bipolar_pairs.values
            new_bipolar_pairs = new_bipolar_pairs.astype(hdf_bipolar_pairs.dtype)
            if not np.in1d(hdf_bipolar_pairs,new_bipolar_pairs).all():
                logger.info('\n\n%s missing from new_bipolar_pairs\n\n'%str(hdf_bipolar_pairs[~np.in1d(hdf_bipolar_pairs,new_bipolar_pairs)]))
            if not np.in1d(new_bipolar_pairs,hdf_bipolar_pairs).all():
                logger.info('\n\n%s missing from new_bipolar_pairs\n\n'%str(new_bipolar_pairs[~np.in1d(new_bipolar_pairs,hdf_bipolar_pairs)]))



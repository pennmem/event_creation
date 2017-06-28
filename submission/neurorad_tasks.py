import os
import json

from neurorad.localization import Localization
from neurorad import vox_mother_converter, calculate_transformation, add_locations,brainshift_correct

from .log import logger
from .tasks import PipelineTask


class LoadVoxelCoordinatesTask(PipelineTask):


    def __init__(self, subject, localization, is_new, critical=True):
        super(LoadVoxelCoordinatesTask, self).__init__(critical)
        self.name = 'Loading {} voxels for {}, loc: {}'.format('new' if is_new else 'old', subject, localization)
        self.is_new = is_new

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        if self.is_new:
            vox_file = files['voxel_coordinates']
        else:
            vox_file = os.path.join(self.pipeline.source_dir, 'converted_voxel_coordinates.json')
            vox_mother_converter.convert(files, vox_file)

        localization = Localization(vox_file)

        self.pipeline.store_object('localization', localization)


class CalculateTransformsTask(PipelineTask):

    def __init__(self, subject, localization, critical=False):
        super(CalculateTransformsTask, self).__init__(critical)
        self.name = 'Calculate transformations {} loc: {}'.format(subject, localization)

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        localization = self.pipeline.retrieve_object('localization')
        calculate_transformation.insert_transformed_coordinates(localization, files)


class CorrectCoordinatesTask(PipelineTask):

    def __init__(self, subject, localization, overwrite=False,critical=False):
        super(CorrectCoordinatesTask, self).__init__(critical)
        self.name = 'Correcting coordinates {} loc {}'.format(subject, localization)
        self.subject=subject
        self.freesurfer_dir = '/data/eeg/freesurfer/subjects/{subject}'.format(subject=subject)
        self.outfolder = '/home1/leond/temp' # TODO: fix this
        self.overwrite=overwrite

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        localization = self.pipeline.retrieve_object('localization')
        fsfolder = outfolder = self.pipeline.source_dir
        brainshift_correct.brainshift_correct(localization,self.subject,
                                              outfolder=outfolder,fsfolder=fsfolder,
                                              overwrite=self.overwrite)

class AddContactLabelsTask(PipelineTask):

    def __init__(self, subject, localization, critical=True):
        super(AddContactLabelsTask, self).__init__(critical)
        self.name = 'Add labels {} loc {}'.format(subject, localization)

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        localization = self.pipeline.retrieve_object('localization')

        logger.info("Adding Autoloc")
        add_locations.add_autoloc(files, localization)

class AddMNICoordinatesTask(PipelineTask):

    def __init__(self, subject, localization, critical=True):
        super(AddMNICoordinatesTask, self).__init__(critical)
        self.name = 'Add MNI {} loc {}'.format(subject, localization)

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        localization = self.pipeline.retrieve_object('localization')

        logger.info("Adding MNI")
        add_locations.add_mni(files, localization)

class WriteFinalLocalizationTask(PipelineTask):

    def _run(self, files, db_folder):
        localization = self.pipeline.retrieve_object('localization')

        logger.info("Writing localization.json file")
        self.create_file(os.path.join(db_folder, 'localization.json',), localization.to_jsons(), 'localization', False)


class CreateMontageTask(PipelineTask):
    def __init__(self,subject,localization,montage,critical=True):
        super(CreateMontageTask, self).__init__(critical=critical)
        self.subject=subject
        self.localization_num = localization
        self.montage_num = montage
        self.contacts_dict = {}
        self.pairs_dict = {}


    def _run(self, files, db_folder):
        """
        The intended design:
            * Go through the leads, extract the contacts
            * Add port numbers to each contact
            * store contacts in dictionary by contact label: contacts.json
            * ***
            * Go through leads, extract pairs
            * store pairs in dictionary by pair label : pairs.json

        This method doesn't write any new information; everything should be populated by previous tasks

        :param files:
        :param db_folder:
        :return:
        """
        self.read_jacksheet(files['jacksheet'])
        self.load_localization(files['localization'])
        self.build_contacts_dict(db_folder)
        self.build_pairs_dict(db_folder)
        # raise NotImplementedError



    def read_jacksheet(self,jacksheet):
        nums_to_labels = {}
        labels_to_nums = {}
        with open(jacksheet) as jack_file:
            for line in jack_file:
                num,label = line.strip().split()
                num=int(num)
                label=label.upper()
                nums_to_labels[num] = label
                labels_to_nums[label] = num
        self.nums_to_labels = nums_to_labels
        self.labels_to_nums = labels_to_nums

    def load_localization(self,localization_file):
        with open(localization_file) as loc_fid:
            self.localization = json.load(loc_fid)

    def build_contacts_dict(self,db_folder):
        contacts = {}
        leads = self.localization['leads']
        types  = {}
        for lead in leads:
            contacts.update(
                {x['name'].upper():x for x in leads[lead]['contacts']}
            )
            types.update({x['name'].upper():leads[lead]['type'] for x in leads[lead]['contacts']})

        for contact in contacts.keys():
            if 'channel' not in contacts[contact]:
                logger.warn('Contact %s not found in localization'%contact)
                contacts[contact] = {}
                del contacts[contact]
            else:
                contacts[contact]['channel'] = self.labels_to_nums[contact]
                contacts[contact]['type'] = types[contact]
        self.contacts_dict[self.subject] = {'contacts':contacts}
        self.create_file(os.path.join(db_folder,'contacts.json'),
                         json.dumps(self.contacts_dict,indent=2,sort_keys=True),'contacts',False)

    def build_pairs_dict(self,db_folder):
        leads = self.localization['leads']
        pairs = {}
        types={}
        for lead in leads:
            pairs.update({'-'.join([y.upper() for y in x['names']]):x for x in leads[lead]['pairs']})
            types.update({'-'.join([y.upper() for y in x['names']]):leads[lead]['type'] for x in leads[lead]['pairs']})
        for pair in pairs.keys():
            (name1,name2) = [x.upper() for x in pairs[pair]['names']]
            if name1 in self.labels_to_nums and name2 in self.labels_to_nums:
                pairs[pair]['channel_1'] =self.labels_to_nums[name1]
                pairs[pair]['channel_2'] = self.labels_to_nums[name2]
                pairs[pair]['type']=types[pair]
            else:
                logger.warn('Pair %s not found in localization'%pair)
                pairs[pair]={}
                del pairs[pair]
        self.pairs_dict[self.subject] = {'pairs':pairs}
        self.create_file(os.path.join(db_folder,'pairs.json'),
                         contents=json.dumps(self.pairs_dict,indent=2,sort_keys=True),
                         label='pairs',index_file=False)






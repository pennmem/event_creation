import os
import json

from neurorad.localization import Localization
from neurorad import vox_mother_converter, calculate_transformation, add_locations

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

    def __init__(self, subject, localization, critical=True):
        super(CalculateTransformsTask, self).__init__(critical)
        self.name = 'Calculate transformations {} loc: {}'.format(subject, localization)

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        localization = self.pipeline.retrieve_object('localization')
        calculate_transformation.insert_transformed_coordinates(localization, files)


class CorrectCoordinatesTask(PipelineTask):

    def __init__(self, subject, localization, critical=False):
        super(CorrectCoordinatesTask, self).__init__(critical)
        self.name = 'Correcting coordinates {} loc {}'.format(subject, localization)

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        localization = self.pipeline.retrieve_object('localization')
        # TODO : Dysktra method here

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
        self.create_file(os.path.join(db_folder, 'localization.json',), localization.to_jsons(), 'localization', True)


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
            * pair off contacts
            * Do something to the coordinates?
            * This might be Dorian's method
            * store pairs in dictionary by pair label : pairs.json

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
                {x['name']:x for x in leads[lead]['contacts']}
            )
            types.update({x['name']:leads[lead]['type'] for x in leads[lead]['contacts']})
        for label in self.labels_to_nums:
            contacts[label]['channel'] = self.labels_to_nums[label]
            contacts[label]['type'] = types[label]
        self.contacts_dict[self.subject] = {'contacts':contacts}
        self.create_file(os.path.join(db_folder,'contacts.json'),
                         json.dumps(self.contacts_dict,indent=2,sort_keys=True),'contacts',True)


    def build_pairs_dict(self,db_folder):
        leads = self.localization['leads']
        for lead in leads:
            pairs  = [x['names'] for x in leads['lead']['pairs']]



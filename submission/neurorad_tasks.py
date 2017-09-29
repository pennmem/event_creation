import os
import json
from neurorad.json_cleaner import clean_json_dumps

from neurorad.localization import Localization
from neurorad import (vox_mother_converter, calculate_transformation, add_locations,brainshift_correct,make_outer_surface)
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
            logger.debug('Loaded file %s'%vox_file)

        localization = Localization(vox_file)

        self.pipeline.store_object('localization', localization)


class CreateDuralSurfaceTask(PipelineTask):
    def __init__(self,subject,localization,critical=False):
        super(CreateDuralSurfaceTask, self).__init__(critical=critical)
        self.name='Creating Dural Surface for {}, loc {}'.format(subject,localization)

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        for side in ['left','right']:
            dural_side = '%s_dural'%side
            pial_file = '%s_pial'%side
            if dural_side not in files:
                logger.info('Constructing %s dural surface'%side)
                dural_file = make_outer_surface.make_smoothed_surface(files[pial_file])
                files[dural_side] = dural_file

class GetFsAverageCoordsTask(PipelineTask):

    def __init__(self,subject,localization,critical=False):
        super(GetFsAverageCoordsTask, self).__init__(critical)
        self.name = 'Find FSAverage coordinates {} loc: {}'.format(subject, localization)

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        localization  =self.pipeline.retrieve_object('localization')
        subject_surf_dir = files['surf']
        avg_surf_dir = files['avg_surf']
        contacts = localization.get_contacts()
        for coord_type in ('raw','corrected'):
            fs_coords = localization.get_contact_coordinates('fs',contacts,coord_type)
            fsaverage_coords = calculate_transformation.map_to_average_brain(fs_coords,subject_surf_dir,avg_surf_dir)
            localization.set_contact_coordinates('fsaverage',contacts,fsaverage_coords,coord_type)




class CalculateTransformsTask(PipelineTask):

    def __init__(self, subject, localization, critical=False):
        super(CalculateTransformsTask, self).__init__(critical)
        self.name = 'Calculate transformations {} loc: {}'.format(subject, localization)

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        localization = self.pipeline.retrieve_object('localization')
        Torig,Norig,talxfm = calculate_transformation.insert_transformed_coordinates(localization, files)
        self.pipeline.store_object('Torig',Torig)
        self.pipeline.store_object('Norig',Norig)
        self.pipeline.store_object('talxfm',talxfm)


class CorrectCoordinatesTask(PipelineTask):

    def __init__(self, subject, localization, overwrite=False,critical=False):
        super(CorrectCoordinatesTask, self).__init__(critical)
        self.name = 'Correcting coordinates {} loc {}'.format(subject, localization)
        self.subject=subject
        self.overwrite=overwrite

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        localization = self.pipeline.retrieve_object('localization')
        tc = self.pipeline.transferer.transfer_config
        fsfolder =  self.pipeline.source_dir
        outfolder = os.path.join( tc._raw_config['directories']['localization_db_dir'].format(**tc.kwargs),'brainshift_correction')
        try:
            os.mkdir(outfolder)
        except OSError:
            pass
        brainshift_correct.brainshift_correct(localization,self.subject,
                                              outfolder=outfolder,fsfolder=fsfolder,
                                              overwrite=self.overwrite)
        Torig = self.pipeline.retrieve_object('Torig')
        Norig = self.pipeline.retrieve_object('Norig')
        talxfm = self.pipeline.retrieve_object('talxfm')
        calculate_transformation.invert_transformed_coords(localization,Torig,Norig,talxfm)


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

class AddManualLocalizationsTask(PipelineTask):
    def __init__(self,subject,localization,critical=True):
        super(AddManualLocalizationsTask, self).__init__(critical)
        self.name = 'Add manual {} loc {}'.format(subject,localization)

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        localization = self.pipeline.retrieve_object('localization')

        logger.info('Adding manual localizations')
        add_locations.add_manual_locations(files,localization)

class WriteFinalLocalizationTask(PipelineTask):

    def _run(self, files, db_folder):
        localization = self.pipeline.retrieve_object('localization')

        logger.info("Writing localization.json file")
        self.create_file(os.path.join(db_folder, 'localization.json',), localization.to_jsons(), 'localization', True)


class CreateMontageTask(PipelineTask):

    FIELD_NAMES_TABLE = {
        'fs':'indiv',
        'fsaverage':'tal',
        'mni':'mni',
        'ct_voxel':'vox',
    }

    ATLAS_NAMES_TABLE = {
        'dk':'indiv',
        'whole_brain':'mni',
    }



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
        for lead in leads:
            for contact in leads[lead]['contacts']:
                atlas_dict = {}
                for k,v in self.FIELD_NAMES_TABLE.items():
                    try:
                        coords = contact['coordinate_spaces'][k]['corrected']
                    except KeyError:
                        coords  = contact['coordinate_spaces'][k]['raw']
                    atlas_dict[v] = {}
                    for i,axis in enumerate(['x','y','z']):
                        atlas_dict[v][axis] = coords[i]
                        atlas_dict['region']=None
                for k,v in self.ATLAS_NAMES_TABLE.items():
                    atlas_dict[v]['region'] = contact['atlases'][k]
                try:
                    contact_dict = {
                        'atlases':atlas_dict,
                        'channel':self.labels_to_nums[contact['name']],
                        'code':contact['name'],
                        'type':leads[lead]['type']
                    }
                except KeyError:
                    logger.info('Contact %s not found in jacksheet' % contact)
                    continue
                contacts[contact['name']]=contact_dict
        self.contacts_dict[self.subject] = {'contacts':contacts}
        self.create_file(os.path.join(db_folder,'contacts.json'),
                         clean_json_dumps(self.contacts_dict,indent=2,sort_keys=True),'contacts',False)

    def build_pairs_dict(self,db_folder):
        leads = self.localization['leads']
        pairs = {}
        types={}
        for lead in leads:
            pairs.update({'-'.join([y.upper() for y in x['names']]):x for x in leads[lead]['pairs']})
            types.update({'-'.join([y.upper() for y in x['names']]):leads[lead]['type'] for x in leads[lead]['pairs']})
        logger.debug('Building pairs.json')
        for pair in pairs.keys():
            logger.debug(str(pair))
            (name1,name2) = [x.upper() for x in pair.split('-')]
            if name1 in self.labels_to_nums and name2 in self.labels_to_nums:
                pairs[pair]['channel_1'] =self.labels_to_nums[name1]
                pairs[pair]['channel_2'] = self.labels_to_nums[name2]
                pairs[pair]['type']=types[pair]
            else:
                logger.info('Pair %s not contained in jacksheet'%pair)
                pairs[pair]={}
                del pairs[pair]
        self.pairs_dict[self.subject] = {'pairs':pairs}
        self.create_file(os.path.join(db_folder,'pairs.json'),
                         contents=clean_json_dumps(self.pairs_dict,indent=2,sort_keys=True),
                         label='pairs',index_file=False)






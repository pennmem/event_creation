import os
import requests
import bptools.pairs

from ..neurorad.json_cleaner import clean_json_dumps
from ..neurorad.localization import Localization,InvalidContactException
from ..neurorad import (vox_mother_converter, calculate_transformation, add_locations,
                      brainshift_correct,make_outer_surface,map_mni_coords,)

from .log import logger
from .tasks import PipelineTask
from .exc import WebAPIError

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
        out_folder = os.path.dirname(files['right_pial'])
        for side in ['left','right']:
            dural_side = '%s_dural'%side
            pial_file = '%s_pial'%side
            if dural_side not in files:
                logger.info('Constructing %s dural surface'%side)
                dural_file = make_outer_surface.make_smoothed_surface(files[pial_file],out_folder)
                files[dural_side] = dural_file

class GetFsAverageCoordsTask(PipelineTask):

    def __init__(self,subject,localization,critical=False):
        super(GetFsAverageCoordsTask, self).__init__(critical)
        self.name = 'Find FSAverage coordinates {} loc: {}'.format(subject, localization)

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        localization  =self.pipeline.retrieve_object('localization')
        contacts = localization.get_contacts()
        for coord_type in ('raw','corrected'):
            fs_coords = localization.get_contact_coordinates('fs',contacts,coord_type)

            fsaverage_coords,fsaverage_labels = calculate_transformation.map_to_average_brain(fs_coords,files['left_pial'],files['right_pial'],
                                                                             files['left_sphere'],files['right_sphere'])
            localization.set_contact_coordinates('fsaverage',contacts,fsaverage_coords,coord_type)
            localization.set_contact_labels('dkavg',contacts,fsaverage_labels,)




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

    def __init__(self, subject, localization, code,overwrite=False,critical=False):
        super(CorrectCoordinatesTask, self).__init__(critical)
        self.name = 'Correcting coordinates {} loc {}'.format(subject, localization)
        self.subject = subject
        self.overwrite = overwrite
        self.code = code

    def _run(self, files, db_folder):
        logger.set_label(self.name)
        localization = self.pipeline.retrieve_object('localization')
        tc = self.pipeline.transferer.transfer_config
        fsfolder =  self.pipeline.source_dir
        outfolder = os.path.join( tc._raw_config['directories']['localization_db_dir'].format(**tc.kwargs),'brainshift_correction')
        imaging_root = tc._raw_config['directories']['imaging_subject_dir'].format(**tc.kwargs)
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
        map_mni_coords.add_corrected_mni_cordinates(localization,imaging_root,self.code)


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
        pairs = localization.get_pairs(localization.get_lead_names())
        for coord_space in localization.VALID_COORDINATE_SPACES:
            for type in 'raw','corrected':
                localization.get_pair_coordinates(coord_space,pairs,type)

        logger.info("Writing localization.json file")
        self.create_file(os.path.join(db_folder, 'localization.json',), localization.to_jsons(), 'localization', False)

class BrainBuilderWebhookTask(PipelineTask):

    def __init__(self,subject,critical=False):
        super(BrainBuilderWebhookTask, self).__init__(critical=critical)
        self.subject = subject


    def _run(self, files, db_folder):
        from .configuration import paths

        api_url = paths.brainviz_url
        parameters={'subject':self.subject,
                    'username':'cmlbrainbuilder',
                    'password':'BoBtheBuilder'}
        response  = requests.post(api_url,data=parameters)
        if response.status_code != 200:
            raise WebAPIError('Request failed with message %s' % response.text)


class CreateMontageTask(PipelineTask):

    FIELD_NAMES_TABLE = {
        'ind':('fs','raw'),
        'ind.corrected':('fs','corrected'),
        'avg':('fsaverage','raw'),
        'avg.corrected':('fsaverage','corrected'),
        'mni':('mni','raw'),
        'vox':('ct_voxel','raw'),
        'hcp':('hcp','raw')
    }

    ATLAS_NAMES_TABLE = {
        'dk':'ind',
        'whole_brain':'mni',
        'hcp':'hcp',
        'manual':'stein',
    }

    def __init__(self,subject,localization,montage,reference_scheme='bipolar',critical=True):
        super(CreateMontageTask, self).__init__(critical=critical)
        self.subject=subject
        self.localization_num = localization
        self.montage_num = montage
        self.reference_scheme = reference_scheme
        self.contacts_dict = {}
        self.pairs_dict = {}
        self.pairs_frame = None # pandas.DataFrame

    def _run(self, files, db_folder):
        """
        The intended design:
            * Go through the leads, extract the contacts
            * Add port numbers to each contact
            * store contacts in dictionary by contact label: contacts.json
            * ***
            * store pairs in dictionary by pair label : pairs.json

        This method doesn't write any new information; everything should be populated by previous tasks

        :param files:
        :param db_folder:
        :return:
        """
        self.read_jacksheet(files['jacksheet'])
        self.localization = Localization(files['localization'])
        logger.info('Creating contacts.json')
        self.build_contacts_dict(db_folder,'contacts')
        logger.info('Creating pairs.json')
        self.build_contacts_dict(db_folder,'pairs')


    def read_jacksheet(self,jacksheet):
        nums_to_labels = {}
        labels_to_nums = {}
        with open(jacksheet) as jack_file:
            for line in jack_file.read().splitlines():
                num,label = line.strip().split()
                num=int(num)
                label=label
                nums_to_labels[num] = label
                labels_to_nums[label] = num
        self.nums_to_labels = nums_to_labels
        self.labels_to_nums = labels_to_nums
        if self.reference_scheme == 'bipolar':
            self.pairs_frame = bptools.pairs.create_pairs(jacksheet)

    def build_contacts_dict(self,db_folder,name):
        contacts = {}
        if name == 'pairs' and self.reference_scheme == 'bipolar':
            for i in self.pairs_frame.index:
                atlas_dict = {}
                pair = self.pairs_frame.loc[i]
                for pairs_name, (loc_name,loc_t) in list(self.FIELD_NAMES_TABLE.items()):
                    coords = self.localization.get_pair_coordinate(loc_name,pair[['label1','label2']].values,loc_t)
                    atlas_dict[pairs_name]={}
                    for i,axis in enumerate('xyz'):
                        atlas_dict[pairs_name][axis] = coords.squeeze()[i]
                        atlas_dict[pairs_name]['region'] = None
                for loc_name,pairs_name in list(self.ATLAS_NAMES_TABLE.items()):
                    if pairs_name not in atlas_dict:
                        atlas_dict[pairs_name] = {}
                        for axis in 'xyz':
                            atlas_dict[pairs_name][axis] = None
                    try:
                        atlas_dict[pairs_name]['region'] = self.localization.get_pair_label(loc_name,pair[['label1','label2']].values)
                    except InvalidContactException as e:
                        logger.warn('Could not find %s for pair %s-%s'%(pairs_name,pair['label1'],pair['label2']))
                        atlas_dict[pairs_name]['region'] = None

                contact_dict = {
                    'atlases': atlas_dict,
                    'channel_1':self.labels_to_nums[pair['label1']],
                    'channel_2':self.labels_to_nums[pair['label2']],
                    'code': '-'.join(pair[['label1','label2']]),
                    'is_stim_only': False,
                    'type_1': self.localization.get_contact_type(pair['label1']),
                    'type_2': self.localization.get_contact_type(pair['label2']),
                    }
                contacts[contact_dict['code']] = contact_dict

        else:
            leads = self.localization._contact_dict['leads']
            for lead in leads:
                for contact in leads[lead][name]:
                    atlas_dict = {}
                    try:
                        for contact_name,(loc_name,loc_t) in list(self.FIELD_NAMES_TABLE.items()):
                            coords = [None,None,None]
                            if loc_name in contact['coordinate_spaces']:
                                coords = contact['coordinate_spaces'][loc_name][loc_t]
                            atlas_dict[contact_name] = {}
                            for i,axis in enumerate(['x','y','z']):
                                atlas_dict[contact_name][axis] = coords[i]
                                atlas_dict[contact_name]['region']=None
                        for contact_name,loc_name in list(self.ATLAS_NAMES_TABLE.items()):
                            if loc_name not in atlas_dict:
                                atlas_dict[loc_name]={}
                                for axis in 'xyz':
                                    atlas_dict[loc_name][axis] = None
                            atlas_dict[loc_name]['region'] = contact['atlases'].get(contact_name)

                        if name=='contacts':
                            contact_dict = {
                                'atlases':atlas_dict,
                                'channel':self.labels_to_nums[contact['name']],
                                'code':contact['name'],
                                'type':leads[lead]['type']
                            }
                        elif name=='pairs':
                            contact_dict={
                                'atlases':atlas_dict,
                                'channel_1':self.labels_to_nums[contact['names'][0]],
                                'channel_2':self.labels_to_nums[contact['names'][1]],
                                'code':'-'.join(contact['names']),
                                'is_stim_only':False,
                                'type_1':leads[lead]['type'],
                                'type_2':leads[lead]['type'],
                            }
                        else:
                            raise RuntimeError('bad name')
                        contacts[contact_dict['code']]=contact_dict
                    except KeyError as ke:
                        if name=='contacts':
                            logger.info('Contact %s not found in jacksheet' %(contact['name']))
                        else:
                            logger.info('Contacts %s not found in jacksheet'%(contact['names']))
        logger.debug('%s entries in %s dict'%(len(contacts),name))
        self.contacts_dict[self.subject] = {name:contacts}
        self.contacts_dict['version'] = self.localization.version
        self.create_file(os.path.join(db_folder,'%s.json'%name),
                         clean_json_dumps(self.contacts_dict,indent=2,sort_keys=True),name,False)
        logger.info('%s.json written to %s'%(name,db_folder))





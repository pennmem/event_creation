from .base_log_parser import BaseLogParser,BaseSys3_1LogParser
import pandas as pd
import json

class BaseHostPCLogParser(BaseSys3_1LogParser):
    """
    This class implements the basic logic for producing event structures from event_log.json files written by the
    host PC in system 3.1+

    In order to account for possible changes to the structure of RAMulator messages, particularly between experiments
    that use pyEPL and those using UnityEPL, this parser and parsers inheriting from it should depend as *little* as
    possible on the existence of particular fields.
    """

    _TYPE_FIELD = 'event_label'


    def __init__(self, protocol, subject, montage, experiment, session, files,
                 primary_log='event_log', allow_unparsed_events=False, include_stim_params=False):
        super(BaseHostPCLogParser, self).__init__(protocol,subject,montage,experiment,session,files,
                                                  primary_log,allow_unparsed_events,include_stim_params)



    def _read_primary_log(self):
        with open(self._primary_log,'r') as primary_log:
            contents = pd.DataFrame.from_records(json.load(primary_log)['events'])
            contents = pd.concat([contents,pd.DataFrame.from_records([msg['data'] for msg in contents.msg_stub])])
            return [e.to_dict() for _, e in contents.iterrows()]

    def parse(self):
        return BaseLogParser.parse(self)





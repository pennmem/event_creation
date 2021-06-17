from base_log_parser import BaseLogParser

class BaseElememLogParser(BaseLogParser):
    """
    Class for parsing Elemem / System4 event.log files. As a design choice, Elemem logs all messages
    received over network cable from Unity tasks (or other sources, in theory)

    This should obviate the need for alignment, as Elemem records these events in the mstime/unix time according
    to its own clock, which is the same clock as the EEG recording.
    """
    def __init__(self, protocol, subject, montage, experiment, session, files, primary_log='event_log'):
        if primary_log not in files:
            primary_log = 'event_log'

        BaseLogParser.__init__(self, protocol, subject, montage, experiment, session, files, primary_log=primary_log,
                               allow_unparsed_events=True)
        self._files = files
        self._trial = -999

    def _get_raw_event_type(self, event_json):
        return event_json['type']

    def parse(self):
        try:
            return super(BaseUnityLTPLogParser, self).parse()
        except Exception as exc:
            traceback.print_exc(exc)
            logger.warn('Encountered error in parsing %s session %s: \n %s: %s' % (self._subject, self._session,
                                                                                   str(type(exc)), exc.message))
            raise exc

    def _read_unityepl_log(self, filename):
        """
        Read events from the UnityEPL format (JSON strings separated by
        newline characters).

        :param str filename: The path to the session log you wish to parse.
        """
        # Read session log
        df = pd.read_json(filename, lines=True)
        # Create a list of dictionaries, where each dictionary is the information about one event
        events = [e.to_dict() for _, e in df.iterrows()]
        # Replace spaces in event type names with underscores
        for i, e in enumerate(events):
            events[i]['type'] = e['type'].replace(' ', '_')
            if e['type'] == 'stimulus' and 'displayed text' in e['data']:
                events[i]['type'] = 'stimulus_display'
        return events

    def _read_primary_log(self):
        evdata = self._read_unityepl_log(self._primary_log)
        return evdata

    def event_default(self, event_json):
        event = self._empty_event
        event.mstime = event_json['time']
        event.type = event_json['type']
        event.trial = self._trial
        return event

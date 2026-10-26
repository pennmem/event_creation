import numpy as np
import pandas as pd
from .valuecourier_log_parser import ValueCourierSessionLogParser


class VCFROPSessionLogParser(ValueCourierSessionLogParser):

    # VCFROP item-value recalls are spoken (audio) and transcribed as numeric tokens
    # (0-9) in the recall annotation files. Extend the base word-only token set to also
    # accept a run of digits in the recalled-item column. Scoped to VCFROP only.
    MATCHING_ANN_REGEX = r'\d+(\.\d+)?\s+-?\d+\s+(([A-Z]+)|(<>)|(\[\?\?\?\])|(\d+))'

    def __init__(self, protocol, subject, montage, experiment, session, files):
        super().__init__(protocol, subject, montage, experiment, session, files)

        # Override parent's field-name indirection so inherited handlers
        # (e.g. add_object_presentation_begins) write to the VCFROP columns.
        self._itemvalue_field = "itemvaluecorrect"
        self._actualvalue_field = "avgvaluecorrect"

        self._add_fields(
            ('itemvalueguess',    -999, 'int16'),
            ('avgvalueguess', -999, 'int16'),
        )

        # Only the average value recall comes from the JSON log (it is typed). Item value
        # recalls are spoken/audio and are derived from numeric annotation tokens instead
        # (see _classify_recall and the recall-flow overrides below).
        self._add_type_to_new_event(
            value_recall = self.add_avg_value_recall,
        )

    def _add_valuerecall_field(self):
        pass

    def _add_compensation_field(self):
        pass

    def _add_itemvalue_field(self):
        self._add_fields(('itemvaluecorrect', -999, 'int16'))

    def _add_actualvalue_field(self):
        self._add_fields(('avgvaluecorrect', -999, 'float32'))

    def add_receive_compensation(self, evdata):
        event = self.event_default(evdata)
        event.type = 'FINAL_COMPENSATION'
        event.multiplier = evdata['data']['multiplier']
        return event

    def add_avg_value_recall(self, evdata):
        event = self.event_default(evdata)
        event.type = "AVG_VALUE_RECALL" if not self.practice else "PRACTICE_AVG_VALUE_RECALL"
        event.trial = evdata['data']['trial number']
        event.avgvalueguess = int(self.stringify_list(evdata['data']['typed response']))
        if 'actual value' in evdata['data']:
            event.avgvaluecorrect = evdata['data']['actual value']
        else:
            event.avgvaluecorrect = -1
            print(
                f"Missing 'actual value' field in AVG_VALUE_RECALL event for subject " +
                f"{self._subject}, session {self._session}, trial {event.trial}"
            )
        return event

    def _new_rec_event(self, recall, evdata):
        # The Courier parser strips trailing '.' and '1' characters from every
        # annotation token (to drop a stray ".1" suffix on words). For a spoken
        # value that turns 41 into 4 and 1 into nothing, so keep digit tokens intact.
        new_event = super()._new_rec_event(recall, evdata)
        token = str(recall[-1]).strip()
        if token.isdigit():
            new_event["item"] = token
        return new_event

    def _classify_recall(self, new_event):
        """
        Strictly route a parsed annotation recall to exactly one population:

        - a purely numeric token (a spoken item value) becomes ITEM_VALUE_RECALL and
          populates itemvalueguess; it is never emitted as a word recall.
        - any other (semantic/word) token becomes REC_WORD / REC_WORD_VV and never sets
          itemvalueguess.

        Returns (new_event, is_item_value). is_item_value lets callers skip intrusion
        bookkeeping, which is meaningless for a number.
        """
        item = str(new_event["item"]).strip()
        if item.isdigit():
            new_event.type = "ITEM_VALUE_RECALL" if not self.practice else "PRACTICE_ITEM_VALUE_RECALL"
            new_event.itemvalueguess = int(item)
            return new_event, True

        new_event.type = 'REC_WORD_VV' if "<>" in new_event["item"] else 'REC_WORD'
        return new_event, False

    # Override the inherited recall flows so numeric (audio) tokens route to
    # ITEM_VALUE_RECALL while word tokens stay REC_WORD / REC_WORD_VV.
    def modify_free_recall(self, events):
        rec_start_event = events[-1]
        try:
            ann_outputs = self._parse_ann_file("final recall")
        except:
            ann_outputs = self._parse_ann_file("final free-0")
            ann_outputs = ann_outputs + self._parse_ann_file("final free-1")

        for recall in ann_outputs:
            new_event = self._new_rec_event(recall, rec_start_event)
            new_event, is_item_value = self._classify_recall(new_event)
            if not is_item_value:
                new_event = self._identify_intrusion(events, new_event)
            new_event.trial = -999  # to match old events
            events = np.append(events, new_event).view(np.recarray)

        return events

    def modify_rec_start(self, events):
        rec_start_event = events[-1]

        if self.practice:
            # Practice parsing not implemented for Courier / NICLS
            return events
        ann_outputs = self._parse_ann_file(str(self._trial))

        for recall in ann_outputs:
            new_event = self._new_rec_event(recall, rec_start_event)
            new_event, is_item_value = self._classify_recall(new_event)
            if not is_item_value:
                new_event = self._identify_intrusion(events, new_event)
                if new_event.intrusion > 0:
                    events.intruded[(events["type"] == 'WORD') & (events["item"] == new_event["item"])] = 1
                elif new_event.intrusion == 0:
                    events.recalled[(events["type"] == 'WORD') & (events["item"] == new_event["item"])] = 1
            events = np.append(events, new_event).view(np.recarray)

        return events

    def modify_after_final_compensation(self, events):
        full = pd.DataFrame.from_records(events)

        # ITEM_VALUE_RECALL events come from numeric tokens in the recall annotation
        # files. A spoken value belongs to the word the participant said just before
        # it, so pair each one with the most recent REC_WORD of the same recall period
        # (same trial, earlier time). Pairing by item number is not possible: WORD
        # events carry no item number, and Penn TotalRecall writes -1 for a token that
        # is not in the wordpool. A value with no preceding word in its recall period
        # (or one following an extra-list intrusion) stays unpaired, with
        # itemvaluecorrect left at -999.
        ivr_mask = full.type.isin(["ITEM_VALUE_RECALL", "PRACTICE_ITEM_VALUE_RECALL"])
        ivr_rows = list(full.index[ivr_mask])
        rec_mask = full.type.isin(["REC_WORD", "PRACTICE_REC_WORD"])
        word_mask = full.type.isin(["WORD", "PRACTICE_WORD"])

        copy_cols = ["serialpos", "store", "storepointtype", "itemvaluecorrect", "itemno"]
        ivr_to_word = {}
        for ivr_idx in ivr_rows:
            trial = full.at[ivr_idx, "trial"]
            when = full.at[ivr_idx, "mstime"]
            earlier = full.index[rec_mask & (full.trial == trial) & (full.mstime <= when)]
            if len(earlier) == 0:
                continue
            rec_idx = earlier[-1]
            # the presentation this recall refers to (PLIs point at an earlier trial)
            intrusion = full.at[rec_idx, "intrusion"]
            if intrusion < 0:
                continue
            pres_trial = trial - intrusion
            candidates = full.index[word_mask & (full.item == full.at[rec_idx, "item"])
                                    & (full.trial == pres_trial)]
            if len(candidates) == 0:
                continue
            word_idx = candidates[0]
            ivr_to_word[ivr_idx] = rec_idx
            for col in copy_cols:
                full.at[ivr_idx, col] = full.at[word_idx, col]
            full.at[ivr_idx, "item"] = full.at[word_idx, "item"]
            full.at[ivr_idx, "intrusion"] = intrusion

        # If the participant corrected a value ("forty... forty-one"), keep the last
        # value given for that recall.
        by_word = {}
        for ivr_idx, rec_idx in ivr_to_word.items():
            by_word.setdefault(rec_idx, []).append(ivr_idx)
        drops = []
        for word_idx, ivrs in by_word.items():
            if len(ivrs) <= 1:
                continue
            keep = max(ivrs, key=lambda i: full.at[i, "mstime"])
            drops.extend(i for i in ivrs if i != keep)
        if drops:
            full = full.drop(index=drops).reset_index(drop=True)

        words = full[full.type == "WORD"]
        avg_recalls = full[full.type == "AVG_VALUE_RECALL"]

        word_trial_to_storepointtype = words.set_index("trial")["storepointtype"].to_dict()
        word_trial_to_recalled       = words.set_index("trial")["recalled"].to_dict()
        for event_type in ["AVG_VALUE_RECALL", "REC_WORD", "REC_WORD_VV"]:
            subset = full[full.type == event_type]
            for idx, row in subset.iterrows():
                trial = row["trial"]
                if trial in word_trial_to_storepointtype:
                    full.at[idx, "storepointtype"] = word_trial_to_storepointtype[trial]
                if trial in word_trial_to_recalled:
                    full.at[idx, "recalled"] = word_trial_to_recalled[trial]

        trial_to_correct = avg_recalls.set_index("trial")["avgvaluecorrect"].to_dict()
        trial_to_guess   = avg_recalls.set_index("trial")["avgvalueguess"].to_dict()
        for event_type in ["WORD", "REC_WORD", "REC_WORD_VV"]:
            subset = full[full.type == event_type]
            for idx, row in subset.iterrows():
                trial = row["trial"]
                if trial in trial_to_correct:
                    full.at[idx, "avgvaluecorrect"] = trial_to_correct[trial]
                if trial in trial_to_guess:
                    full.at[idx, "avgvalueguess"] = trial_to_guess[trial]

        final_comp = full[full.type == "FINAL_COMPENSATION"]
        full["multiplier"] = final_comp["multiplier"].values[0]

        word_ev = full[full.type == "WORD"]
        full["primacybuf"]       = word_ev["primacybuf"].values[0]
        full["recencybuf"]       = word_ev["recencybuf"].values[0]
        full["numingroupchosen"] = word_ev["numingroupchosen"].values[0]

        return full.to_records(
            index=False,
            column_dtypes={x: str(y[0]) for x, y in events.dtype.fields.items()},
        )

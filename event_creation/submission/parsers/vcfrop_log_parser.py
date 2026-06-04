import numpy as np
import pandas as pd
from . import dtypes
from .valuecourier_log_parser import ValueCourierSessionLogParser


class VCFROPSessionLogParser(ValueCourierSessionLogParser):
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

        self._add_type_to_new_event(
            value_recall      = self.add_avg_value_recall,
            item_value_recall = self.add_item_value_recall,
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

    def add_item_value_recall(self, evdata):
        event = self.event_default(evdata)
        event.type = "ITEM_VALUE_RECALL" if not self.practice else "PRACTICE_ITEM_VALUE_RECALL"
        event.trial = evdata['data']['trial number']
        event.itemvalueguess = int(self.stringify_list(evdata['data']['typed response']))
        return event

    def modify_after_final_compensation(self, events):
        full = pd.DataFrame.from_records(events)

        word_mask = full.type.isin(["WORD", "PRACTICE_WORD"])
        word_positions = np.flatnonzero(word_mask.values)

        ivr_mask = full.type.isin(["ITEM_VALUE_RECALL", "PRACTICE_ITEM_VALUE_RECALL"])
        ivr_rows = list(full.index[ivr_mask])

        copy_cols = ["serialpos", "store", "storepointtype", "itemvaluecorrect", "itemno"]
        ivr_to_word = {}
        for ivr_idx in ivr_rows:
            prior = word_positions[word_positions < ivr_idx]
            if len(prior) == 0:
                continue
            word_idx = int(prior[-1])
            ivr_to_word[ivr_idx] = word_idx
            for col in copy_cols:
                full.at[ivr_idx, col] = full.at[word_idx, col]

        by_word = {}
        for ivr_idx, word_idx in ivr_to_word.items():
            by_word.setdefault(word_idx, []).append(ivr_idx)
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

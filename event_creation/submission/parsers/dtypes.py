"""
Datatype specifications for use by parser classes,
to make sure that different parsers for the same experiment return arrays with the same fields.
Fields are specified as (name, default_value,dtype_string)
"""



# Defaults

base_fields = (
    ('protocol', '', 'U64'),
    ('subject', '', 'U64'),
    ('montage', '', 'U64'),
    ('experiment', '', 'U64'),
    ('session', -1, 'int16'),
    ('type', '', 'U64'),
    ('mstime', -1, 'int64'),
    ('msoffset', -1, 'int16'),
    ('eegoffset', -1, 'int64'),
    ('eegfile', '', 'U256'),
    ('phase','','U16')
)

stim_fields = (
        ('anode_number', -1, 'int16'),
        ('cathode_number', -1, 'int16'),
        ('anode_label', '', 'U64'),
        ('cathode_label', '', 'U64'),
        ('amplitude', -1, 'float16'),
        ('pulse_freq', -1, 'int16'),
        ('n_pulses', -1, 'int16'),
        ('burst_freq', -1, 'int16'),
        ('n_bursts', -1, 'int16'),
        ('pulse_width', -1, 'int16'),
        ('stim_on', False, bool),
        ('stim_duration', -1, 'int16'),
        ('biomarker_value', -1, 'float64'),
        ('host_time',-999,'float64'),
        ('id', '', 'U64'),
        ('position', '', 'U64'),
        ('_remove', True, 'b'),  # This field is removed before saving, and it used to mark whether it should be output
                                 # to JSON
)

courier_fields = (
    ('trial', -999, 'int16'), 
    ('serialPos', -999, 'int16'), 
    ('item', '-999', 'U64'), 
    ('store', '-999', 'U64'), 
    ('storeX', -999, 'float16'), 
    ('storeZ', -999, 'float16'), 
    ('presX', -999, 'float16'), 
    ('presZ', -999, 'float16'), 
    ('itemno', -999, 'int16'), 
    ('recalled', -999, 'int16'), 
    ('intruded', -999, 'int16'), 
    ('finalrecalled', -999, 'int16'), 
    ('rectime', -999, 'float32'), 
    ('intrusion', -999, 'int16'), 
    ('correctPointingDirection', -999, 'float16'), 
    ('submittedPointingDirection', -999, 'float16')
)

repFR_fields = (
    ('repeats', -999, 'int16'),
    ('is_repeat', False, 'b1'),
    ('is_stim', False, 'b1'),
    ('stim_list', False, 'b1'),
    ('list', -999, 'int16'),
    ('serialpos', -999, 'int16'),
    ('item_name', '', 'U64'),
    ('item_num', -999, 'int16'),
    ('recalled', False, 'b1'),
    ('intruded', 0, 'int16'),
    ('rectime', -999, 'int32'),
    ('intrusion', -999, 'int16'),
)

# FR

fr_fields = (
            ('list', -999, 'int16'),
            ('serialpos', -999, 'int16'),
            ('item_name', 'X', 'U64'),
            ('item_num', -999, 'int16'),
            ('recalled', False, 'b1'),
            ('intrusion', -999, 'int16'),
            ('exp_version', '', 'U64'),
            ('stim_list', False, 'b1'),
            ('is_stim', False, 'b1'),
            ('rectime',-999,'int16'),
            # Recognition stuff goes here
            ('recognized', -999, 'int16'),
            ('rejected', -999, 'int16'),
            ('recog_resp', -999, 'int16'),
            ('recog_rt', -999, 'int16'),
)

# catFR
category_fields = (
        ('category','X','U64'),
        ('category_num',-999,'int16'),
)

#PAL

pal_fields = (
            ('list', -999, 'int16'),
            ('serialpos', -999, 'int16'),
            ('probepos', -999, 'int16'),
            ('study_1', '', 'U16'),
            ('study_2', '', 'U16'),
            ('cue_direction', -999, 'int16'),
            ('probe_word', '', 'U16'),
            ('expecting_word', '', 'U16'),
            ('resp_word', '', 'U16'),
            ('correct', -999, 'int16'),
            ('intrusion', -999, 'int16'),
            ('resp_pass', 0 , 'int16'),
            ('vocalization', -999, 'int16'),
            ('RT', -999, 'int16'),
            ('exp_version', '', 'U16'),
            ('stim_type', '', 'U16'),
            ('stim_list', 0, 'b1'),
            ('is_stim', False, 'b1'),
)

# Math

math_fields = (
            ('list', -999, 'int16'),
            ('test', -999, 'int16', 3),
            ('answer', -999, 'int16'),
            ('iscorrect', -999, 'int16'),
            ('rectime', -999, 'int32'),
        )

# LTP

ltp_fields = (('eogArtifact', -1, 'int8'),)

ltpFR2_fields = (
            ('trial', -999, 'int16'),
            ('serialpos', -999, 'int16'),
            ('begin_distractor', -999, 'int16'),
            ('final_distractor', -999, 'int16'),
            ('begin_math_correct', -999, 'int16'),
            ('final_math_correct', -999, 'int16'),
            ('item_name', '', 'U16'),
            ('item_num', -999, 'int16'),
            ('recalled', False, 'b1'),
            ('intruded', 0, 'int16'),
            ('rectime', -999, 'int32'),
            ('intrusion', -999, 'int16'),
            ('eogArtifact', -1, 'int8')
)

ltpFR_fields = (
        ('trial', -999, 'int16'),
        ('studytrial', -999, 'int16'),
        ('listtype', -999, 'int16'),
        ('serialpos', -999, 'int16'),
        ('distractor', -999, 'int16'),
        ('final_distractor', -999, 'int16'),
        ('math_correct', -999, 'int16'),
        ('final_math_correct', -999, 'int16'),
        ('task', -999, 'int16'),
        ('resp', -999, 'int16'),
        ('rt', -999, 'int16'),
        ('recog_resp', -999, 'int16'),
        ('recog_conf', -999, 'int16'),
        ('recog_rt', -999, 'int32'),
        ('item_name', '', 'U16'),
        ('item_num', -999, 'int16'),
        ('recalled', False, 'b1'),
        ('intruded', 0, 'int16'),
        ('finalrecalled', False, 'b1'),
        ('recognized', False, 'b1'),
        ('rectime', -999, 'int32'),
        ('intrusion', -999, 'int16'),
        ('color_r', -999, 'float16'),
        ('color_g', -999, 'float16'),
        ('color_b', -999, 'float16'),
        ('font', '', 'U32'),
        ('case', '', 'U8'),
        ('rejected', False, 'b1'),
        ('rej_time', -999, 'int32'),
        ('eogArtifact', -1, 'int8')
)

vffr_fields = (
        ('trial', -999, 'int16'),
        ('serialpos', -999, 'int16'),
        ('item_name', '', 'U16'),
        ('item_num', -999, 'int16'),
        ('recalled', False, 'b1'),
        ('rectime', -999, 'int32'),
        ('intrusion', False, 'b1'),
        ('pres_dur', -999, 'int32'),
        ('too_fast', False, 'b1'),
        ('too_fast_msg', False, 'b1'),
        ('correct', -1, 'int8'),
        ('eogArtifact', -1, 'int8')
)

prelim_fields = (
        ('trial', -999, 'int16'),
        ('serialpos', -999, 'int16'),
        ('item_name', '', 'U16'),
        ('item_num', -999, 'int16'),
        ('recalled', False, 'b1'),
        ('rectime', -999, 'int32'),
        ('intrusion', False, 'b1'),
        ('intruded', False, 'b1'),
        ('pres_dur', -999, 'int32'),
        ('eogArtifact', -1, 'int8')
)


# PS2-3
ps_fields = (
            ('exp_version', '', 'U16'),
            ('ad_observed', 0, 'b1'),
            ('is_stim', 0, 'b1')
        )


system2_ps_fields = (
        ('hosttime', -1, 'int64'),
        ('file_index', -1, 'int16')
    )


# PS4

location_subfields = (
        ('loc_name','','U16'),
        ('amplitude',-999,'float64'),
        ('delta_classifier',-999,'float64'),
        ('sem',-999,'float64'),
        ('snr',-999,'float64')
    )

sham_subfields = (
        ('delta_classifier',-999,'float64'),
        ('sem',-999,'float64'),
        ('p_val',-999,'float64',),
        ('t_stat',-999,'float64',),
    )

decision_subfields = (
        ('p_val',-999.0, 'float64'),
        ('t_stat',-999.0,'float64'),
        ('best_location','','U16'),
        ('tie',-1,'int16'),

    )


# TH
th_fields = (
            ('trial', -999, 'int16'),
            ('item_name', '', 'U64'),
            ('chestNum', -999, 'int16'),
            ('block', -999, 'int16'),
            ('locationX', -999, 'float64'),
            ('locationY', -999, 'float64'),
            ('chosenLocationX', -999, 'float64'),
            ('chosenLocationY', -999, 'float64'),
            ('navStartLocationX', -999, 'float64'),
            ('navStartLocationY', -999, 'float64'),
            ('recStartLocationX', -999, 'float64'),
            ('recStartLocationY', -999, 'float64'),
            ('isRecFromNearSide', False, 'b1'),
            ('isRecFromStartSide', False, 'b1'),
            ('reactionTime', -999, 'float64'),
            ('confidence', -999, 'int16'),
            ('radius_size', -999, 'float64'),
            ('listLength', -999, 'int16'),
            ('distErr', -999, 'float64'),
            ('normErr', -999, 'float64'),
            ('recalled', False, 'b1'),
            ('exp_version', '', 'U64'),
            ('stim_list', False, 'b1'),
            ('is_stim', False, 'b1'),
        )


thr_fields =(
    ('trial', -999, 'int16'),
    ('item_name', '', 'U64'),
    ('resp_word', '', 'U64'),
    ('serialpos', -999, 'int16'),
    ('probepos', -999, 'int16'),
    ('block', -999, 'int16'),
    ('locationX', -999, 'float64'),
    ('locationY', -999, 'float64'),
    ('navStartLocationX', -999, 'float64'),
    ('navStartLocationY', -999, 'float64'),
    ('recStartLocationX', -999, 'float64'),
    ('recStartLocationY', -999, 'float64'),
    ('reactionTime', -999, 'float64'),
    ('list_length', -999, 'int16'),
    ('recalled', False, 'b1'),
    ('exp_version', '', 'U64'),
    ('stim_list', False, 'b1'),
    ('is_stim', False, 'b1'),
)

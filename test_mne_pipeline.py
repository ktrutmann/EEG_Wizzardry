import mne
import mne_pipeline
import os
import pandas as pd


def test_pipeline_kev_data():
    trigger_dict = {'Masked': 2, 'Reveal': 4, 'Left_choice': 8, 'Right_choice': 16, 'No_choice': 32}

    # Reading in the data:
    eeg_prep = mne_pipeline.EEGPrep(os.path.join('Data', 'raw', 'kevin_raw.fif'), trigger_dict,
                                    participant_identifier='test_case')
    assert isinstance(eeg_prep.raw, mne.io.fiff.raw.Raw)

    eeg_prep.fix_channels(n_ext_channels=9,
                          ext_ch_mapping=None)

    eeg_prep.set_references(bipolar_dict=dict(eye_horizontal=['PO10', 'FT10'],
                                              eye_vertical=['HeRe', 'FT10'],
                                              right_hand=['HeLi', 'VeDo'],
                                              left_hand=['VeUp', 'EMG1a']))

    eeg_prep.find_events(stim_channel='Status', uint_cast=True, consecutive=True, min_duration=.01)
    assert eeg_prep.events is not None

    eeg_prep.find_ica_components()
    eeg_prep.remove_ica_components(reject_from_file=False, reject_list_file_location='test_outputs')
    assert eeg_prep.ica is not None
    assert len(eeg_prep.ica.exclude) > 0

    events_to_be_used = ['Left_choice', 'Right_choice']

    eeg_prep.get_epochs(event_labels=events_to_be_used)
    data_frame = eeg_prep.get_epochs_df(event_labels=events_to_be_used)
    assert isinstance(data_frame, pd.DataFrame)

    # Assert that they have the same number of chanels and number of events
    assert eeg_prep.epochs.info['nchan'] + 1 == data_frame.shape[1]
    assert eeg_prep.epochs.events.shape[0] == len(data_frame.index.levels[2])

    eeg_prep.filters(low_freq=1/7, high_freq=128, notch_freq=50)

    # TODO: Test the other methods as soon as they are implemented

    # Saving data:
    eeg_prep.save_prepared_data(save_path=os.path.join('Data', 'prepared'),
                                overwrite=True)


def test_pipeline_laura_dat():
    events_id = dict(show_options=2,
                     start_choice=4,
                     end_choice=8,
                     feedback=16,
                     feedback_null=32,
                     end_trial=64)
    
    data_preprocessed = mne_pipeline.EEGPrep(
        eeg_path='Data/raw/laura_raw.fif', trigger_dict=events_id, participant_identifier=1)
    assert isinstance(data_preprocessed.raw, mne.io.fiff.raw.Raw)
    
    data_preprocessed.fix_channels(
        ext_ch_mapping={'FT10': 'eog',
                        'PO10': 'eog',
                        'HeRe': 'eog',
                        'HeLi': 'emg',
                        'VeUp': 'emg',
                        'VeDo': 'emg',
                        'EMG1a': 'emg',
                        'Status': 'resp'},
        n_ext_channels=9)
    
    data_preprocessed.set_references(ref_ch=['FT9', 'PO9'],
                                     bipolar_dict=dict(
                                         eye_horizontal=['PO10', 'FT10'],
                                         eye_vertical=['HeRe', 'FT10'],
                                         right_hand=['HeLi', 'VeDo'],
                                         left_hand=['VeUp', 'EMG1a']))
    
    data_preprocessed.filters(low_freq=.1, high_freq=100, notch_freq=50)
    
    data_preprocessed.find_events(stim_channel='Status')
    assert data_preprocessed.events is not None
    
    epochs_df = data_preprocessed.get_epochs_df(event_labels=['start_choice', 'feedback'])
    print(epochs_df.head())
    
    # data_preprocessed.automatic_bad_channel_marking(interpolate=True,
    #                                                 threshold_sd_of_mean=40,
    #                                                 event_labels=['show_options', 'start_choice', 'feedback'],
    #                                                 tmin=-.5,
    #                                                 tmax=3)
    data_preprocessed.find_ica_components()
    
    
def test_pipeline_peter_dat():
    pass  # TODO (peter): Implement tests

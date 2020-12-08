import mne
import mne_pipeline
import os
import pandas as pd


def test_pipeline_kev_dat():
    trigger_dict = {'Masked': 2, 'Reveal': 4, 'Left_choice': 8, 'Right_choice': 16, 'No_choice': 32}

    # Reading in the data:
    eeg_prep = mne_pipeline.EEGPrep(os.path.join('Data', 'raw', 'kevin_raw.fif'), trigger_dict,
                                    participant_identifier='test_case')
    assert isinstance(eeg_prep.raw, mne.io.fiff.raw.Raw)

    eeg_prep.fix_channels(n_ext_channels=9, ext_ch_mapping=None)

    eeg_prep.set_references(bipolar_dict=dict(eye_horizontal=['PO10', 'FT10'],
                                              eye_vertical=['HeRe', 'FT10'],
                                              right_hand=['HeLi', 'VeDo'],
                                              left_hand=['VeUp', 'EMG1a']))

    eeg_prep.set_montage()

    eeg_prep.find_events(stim_channel='Status', consecutive=True, min_duration=.01)
    assert eeg_prep.events is not None

    eeg_prep.find_ica_components()
    eeg_prep.remove_ica_components(reject_from_file=False, reject_list_file_location='test_outputs')
    assert eeg_prep.ica is not None
    assert len(eeg_prep.ica.exclude) > 0

    eeg_prep.filters(low_freq=1/7, high_freq=128, notch_freq=50)

    events_to_be_used = ['Left_choice', 'Right_choice']
    eeg_prep.get_epochs(event_labels=events_to_be_used, tmin=-1.5, tmax=.2, baseline=(-1.5, -1.2))
    data_frame = eeg_prep.get_epochs_df(event_labels=events_to_be_used, tmin=-1.5, tmax=.2, baseline=(-1.5, -1.2))
    assert isinstance(data_frame, pd.DataFrame)

    # Assert that they have the same number of channels and number of events
    assert eeg_prep.epochs.info['nchan'] + 1 == data_frame.shape[1]
    assert eeg_prep.epochs.events.shape[0] == len(data_frame.index.levels[2])

    eeg_prep.deal_with_bad_channels(selection_method='automatic', threshold_sd_of_mean=40,
                                    interpolate=False, file_path=os.path.join('Data', 'bads'))

    # Testing whether getting bad channels from the file also works:
    eeg_prep.deal_with_bad_channels(selection_method='file', interpolate=True,
                                    file_path=os.path.join('Data', 'bads'))

    eeg_prep.deal_with_bad_epochs(selection_method='automatic', drop_epochs=False,
                                  file_path=os.path.join('Data', 'bads'))

    # Testing whether getting bad epochs from the file also works:
    eeg_prep.deal_with_bad_epochs(selection_method='file', drop_epochs=True,
                                  file_path=os.path.join('Data', 'bads'))

    # Saving data:
    eeg_prep.save_prepared_data(save_path=os.path.join('Data', 'prepared'), save_epochs=True, save_events=True,
                                overwrite=True)


def test_pipeline_laura_dat():
    events_id = dict(show_options=2,
                     start_choice=4,
                     end_choice=8,
                     feedback=16,
                     feedback_null=32,
                     end_trial=64)
    
    data_preprocessed = mne_pipeline.EEGPrep(eeg_path = 'Data/raw/laura_raw.fif', 
                                             trigger_dict = events_id, 
                                             participant_identifier=1)
    assert isinstance(data_preprocessed.raw, mne.io.fiff.raw.Raw)
    
    data_preprocessed.fix_channels(
        ext_ch_mapping = {'FT10': 'eog', 
                          'PO10': 'eog', 
                          'HeRe': 'eog', 
                          'HeLi': 'emg', 
                          'VeUp': 'emg', 
                          'VeDo': 'emg',
                          'EMG1a': 'emg', 
                          'Status': 'resp'},
        n_ext_channels = 9)
    
    data_preprocessed.set_references(ref_ch = ['FT9', 'PO9'], 
                                     bipolar_dict = dict(
                                     eye_horizontal=['PO10', 'FT10'],
                                     eye_vertical=['HeRe', 'FT10'],
                                     right_hand=['HeLi', 'VeDo'],
                                     left_hand=['VeUp', 'EMG1a']))
    
    eeg_prep.set_montage()
    
    data_preprocessed.filters(low_freq=.1, high_freq=100, notch_freq=50)
    
    data_preprocessed.find_events(stim_channel='Status')
    assert data_preprocessed.events is not None
    
    epochs = mne.Epochs(data_preprocessed.raw,
                        events=data_preprocessed.events)
    print("Size of epochs object with previously found events: ", epochs.get_data().shape)
    
    epochs = data_preprocessed.get_epochs_df(event_labels=['start_choice', 'feedback'])
    print("Index epochs df object (only 2 type of events): ", epochs.index)
    
    print("Bad channels before doing anything: ", data_preprocessed.raw.info['bads'])
    
    data_preprocessed.deal_with_bad_channels(selection_method='automatic',
                                             threshold_sd_of_mean=40,
                                             event_labels=['show_options', 'start_choice', 'feedback'],
                                             file_path=os.path.join('Data', 'bads'),
                                             interpolate=False,
                                             plot=True,
                                             tmin=-.5,
                                             tmax=3)
    print("Bad channels after automatic selection, no interpolation: ", data_preprocessed.raw.info['bads'])
    
    data_preprocessed.deal_with_bad_channels(selection_method='file',
                                             interpolate=True,
                                             file_path=os.path.join('Data', 'bads'))
    
    print("Bad channels after getting from file & interpolation: ", data_preprocessed.raw.info['bads'])
    
    data_preprocessed.find_ica_components()
    data_preprocessed.remove_ica_components(reject_from_file=False, reject_list_file_location='test_outputs')
    
    
def test_pipeline_peter_dat():
    # TODO (Peter): Write a test for each method!
    trigger_dict = {'Stimulus': 36, 'Left_choice': 62, 'Right_choice': 64}
    # Reading in the data:
    eeg_prep = mne_pipeline.EEGPrep(os.path.join('Data', 'raw', 'peter_raw.fif'), trigger_dict)
    assert isinstance(eeg_prep.raw, mne.io.fiff.raw.Raw)
    eeg_prep.fix_channels(n_ext_channels=10,
                          ext_ch_mapping={'EMG1a': 'emg',
                                          'EMG1b': 'emg',
                                          'FT10': 'emg',
                                          'PO10': 'emg',
                                          'HeRe': 'eog',
                                          'HeLi': 'eog',
                                          'VeUp': 'eog',
                                          'VeDo': 'eog',
                                          'STI 014': 'resp'},
                          montage_path='/usr/local/lib/python3.7/site-packages/mne/channels/data/montages')

    eeg_prep.set_references(ref_ch=['PO9', 'FT9'], bipolar_dict=dict(
                                              eye_horizontal=['HeRe', 'HeLi'],
                                              eye_vertical=['VeUp', 'VeDo'],
                                              right_hand=['PO10', 'FT10'],
                                              left_hand=['EMG1b', 'EMG1a']))

    eeg_prep.find_events(stim_channel='STI 014', consecutive=True, min_duration=.01)
    assert eeg_prep.events is not None

    eeg_prep.get_epochs(event_labels=['Stimulus'], tmin=-.5, tmax=2, baseline=(-.5, 0))
    eeg_prep.deal_with_bad_epochs(selection_method='file', drop_epochs=True,
                                  file_name='/Users/peterkraemer/PycharmProjects/EEG_Wizzardry/Data/prepared/test1')

    # Saving data:
    eeg_prep.save_prepared_data(save_path=os.path.join('Data', 'prepared'),
                                overwrite=True)

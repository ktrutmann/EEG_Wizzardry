import mne
import mne_pipeline
import os


def test_pipeline_kev_data():
    trigger_dict = {'Masked': 2, 'Reveal': 4, 'Left_choice': 8, 'Right_choice': 16, 'No_choice': 32}

    # Reading in the data:
    eeg_prep = mne_pipeline.EEGPrep(os.path.join('Data', 'raw', 'kevin_raw.fif'), trigger_dict)
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

    # TODO (Kevin): Write tests for everything else

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
    
    data_preprocessed = mne_pipeline.EEGPrep(eeg_path = 'Data/raw/laura_raw.fif', trigger_dict = events_id)
    
    data_preprocessed.fix_channels(
        montage_path = '/Users/laurafontanesi/miniconda3/pkgs/mne-0.19.2-py_2/site-packages/mne/channels/data/montages/',
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
    
    data_preprocessed.filters(low_freq=.1, high_freq=100, notch_freq=50)
    
    data_preprocessed.find_events(stim_channel='Status')
    
    
    
    
def test_pipeline_peter_dat():
    pass  # TODO (peter): Implement tests
import mne
import mne_pipeline
import os


def test_pipeline_kev_data():
    trigger_dict = {'Masked': 2, 'Reveal': 4, 'Left_choice': 8, 'Right_choice': 16, 'No_choice': 32}

    # Reading in the data:
    eeg_prep = mne_pipeline.EEGPrep(os.path.join('Data', 'raw', 'kevin_raw.fif'), trigger_dict)
    assert isinstance(eeg_prep.raw, mne.io.fiff.raw.Raw)

    eeg_prep.fix_channels(montage_path=os.path.join('C:\\', 'Users', 'Kevin', 'Anaconda3', 'Lib', 'site-packages',
                                                    'mne', 'channels', 'data', 'montages'),
                          n_ext_channels=9,
                          ext_ch_mapping=None)

    eeg_prep.set_references(bipolar_dict=dict(eye_horizontal=['PO10', 'FT10'],
                                              eye_vertical=['HeRe', 'FT10'],
                                              right_hand=['HeLi', 'VeDo'],
                                              left_hand=['VeUp', 'EMG1a']))

    eeg_prep.find_events(stim_channel='Status', uint_cast=True, consecutive=True, min_duration=.01)
    assert eeg_prep.events is not None

    eeg_prep.remove_artifacts_by_ica(auto_select_eye_artifacts=True, reject_list_save_location='test_outputs')
    assert eeg_prep.ica is not None

    # TODO (Kevin): Write tests for everything else

    # Saving data:
    eeg_prep.save_prepared_data(save_path=os.path.join('Data', 'prepared'),
                                file_name='kevin_test_data',
                                overwrite=True)

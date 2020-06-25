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

def test_pipeline_laura_data():
    pass  # TODO (laura): Implement tests

def test_pipeline_peter_data():
    pass  # TODO (peter): Implement tests
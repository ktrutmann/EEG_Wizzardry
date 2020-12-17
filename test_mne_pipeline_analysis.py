import mne
import os
import pandas as pd
import mne_pipeline_prep
import mne_pipeline_analysis
import mne_pipeline


def test_pipeline_kev_dat():
    pass

def test_pipeline_laura_dat():
    pass
    
def test_pipeline_peter_dat():
    # load preprocessed data
    data = mne_pipeline.EEGAnalysis(prepared_dat_preprocessed_name='Data/prepared/participant_xxx_prepared_raw.fif')
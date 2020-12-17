import os
import warnings
import pickle
import mne
import pandas as pd
import numpy as np


class EEGAnalysis(object):
    def __init__(self, prepared_dat_preprocessed_name=None, prepared_dat_epochs_name=None):
        """
        Parameters
        ----------
        prepared_dat_preprocessed_name: str, optional
            The path and name of the preprocessed data as one long time series in .fif or .bdf format.

        prepared_dat_epochs_name: str, optional
            The path and name of the preprocessed data as epochs in .fif or .bdf format.
        """

        self.preprocessed = None
        self.epochs = None
        if prepared_dat_preprocessed_name is not None:
            if prepared_dat_preprocessed_name.endswith('.bdf'):
                self.raw = mne.io.read_raw_bdf(prepared_dat_preprocessed_name, preload=True)
            elif prepared_dat_preprocessed_name.endswith('.fif'):
                self.raw = mne.io.read_raw_fif(prepared_dat_preprocessed_name, preload=True)
            else:
                raise ValueError('Please provide a .bdf or .fif file for the prepared raw data.')

        if prepared_dat_epochs_name is not None:
            if prepared_dat_epochs_name.endswith('.fif'):
                self.epochs = mne.read_epochs(prepared_dat_epochs_name, preload=True)
            else:
                raise ValueError('Please provide a .bdf or .fif file for the prepared epoched data.')

    # TODO (All): Create methods to get the LRP plot
    def ComputeLRP(self,electrodes=['C3','C4']):
        """
        Parameters
        ----------

        """

        # 1. check if data is epoched.

        # 2. compute average signal for trials with left/right button press

        # 3. subtract the data like in the code snippet below

        #def CalculateLRP(dataLeft, dataRight, channelsLeft, channelsRight):
        #    output = (dataRight[:, channelsLeft, :, :] - dataLeft[:, channelsLeft, :, :]) - (
        #                dataRight[:, channelsRight, :, :] - dataLeft[:, channelsRight, :, :])
        #    return output

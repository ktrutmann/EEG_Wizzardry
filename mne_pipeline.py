import os
import warnings
import pickle
import mne
import numpy as np


class EEGPrep(object):
    """
    Allows to load EEG data from file, perform different preprocessing steps
    and save the results for further analysis.
    """

    def __init__(self, eeg_path, trigger_dict, participant_identifier=''):
        """
        Initiates the EEGPrep object, given a file path and a trigger dictionary,
        containing the number coding of the trigger to events.

        Parameters
        ----------
        eeg_path : str, path object
            A valid string path pointing to a stored .bdf or .fif file.

        trigger_dict : dict of {str : int}
            A dictionary in which keys are labels for the triggered events,
            and their respective values are the trigger
            code number as recorded in the data.

        participant_identifier: str, int
            Some identifier of the data source. Is used for naming the safed files but can also be left empty.

        Attributes
        ----------
        raw : instance of RawEDF
            mne class for raw data.
            See: https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw
        # TODO (Meeting): Call it "participant" or "subject" or something else?
        """
        self.eeg_path = eeg_path
        self.trigger_dict = trigger_dict
        self.participant_id = participant_identifier
        self.events = None
        self.epochs = None
        self.epochs_pd_df = None
        self.ica = None

        # import data
        if eeg_path.endswith('.bdf'):
            self.raw = mne.io.read_raw_bdf(self.eeg_path, preload=True)
        elif eeg_path.endswith('.fif'):
            self.raw = mne.io.read_raw_fif(self.eeg_path, preload=True)

    def fix_channels(self, montage_path, n_ext_channels, ext_ch_mapping=None):
        """
        Removes the '1-' from the start of the channel names, sets the Montage (telling MNE which electrode went where)
        and sets the type of the additional electrodes. For the fixing of the channel names it is assumed that the
        last channel is the 'Status' channel.

        Parameters
        ----------
        montage_path : str
            A valid string path pointing to the montage folder of the MNE module.

        n_ext_channels : int
            The number of extra channels used above the 64 standard ones. This includes the reference channels
            but not the Stim (trigger) channel.

        ext_ch_mapping : dict of {str : str}
            A dictionary containing the extra channels (including the 'Status' or 'Stim' channel!) as key and their
            type ('eog', 'emg', 'resp') as values. A standard mapping using seven extra electrodes is provided if this
            argument is omitted, but it is recommended to explicitly define one. If the standard mapping is used,
            `n_ext_channels` is overwritten to 9. The default dictionary is as follows:
            `{'FT10': 'eog', 'PO10': 'eog', 'HeRe': 'eog', 'HeLi': 'emg', 'VeUp': 'emg', 'VeDo': 'emg',
             'EMG1a': 'emg', 'Status': 'resp'}`
        """

        if ext_ch_mapping is None:
            ext_ch_mapping = {'FT10': 'eog', 'PO10': 'eog', 'HeRe': 'eog', 'HeLi': 'emg', 'VeUp': 'emg', 'VeDo': 'emg',
                              'EMG1a': 'emg', 'Status': 'resp'}
            n_ext_channels = 9
            warnings.warn('You are using the default mapping for the extra channels!\n' +
                          'Please make sure it is the correct one for your use case.', UserWarning)

        # Removing superfluous '1-' in ch_names
        for name in self.raw.ch_names[:-1]:
            self.raw.rename_channels({name: name[2:]})

        # Drop unused channels
        self.raw.drop_channels(self.raw.ch_names[64 + n_ext_channels:len(self.raw.ch_names) - 1])

        # Set montage
        montage = mne.channels.read_montage(kind='biosemi64', path=montage_path)
        self.raw.set_montage(montage)

        # Set channel types
        self.raw.set_channel_types(mapping=ext_ch_mapping)

        print('Fixed channel names, dropped unused channels, changed channel types and set montage.')

    def set_references(self, ref_ch=('PO9', 'FT9'), bipolar_dict=None):
        """
        This method re-references the prepared raw eeg data to the average signal at the Mastoid bones.
        Further, it sets bipolar references for eog and emg data and creates a new channel for the referenced signal.

        Parameters
        ----------
        bipolar_dict: dict, optional
            A dictionary containing the name of the new channel as key and the to-be-referenced channels as values.
            dict(EMG_right=['PO10','FT10'],
                EMG_left=['EMG1b','EMG1a'],
                EOG_x=['HeRe', 'HeLi'],
                EOG_y=['VeUp', 'VeDo'])

        ref_ch: list, optional
            A list, containing the reference channels.
        """

        if isinstance(ref_ch, tuple):
            ref_ch = list(ref_ch)

        if bipolar_dict is None:
            mne.set_bipolar_reference(self.raw,
                                      anode=[val[0] for val in bipolar_dict.values()],
                                      cathode=[val[1] for val in bipolar_dict.values()],
                                      ch_name=list(bipolar_dict.keys()),
                                      copy=False)
        # TODO: Check if EOG channels keep their channel type!

        self.raw.set_eeg_reference(ref_channels=ref_ch)
        self.raw.drop_channels(ref_ch)

    def find_events(self, **kwargs):
        """
        Finds events in raw file.

        Parameters
        ----------
        kwargs : keyword arguments
            All keyword arguments are passed to mne.find_events().
            See: https://mne.tools/stable/generated/mne.find_events.html.

        Returns
        -------
        events : array, shape = (n_events, 3)
            All events that were found. The first column contains the event time in samples
            and the third column contains the event id.
            For output = ‘onset’ or ‘step’, the second column contains the value
            of the stim channel immediately before the event/step.
            For output = ‘offset’, the second column contains the value
            of the stim channel after the event offset.

        """

        self.events = mne.find_events(raw=self.raw, **kwargs)

        return self.events

    def remove_artifacts_by_ica(self, fit_on_epochs=False, auto_select_eye_artifacts=False, high_pass_freq=1,
                                decim=3, reject_list_save_location='', **kwargs):
        """
        Uses independent component analysis to remove movement components (primarily eye artifacts).
        The fitting procedure can either use the complete raw data or the epoched data to exclude the breaks
        between trials. The exclusions are then however applied to both the raw data and (if it exists) the epoched
        data.

        Parameters
        ----------
        fit_on_epochs: bool
            Indicates whether the ICA will be fit on the epoched data, thus leaving out the gaps between trials.
            Default behavior is to use all of the raw data.
        auto_select_eye_artifacts: bool
            If set to true the method will use mne's `ica.find_bads_eog` to automatically select which components
            to remove from the data. Defaults to False (manual selection of which components to remove).
        high_pass_freq: float, int
            Before fitting the ICA a high pass filter is applied since the procedure is very sensitive to low
            frequencies. This value indicates the cutoff frequency. Defaults to 1 Hz.
        decim: int
            Determines by how much the data is decimated when fitting the ICA. This maked the process faster as
            only every n-th sample is used. Defaults to 3.
        reject_list_save_location: str, None
            Where to save the list of rejected ICA components. Default is the current directory.
            Set to None if this list should not be saved.
        kwargs:
            Will be passed on to the mne.preprocessing.ICA (for creating the ICA object).
            For available arguments see https://mne.tools/stable/generated/mne.preprocessing.ICA.html

        Notes
        -----
        The mna ICA method excludes segments that have previously been annotated as "bad" from the fitting procedure,
        so make sure you have excluded segments by hand where the data is completely unusable (e.g. during a sneeze
        or yawning).

        TODO (Meeting): Discuss whether and how to use the "reject" argument of ICA.fit()
        TODO: Maybe make it possible to use a "reject component list" as input (for reproducability).
        TODO (Kevin): Test ica
        """

        if fit_on_epochs and self.epochs is None:
            raise AttributeError('No epochs found. You have to create epochs using `get_epochs()` before '
                                 'you can fit the ICA on them.')

        fit_data = self.epochs if fit_on_epochs else self.raw

        self.ica = mne.preprocessing.ICA(**kwargs)
        self.ica.fit(fit_data.filter(high_pass_freq, None), decim=decim)

        # Rejecting bad components:
        if auto_select_eye_artifacts:
            eog_indices, eog_scores = self.ica.find_bads_eog(fit_data)
            self.ica.exclude = eog_indices
            print('Automatically excluding the following components: {}'.format(eog_indices))
            # TODO (Kevin): Maybe still plot the components?
        else:
            self.ica.exclude = []  # TODO (Kevin): Implement manual component removal
            # TODO (Meeting): Discuss how to enter which components to exclude.

        if reject_list_save_location is not None:
            np.savetxt(os.path.join(reject_list_save_location,
                                    'participant_{}_rejected_ICA_components.csv'.format(self.participant_id)),
                       self.ica.exclude, fmt='%i')

        # Applying the ica
        if len(self.ica.exclude) > 0:
            self.ica.apply(self.raw, exclude=self.ica.exclude)
            if self.epochs is not None:
                self.ica.apply(self.epochs, exclude=self.ica.exclude)


    def get_epochs(self, epoch_event_dict, **kwargs):
        # TODO (Laura): Add event list support here as well
        """
        This method creates the epochs from the events found previously by calling the `make_events` method.

        Parameters
        ----------
        epoch_event_dict : int, list, dict
            It is recommended to provide a dictionary with the events label as key and the event id as value.
            mne can however also handle int or lists.

        kwargs:
            Will be passed on to the `mne.Epochs` method.
        """
        if self.events is None:
            raise AttributeError('No events found. Please find them by running the find_events() method first.')

        self.epochs = mne.Epochs(self.raw, events=self.events, event_id=epoch_event_dict, **kwargs)
        print('Created epochs using the provided event ids and / or labels.')

    def get_epochs_df(self, event_labels, events_kws=None, **kwargs):
        """
        Gets epoch data as pandas DataFrames, given a list of event labels
        (for now, just an idea).

        Parameters
        ----------
        events_kws : dict, optional
            Additional arguments to find_events.

        kwargs : keyword arguments
            All keyword arguments are passed to mne.Epochs().
            See: https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.

        Returns
        -------
        df : DataFrame
            A dataframe suitable for usage with other statistical/plotting/analysis packages.
        """

        epochs = mne.Epochs(self.raw,
                            events=self.find_events(**events_kws),
                            event_id=[self.trigger_dict[event] for event in event_labels],
                            **kwargs)

        df = epochs.to_data_frame()
        df.reset_index(inplace=True)

        return df

    def filters(self, low_freq=1/7, high_freq=128, notch_freq=50):
        """
        This method applies a bandpass filter and a notch filter to the data

        Parameters
        ----------
        low_freq: float, optional
            frequency for high pass filter
        high_freq: float, optional
            frequency for low pass filter
        notch_freq: float, optional
            frequency for notch filter
        """
        self.raw.filter(l_freq= low_freq, h_freq= high_freq)
        self.raw.notch_filter(range(notch_freq, high_freq, notch_freq), filter_length='auto',
                              phase='zero', fir_design='firwin')

    def save_prepared_data(self, save_path='', file_name='EEG_data', save_events=False, save_epochs=False, **kwargs):
        # TODO (Kevin): Test
        # TODO (Kevin): Include participant id in names
        """
        This method saves the prepared raw data and (optionally) the epochs.
        It can also save the events as a pickle file so it can easily be reused later.

        Parameters
        ----------
        save_path: string, optional
            A path to the folder where the data should be saved in. If not provided, the file will be saved in the
            directory the script is run from.

        file_name: string, optional
            A string providing the name of the file. '_prep_raw.fif', '_epochs.fif' ect. will automatically be added.

        save_events: boolean, optional
            Indicates whether the found events should be saved as pickle files as well.

        save_epochs: boolean, optional
            Indicates whether the created epochs should be saved as well.

        kwargs: dict, optional
            Will be passed on to the raw.save() method.
        """

        file_path_and_name = os.path.join(save_path, file_name.__add__('_prepared_raw.fif'))
        self.raw.save(file_path_and_name, **kwargs)
        print('Saved the prepared raw file to {}.'.format(file_path_and_name))

        if save_events:
            if self.events is None:
                raise AttributeError('No events to save. Please find them by running the find_events() method first.')

            file_path_and_name = os.path.join(save_path, file_name.__add__('_events.pickle'))
            pickle_file = open(file_path_and_name, 'wb')
            pickle.dump(self.events, pickle_file)
            pickle_file.close()
            print('Saved the events to {}.'.format(file_path_and_name))

        if save_epochs:
            if self.epochs is None:
                raise AttributeError('You have not created any epochs yet.\n'
                                     'Create them by running the make_epochs() method first.')

            file_path_and_name = os.path.join(save_path, file_name.__add__('epochs.fif'))
            self.epochs.save(file_path_and_name)

# TODO: Maybe have a function to plot raw data to files.
# TODO: Method to interpolate bad channels
# TODO: Method to exclude bad epochs

# TODO (Kevin): Start implementing ICA
#   - Give option to let ica.find_bads_eog() do the work and/or manually exclude components
#   - Find a way to only fit ICA on "trial data" (exclude pauses; Make this an option?)

# TODO (Laura): Implement bad_channels methods:
#   - detect_bad_channels() with your algorithm
#   - mark_bad_channels() where you plot the data and can click on the bad channels
#   - interpolate_bad_channels() just a wrapper for the mne method.

import os
import warnings
import pickle
import mne


class EEGPrep(object):
    """
    Allows to load EEG data from file, to perform different preprocessing steps
    and pickle the results or export them as pandas DataFrames.
    """

    def __init__(self, eeg_path, trigger_dict):
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

        Attributes
        ----------
        raw : instance of RawEDF
            mne class for raw data.
            See: https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw

        """
        self.eeg_path = eeg_path
        self.trigger_dict = trigger_dict
        self.events = None
        self.epochs = None
        self.epochs_pd_df = None


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
            warnings.warn('You are using the default mapping for the extra channels!\nPlease make sure it is '
                          'the correct one for your use case.', UserWarning)

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

    def set_references(self, ref_ch=('PO9', 'FT9')):
        """
        EXPLAIN TO MEEE!!!
        """

        self.raw.set_eeg_reference(ref_channels=ref_ch)
        self.raw.drop_channels(ref_ch)

        # TODO (Peter): Set bipolar references for EMG & EOG
        # TODO (Peter): documentation

        pass

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

    def make_epochs(self, epoch_event_dict, **kwargs):
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
        # TODO (Laura): Is it okay to split this?
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

    def filters(self):
        """
        EXPLAIN TO MEEE!!!
        """
        pass

    def save_prepared_data(self, save_path='', file_name='EEG_data', save_events=False, save_epochs=False, **kwargs):
        # TODO (Kevin): Test
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

        file_path_and_name = os.path.join(save_path, file_name.__add__('prepared_raw.fif'))
        self.raw.save(file_path_and_name, **kwargs)
        print('Saved the prepared raw file to {}.'.format(file_path_and_name))

        if save_events:
            if self.events is None:
                raise AttributeError('No events to save. Please find them by running the find_events() method first.')

            file_path_and_name = os.path.join(save_path, file_name.__add__('events.pickle'))
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

#TODO: Maybe have a function to plot raw data to files.
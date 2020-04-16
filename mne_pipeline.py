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
            A valid string path pointing to a stored .bdf file.

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

        # import data
        self.raw = mne.io.read_raw_edf(self.eeg_path, preload=True)

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
            `n_ext_channels` is overwritten to 9.
        """

        if ext_ch_mapping is None:
            ext_ch_mapping = {'FT10': 'eog', 'PO10': 'eog', 'HeRe': 'eog', 'HeLi': 'emg', 'VeUp': 'emg', 'VeDo': 'emg',
                            'EMG1a': 'emg', 'Status': 'resp'}  # TODO: discuss
            n_ext_channels = 9

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

    def set_references(self):  # TODO (Peter): Implement
        """
        EXPLAIN TO MEEE!!!
        """
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

        events = mne.find_events(raw=self.raw, **kwargs)

        return events

    def get_epochs_df(self, event_labels, events_kws=None, **kwargs):
        """
        Gets epoch data as pandas DataFrames, given a list of event labels
        (for now, just an idea).

        Parameters
        ----------
        kwargs : keyword arguments
            All keyword arguments are passed to mne.Epochs().
            See: https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.

        events_kws : dict, optional
            Additional arguments to find_events.

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

#TODO: Maybe have a function to plot raw data to files.
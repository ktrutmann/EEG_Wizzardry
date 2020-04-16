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

    def fix_channels(self):  # TODO (Kevin): Implement
        """
        EXPLAIN TO MEEE!!!
        """
        pass

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

        events = mne.find_events(raw = self.raw, **kwargs)

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
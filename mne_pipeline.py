class EEGPrep:
    """
        EXPLAIN TO MEEE!!!
    """

    def __init__(self, eeg_path, trigger_dict):  # TODO (Laura): Implement
        """
        EXPLAIN TO MEEE!!!
        """
        self.eeg_path = eeg_path
        self.trigger_dict = trigger_dict

        # TODO: implement the loading of the data here
        self.raw = 42

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

    def find_events(self):  # TODO (Laura): Implement
        """
        EXPLAIN TO MEEE!!!
        """
        pass

    def filters(self):
        """
        EXPLAIN TO MEEE!!!
        """
        pass

    def extract_events(self):
        """
        EXPLAIN TO MEEE!!!
        """
        pass
        # mne.find_events(raw, stim_channel='Status', uint_cast=True, consecutive=True, min_duration=.01)

    # TODO: Maybe have a function to plot raw data to files.

import os
import warnings
import pickle
import mne
import pandas as pd
import numpy as np


class EEGPrep(object):
    """
    Allows to load EEG data from file, perform different preprocessing steps
    and save the results for further analysis.
    """

    def __init__(self, eeg_path, trigger_dict, participant_identifier='xxx'):
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
        """
        self.eeg_path = eeg_path
        self.trigger_dict = trigger_dict
        self.participant_id = participant_identifier
        self.events = None
        self.epochs = None
        self.epochs_pd_df = None
        self.ica = None
        self.ica_fit_params = None

        # import data
        if eeg_path.endswith('.bdf'):
            self.raw = mne.io.read_raw_bdf(self.eeg_path, preload=True)
        elif eeg_path.endswith('.fif'):
            self.raw = mne.io.read_raw_fif(self.eeg_path, preload=True)
        else:
            raise ValueError('Please provide a .bdf or .fif file.')

    def fix_channels(self, n_ext_channels=None, ext_ch_mapping=None):
        """
        Removes the '1-' from the start of the channel names, sets the Montage (telling MNE which electrode went where)
        and sets the type of the additional electrodes. For the fixing of the channel names it is assumed that the
        last channel is the 'Status' channel.

        Parameters
        ----------
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
        if n_ext_channels is None:
            n_ext_channels = 9
            warnings.warn('You are using the default mapping for the extra channels!\n' +
                          'Please make sure it is the correct one for your use case.', UserWarning)
            # TODO: Check whether this warning is displayed

        # Removing superfluous '1-' in ch_names
        for name in self.raw.ch_names[:-1]:
            self.raw.rename_channels({name: name[2:]})

        # Drop unused channels
        self.raw.drop_channels(self.raw.ch_names[64 + n_ext_channels:len(self.raw.ch_names) - 1])

        # Set channel types
        self.raw.set_channel_types(mapping=ext_ch_mapping)

        print('Fixed channel names, dropped unused channels and changed channel types.')

    def set_references(self, ref_ch=None, bipolar_dict=None):
        """
        This method re-references the prepared raw eeg data to the average signal at the mastoid bones.
        Further, it sets bipolar references for eog and emg data and creates a new channel for the referenced signal.

        Parameters
        ----------
        bipolar_dict: dict, optional
            A dictionary containing the name of the new channel as key and the to-be-referenced channels as values.
            dict(EMG_right=['PO10','FT10'],
                EMG_left=['EMG1b','EMG1a'],
                EOG_x=['HeRe', ' HeLi'],
                EOG_y=['VeUp', 'VeDo'])

        ref_ch: list, optional
            A list, containing the reference channels.
        """

        if ref_ch is None:
            ref_ch = ['PO9', 'FT9']

        if bipolar_dict is not None:
            mne.set_bipolar_reference(self.raw,
                                      anode=[val[0] for val in bipolar_dict.values()],
                                      cathode=[val[1] for val in bipolar_dict.values()],
                                      ch_name=list(bipolar_dict.keys()),
                                      copy=False)

        self.raw.set_eeg_reference(ref_channels=ref_ch)
        self.raw.drop_channels(ref_ch)

    def set_montage(self, montage_path=None, montage_kind='biosemi64'):
        """
        Parameters
        ----------
        montage_path: str
            The path to a custom montage, if desired.

        montage_kind: str
            Defaults to 'biosemi64' but can be any inbuilt montage of mne.
        """
        # Set montage
        if montage_path is None:
            try:
                montage = mne.channels.read_montage(kind=montage_kind)
            except:
                montage = mne.channels.make_standard_montage(kind=montage_kind)
            print('Using default montage biosemi64')
        else:
            try:
                montage = mne.channels.read_montage(kind=montage_kind, path=montage_path)
            except:
                montage = mne.channels.read_custom_montage(fname=montage_path)
        # TODO: needs to be tested for mne versions > 0.17

        # Needs to be able to handle different mne versions:
        self.raw.set_montage(montage)

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

    def find_ica_components(self, fit_on_epochs=False, high_pass_freq=1, decim=3, n_components=20, **kwargs):
        """
        Uses independent component analysis to find movement components (primarily eye artifacts).
        The fitting procedure can either use the complete raw data or the epoched data to exclude the breaks
        between trials. If the function is re-run with the same parameters it is not re-fit but the components
        are plotted again.

        Parameters
        ----------
        fit_on_epochs: bool
            Indicates whether the ICA will be fit on the epoched data, thus leaving out the gaps between trials.
            Default behavior is to use all of the raw data.
        high_pass_freq: float, int
            Before fitting the ICA a high pass filter is applied since the procedure is very sensitive to low
            frequencies. This value indicates the cutoff frequency. Defaults to 1 Hz.
        decim: int
            Determines by how much the data is decimated when fitting the ICA. This makes the process faster as
            only every n-th sample is used. Defaults to 3.
        n_components: int
            The number of components that are used while fitting the ICA. Defaults to 20. Set to None if the maximum
            number (number of channels) should be used.
        kwargs:
            Will be passed on to the mne.preprocessing.ICA (for creating the ICA object).
            For available arguments see https://mne.tools/stable/generated/mne.preprocessing.ICA.html

        Notes
        -----
        The mna ICA method excludes segments that have previously been annotated as "bad" from the fitting procedure,
        so make sure you have excluded segments by hand where the data is completely unusable (e.g. during a sneeze
        or yawning).
        """
        if fit_on_epochs and self.epochs is None:
            raise AttributeError('No epochs found. You have to create epochs using `get_epochs()` before '
                                 'you can fit the ICA on them.')

        # Checking whether the ICA needs to be re-run:
        these_params = dict(fit_on_epochs=fit_on_epochs, high_pass_freq=high_pass_freq, decim=decim)
        these_params.update(kwargs)
        if self.ica is None or these_params != self.ica_fit_params:

            fit_data = self.epochs if fit_on_epochs else self.raw

            self.ica = mne.preprocessing.ICA(n_components=n_components, **kwargs)
            self.ica.fit(fit_data.filter(high_pass_freq, None), decim=decim)
            self.ica_fit_params = dict(fit_on_epochs=fit_on_epochs, high_pass_freq=high_pass_freq, decim=decim)
            self.ica_fit_params.update(kwargs)

        print('Plotting found components. If you plan to exclude components manually make sure to create a file '
              'listing the numbers of those you wish to exclude.')
        self.ica.plot_components()

    def remove_ica_components(self, reject_from_file=True, reject_list_file_location='', **kwargs):
        """
        If an ICA object has previously been generated this method provides the possibility of either automatically
        or manually rejecting and removing individual components from the signal.

        Parameters
        ----------
        reject_from_file: bool
            Indicates whether the components to be rejected should be taken from a list in a file whose location is
            provided by the `reject_lift_file_location` argument. If set to `False`, mne's `find_bads_eog()` method
            will be evoked.
        reject_list_file_location: str, None
            Where to save the list of rejected ICA components. Default is the current directory.
            If the directory does not exist it will be created.
            Set to None if this list should not be saved.
        kwargs:
            Will be passed on to ica.apply() for both the raw data and epoched data.

        Notes
        -----
        The mna ICA method excludes segments that have previously been annotated as "bad" from the fitting procedure,
        so make sure you have excluded segments by hand where the data is completely unusable (e.g. during a sneeze
        or yawning).
        """
        if not os.path.exists(reject_list_file_location):
            os.makedirs(reject_list_file_location)

        exclude_list_file = os.path.join(reject_list_file_location,
                                         'participant_{}_rejected_ICA_components.csv'.format(self.participant_id))

        if reject_from_file:
            self.ica.exclude = pd.read_csv(exclude_list_file, header=None).iloc[:, 0].to_list()
        else:
            fit_data = self.epochs if self.ica_fit_params['fit_on_epochs'] else self.raw
            eog_indices, eog_scores = self.ica.find_bads_eog(fit_data)
            self.ica.exclude = eog_indices
            pd.Series(self.ica.exclude).to_csv(exclude_list_file, header=False, index=False)

            if len(self.ica.exclude) == 0:
                print('No bad components detected. Excluding 0 components.')
                return

            print('Plotting components that will be excluded.')
            self.ica.plot_scores(eog_scores)
            print('The excluded component indices have been saved to {}.'.format(exclude_list_file))

        # Applying the ica
        print('Applying ICA to the raw data excluding the following components: {}'.format(
            self.ica.exclude))
        self.ica.apply(self.raw, exclude=self.ica.exclude, **kwargs)

        if self.epochs is not None:
            print('Applying ICA to the epoched data excluding the following components: {}'.format(
                self.ica.exclude))
            self.ica.apply(self.epochs, exclude=self.ica.exclude, **kwargs)

    def get_epochs(self, event_labels, **kwargs):
        # TODO (Discuss): Why not add a `as_df` argument here and merge with `get_epochs_df`?
        # TODO (Peter): Test how mne handles multiple conditions (i.e. left/right button). Does it retain the order?
        """
        This method creates the epochs from the events found previously by calling the `make_events` method.

        Parameters
        ----------
        event_labels: array or list
            Labels corresponding to the events that you want to select.

        kwargs : keyword arguments
            All keyword arguments are passed to mne.Epochs().
            See: https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.

        """
        if self.events is None:
            raise AttributeError('No events found. Please find them by running the find_events() method first.')

        for event in event_labels:
            if event not in self.trigger_dict.keys():
                raise ValueError('{} in event_labels is not one of the your event names. Those are {}'.format(
                    event, self.trigger_dict.keys()))

        self.epochs = mne.Epochs(self.raw, 
                                 events=self.events, 
                                 event_id=[self.trigger_dict[event] for event in event_labels], 
                                 **kwargs)
        print('Created epochs using the provided event ids and / or labels.')

    def get_epochs_df(self, event_labels, **kwargs):
        # TODO (Laura): Suspected bug. Do we get all conditions and meta information back?
        """
        Gets epoch data as pandas DataFrames, given a list of event labels.

        Parameters
        ----------
        event_labels: array or list
            Labels corresponding to the events that you want to select.

        kwargs : keyword arguments
            All keyword arguments are passed to mne.Epochs().
            See: https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.

        Returns
        -------
        df : DataFrame
            A dataframe suitable for usage with other statistical/plotting/analysis packages.

        """
        if self.events is None:
            raise AttributeError('No events found. Please find them by running the find_events() method first.')

        for event in event_labels:
            if event not in self.trigger_dict.keys():
                raise ValueError('{} in event_labels is not one of the your event names. Those are {}'.format(
                    event, self.trigger_dict.keys()))

        epochs = mne.Epochs(self.raw,
                            events=self.events,
                            event_id=dict((k, self.trigger_dict[k]) for k in event_labels),
                            **kwargs)

        df = epochs.to_data_frame()
        df['participant'] = self.participant_id
        df = df.reset_index().set_index(['participant', 'condition', 'epoch', 'time'])

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
            frequency for notch filter.
            The base frequency as well as all its harmonies up to `high_freq` are filtered.
        """
        self.raw.filter(l_freq=low_freq, h_freq=high_freq)
        self.raw.notch_filter(range(notch_freq, high_freq, notch_freq), filter_length='auto',
                              phase='zero', fir_design='firwin')

    def deal_with_bad_channels(self, selection_method, plot=True, threshold_sd_of_mean=40, interpolate=True,
                               file_path=None, **kwargs):
        """
        This method helps identifying and interpolating bad channels.
        The identification can be done automatically, based on the channels' variance;
        and manually, by selecting channels in the interactive plot.
        Both results will be salved to file, for reproducibility.
        With the "file" option, bad channels can be simply recovered from file.

        Parameters
        ----------
        selection_method : string
            Should be either "automatic", "manual", or "file"

        plot : boolean, default True
            Whether an interactive plot should be shown

        threshold_sd_of_mean : real, default 40
            Threshold to flag a channel as bad.
            The SD of mean is calculated as the SD of the mean activation across epochs.
            Channels with very high SD of mean across epochs are likely to be bad.

        interpolate : boolean, default True
            Whether the bad channels should be interpolated.
            After interpolation, they are de-flagged as bad.

        file_path : string, default None
            The file path where the bad channel list will be stored as a .csv file.
            By default, it is saved in the current working directory.
        """

        if file_path is None:
            file_path = os.getcwd()
        file_name = os.path.join(file_path, 'participant_{}_bad_channels.csv'.format(self.participant_id))
        
        if selection_method == "automatic":
            if self.epochs is None:
                raise AttributeError('Please create epochs first, as the automatic algorithm needs them to work.')
            else:
                df = self.epochs.to_data_frame()

            # TODO: (Discuss) Why also group by condition and shouldn't we take the mean of the abs. amplitude?
            group = df.groupby(['condition', 'epoch'])
            mean = group.mean()

            a = mean.std()
            a = a[1:]
            print('standard deviation of mean across epochs:')
            print(np.mean(a), np.std(a))
            print('higher than %s:' % threshold_sd_of_mean)
            print(a[a > threshold_sd_of_mean].index)

            for i in a[a > threshold_sd_of_mean].index:
                self.raw.info['bads'].append(i)

            print("Marked as bad: ", self.raw.info['bads'])

            print("N marked as bad: ", len(self.raw.info['bads']))

            pd.DataFrame({'participant': self.participant_id,
                          'bad_channels': self.raw.info['bads']}).to_csv(path_or_buf=file_name,
                                                                         index=False)

            print("Saving bad channels as {}".format(file_name))

        elif selection_method == "file":
            bads = pd.read_csv(file_name)
            self.raw.info['bads'] = bads['bad_channels'].values

            print("Marked as bad: ", self.raw.info['bads'])

            print("N marked as bad: ", len(self.raw.info['bads']))

        elif selection_method != "manual":
            ValueError("selection_method can be automatic, file, or manual")

        if plot or selection_method == "manual":
            self.raw.plot(block=True)

            print("Marked as bad: ", self.raw.info['bads'])

            print("N marked as bad: ", len(self.raw.info['bads']))

            if file_path is None:
                file_path = os.getcwd()
            file_name = os.path.join(file_path, 'participant_{}_bad_channels.csv'.format(self.participant_id))
            pd.DataFrame({'participant': self.participant_id,
                          'bad_channels': self.raw.info['bads']}).to_csv(path_or_buf=file_name,
                                                                         index=False)

            print("Saving bad channels as {}".format(file_name))

        if interpolate:
            "Interpolating bad channels..."
            if len(self.raw.info['bads']) > 0:
                self.raw.interpolate_bads(reset_bads=True)

    def deal_with_bad_epochs(self, selection_method='automatic', scale_params='default',
                             drop_epochs=False, file_path=None):
        """
        This method identifies epochs that will be rejected for further analysis

        Parameters
        ----------
        selection_method: string, optional
            use automatic or manual epoch detection. When using file, epochs get selected from a specified file.

        scale_params: dict, optional
            parameters passed to the plot function to depict epochs in a visually accessible way
            if selection_method "manual" is chosen.

        drop_epochs: bool, optional
            directly remove epochs from EEG data

        file_path: string, optional
            allows to specify the location of the output file. Also used if selection_method is "file".
            By default the file is stored in the current working directory.
        """
        if scale_params == 'default':
            scale_params = dict(mag=1e-12, grad=4e-11, eeg=150e-6, eog=25e-5, ecg=5e-4,
                                emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4,
                                whitened=10.)
        if file_path is None:
            file_path = os.getcwd()
        file_name = os.path.join(file_path, 'participant_{}_bad_epochs.csv'.format(self.participant_id))

        epochs_copy = self.epochs.copy()
        if selection_method == 'automatic':
            epochs_copy.drop_bad()
        elif selection_method == 'manual':
            epochs_copy.plot(n_channels=68, scalings=scale_params, block=True)
        elif selection_method == 'file':
            epochs_to_be_dropped = pd.read_csv(file_name).epochs.to_list()
            epochs_copy.drop(epochs_to_be_dropped)
        else:
            raise ValueError('Invalid selection method. Permitted methods are automatic, manual and file.')

        epoch_idx = 0
        epochs_to_be_dropped = []
        for i in range(len(self.epochs.drop_log)):
            if not self.epochs.drop_log[i]:  # select epochs of interest
                if self.epochs.drop_log[i] != epochs_copy.drop_log[i]:  # find index of epochs to be dropped
                    epochs_to_be_dropped.append(epoch_idx)
                epoch_idx += 1
        pd.DataFrame(epochs_to_be_dropped, columns=['epochs']).to_csv(file_name)
        print('Saved to-be-dropped epochs to {}.'.format(file_name))

        if drop_epochs:
            epochs_dropped = pd.read_csv(file_name).epochs.to_list()
            self.epochs.drop(epochs_dropped)

        del epochs_copy

    def save_prepared_data(self, save_path='', save_events=False, save_epochs=False, **kwargs):
        """
        This method saves the prepared raw data and (optionally) the epochs.
        It can also save the events as a pickle file so it can easily be reused later.

        Parameters
        ----------
        save_path: string, optional
            A path to the folder where the data should be saved in. If not provided, the file will be saved in the
            directory the script is run from.

        save_events: boolean, optional
            Indicates whether the found events should be saved as pickle files as well.

        save_epochs: boolean, optional
            Indicates whether the created epochs should be saved as well.

        kwargs: dict, optional
            Will be passed on to the raw.save() method.
        """

        file_path_and_name = os.path.join(save_path, 'participant_{}_prepared_raw.fif'.format(self.participant_id))
        self.raw.save(file_path_and_name, **kwargs)
        print('Saved the prepared raw file to {}.'.format(file_path_and_name))

        if save_events:
            if self.events is None:
                raise AttributeError('No events to save. Please find them by running the find_events() method first.')

            file_path_and_name = os.path.join(save_path, '{}_events.pickle'.format(self.participant_id))
            pickle_file = open(file_path_and_name, 'wb')
            pickle.dump(self.events, pickle_file)
            pickle_file.close()
            print('Saved the events to {}.'.format(file_path_and_name))

        if save_epochs:
            if self.epochs is None:
                raise AttributeError('You have not created any epochs yet.\n'
                                     'Create them by running the make_epochs() method first.')

            file_path_and_name = os.path.join(save_path, 'participant_{}_epochs.fif'.format(self.participant_id))
            self.epochs.save(file_path_and_name)

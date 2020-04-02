import numpy as np
import pandas as pd
import mne
import os

#%matplotlib inline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
#import eeg_plotting as ep

sns.set_context("poster") 
sns.set_style("white")
sns.set_palette(sns.husl_palette(2))

proj_folder = '/Network/Servers/psychoserverx.psycho.unibas.ch/Volumes/datacenter/nfs_homepoint/lfontanesi/Python/EEG_experiment'
#proj_folder = os.getcwd()

#data = pd.read_pickle(os.path.join(proj_folder, 'data'))
data = pd.read_csv(os.path.join(proj_folder, 'data.csv'))

sub_id_EEG_array = np.arange(7, 40)
sub_id_EEG_array = np.delete(sub_id_EEG_array, [np.where(sub_id_EEG_array==x)[0][0] for x in [15, 16, 39]])

sub_id_beh_array = np.arange(1, 31)
#sub_id_beh_array = np.delete(sub_id_beh_array, [np.where(sub_id_beh_array==x)[0][0] for x in [9]]) # participant 15

n = len(sub_id_beh_array)


decim = 10
frequencies = np.arange(7, 60, 3)  # define frequencies of interest
n_cycles = frequencies / frequencies[0]
print(len(frequencies))
zero_mean = False  # don't correct morlet wavelet to be of mean zero
# To have a true wavelet zero_mean should be True but here for illustration
# purposes it helps to spot the evoked response.

from mne.time_frequency import tfr_morlet


# Bias towards Specific Option:
def lateralisation(eeg_epochs, behavior, block):
    # select block
    eeg_epochs = eeg_epochs.copy()
    eeg_epochs.drop(np.logical_not(np.array(behavior.event==block, dtype=bool)))
    behavior = behavior.copy()
    behavior = behavior[behavior.event==block].reset_index()
    
    eeg_epochs_tfr = tfr_morlet(eeg_epochs,
                                frequencies,
                                n_cycles=n_cycles, # The number of cycles globally or for each frequency.
                                decim=decim, # To reduce memory usage, decimation factor after time-frequency decomposition.
                                average=False, # If True average across Epochs.
                                zero_mean=zero_mean, # Make sure the wavelet has a mean of zero.
                                return_itc=False) # The inter-trial coherence (ITC)
    eeg_epochs_tfr.apply_baseline(mode='percent', baseline=(-200, 0))
    
    ch_index_r = [np.where(np.array(eeg_epochs_tfr.ch_names) == i)[0][0] for i in ['C6', 'C4', 'C2', 'FC4', 'FC2']]
    ch_index_l = [np.where(np.array(eeg_epochs_tfr.ch_names) == i)[0][0] for i in ['C5', 'C3', 'C1', 'FC3', "FC1"]]
    
    # for the 'high' condition: 
    trials_l = np.where((behavior.specific_key == 'q')&(behavior.means == 'high')&(behavior.outlier == False))[0]
    trials_r = np.where((behavior.specific_key == 'p')&(behavior.means == 'high')&(behavior.outlier == False))[0]

    trials_l_mean = np.mean(eeg_epochs_tfr.data[trials_l,:,:,:][:,ch_index_r,:,:], axis=1) - np.mean(eeg_epochs_tfr.data[trials_l,:,:,:][:,ch_index_l,:,:], axis=1)
    trials_r_mean = np.mean(eeg_epochs_tfr.data[trials_r,:,:,:][:,ch_index_l,:,:], axis=1) - np.mean(eeg_epochs_tfr.data[trials_r,:,:,:][:,ch_index_r,:,:], axis=1)
    
    trials_high_df = pd.DataFrame([])
    for i in np.arange(len(trials_l)):
        temp = pd.DataFrame(trials_l_mean[i,:,:], index=frequencies, columns=eeg_epochs_tfr.times)
        temp['epoch'] = trials_l[i]
        trials_high_df = trials_high_df.append(temp)
    for i in np.arange(len(trials_r)):
        temp = pd.DataFrame(trials_r_mean[i,:,:], index=frequencies, columns=eeg_epochs_tfr.times)
        temp['epoch'] = trials_r[i]
        trials_high_df = trials_high_df.append(temp)    

    tfr_high = (np.mean(trials_l_mean, axis=0)+np.mean(trials_r_mean, axis=0))/2 #averaged across trials
    
    # for the 'low' condition: 
    trials_l = np.where((behavior.specific_key == 'p')&(behavior.means == 'low')&(behavior.outlier == False))[0]
    trials_r = np.where((behavior.specific_key == 'q')&(behavior.means == 'low')&(behavior.outlier == False))[0]

    trials_l_mean = np.mean(eeg_epochs_tfr.data[trials_l,:,:,:][:,ch_index_r,:,:], axis=1) - np.mean(eeg_epochs_tfr.data[trials_l,:,:,:][:,ch_index_l,:,:], axis=1)
    trials_r_mean = np.mean(eeg_epochs_tfr.data[trials_r,:,:,:][:,ch_index_l,:,:], axis=1) - np.mean(eeg_epochs_tfr.data[trials_r,:,:,:][:,ch_index_r,:,:], axis=1)
    
    trials_low_df = pd.DataFrame([])
    for i in np.arange(len(trials_l)):
        temp = pd.DataFrame(trials_l_mean[i,:,:], index=frequencies, columns=eeg_epochs_tfr.times)
        temp['epoch'] = trials_l[i]
        trials_low_df = trials_low_df.append(temp)
    for i in np.arange(len(trials_r)):
        temp = pd.DataFrame(trials_r_mean[i,:,:], index=frequencies, columns=eeg_epochs_tfr.times)
        temp['epoch'] = trials_r[i]
        trials_low_df = trials_low_df.append(temp)

    tfr_low = (np.mean(trials_l_mean, axis=0)+np.mean(trials_r_mean, axis=0))/2 #averaged across trials
    
    return tfr_high, tfr_low, eeg_epochs_tfr.times, trials_high_df, trials_low_df
    

for pp in np.arange(n)[1:]:
    sub_id_EEG = sub_id_EEG_array[pp]
    sub_id_beh = sub_id_beh_array[pp]
    print("")
    print("Start with participant %s-%s..." % (sub_id_EEG, sub_id_beh))
    
    data_sub = data[data.participant==sub_id_beh]
    print("N trials = %s" % data_sub.shape[0])
    
    epochs = mne.read_epochs(os.path.join(proj_folder, 'EEG_data_clean', 'subject_%s_event2-epo.fif' % sub_id_EEG))
    
    (learning_high, learning_low, times, learning_high_t, learning_low_t) = lateralisation(epochs, data_sub, block='choice_learning')
    (description_high, description_low, times, description_high_t, description_low_t) = lateralisation(epochs, data_sub, block='choice_description')
    
    #save to file:
    mean_LRP = pd.DataFrame([])
    
    temp = pd.DataFrame({'means': np.repeat('high', learning_high.shape[0]),
                         'event': np.repeat('learning', learning_high.shape[0])})
    temp = pd.concat([temp, pd.DataFrame(learning_high, columns=times)], axis=1)
    mean_LRP = mean_LRP.append(temp)
    
    temp = pd.DataFrame({'means': np.repeat('low', learning_low.shape[0]),
                         'event': np.repeat('learning', learning_low.shape[0])})
    temp = pd.concat([temp, pd.DataFrame(learning_low, columns=times)], axis=1)
    mean_LRP = mean_LRP.append(temp)
    
    temp = pd.DataFrame({'means': np.repeat('high', description_high.shape[0]),
                         'event': np.repeat('description', description_high.shape[0])})
    temp = pd.concat([temp, pd.DataFrame(description_high, columns=times)], axis=1)
    mean_LRP = mean_LRP.append(temp)
    
    temp = pd.DataFrame({'means': np.repeat('low', description_low.shape[0]),
                         'event': np.repeat('description', description_low.shape[0])})
    temp = pd.concat([temp, pd.DataFrame(description_low, columns=times)], axis=1)
    mean_LRP = mean_LRP.append(temp)
    
    mean_LRP.to_csv(os.path.join(proj_folder, 'EEG_data_clean','subject_%s_event2-LRPpow-average.csv' % sub_id_EEG))
    
    description_high_t.to_csv(os.path.join(proj_folder, 'EEG_data_clean','subject_%s_event2-LRPpow-trials-D_high.csv' % sub_id_EEG))
    description_low_t.to_csv(os.path.join(proj_folder, 'EEG_data_clean','subject_%s_event2-LRPpow-trials-D_low.csv' % sub_id_EEG))
    learning_high_t.to_csv(os.path.join(proj_folder, 'EEG_data_clean','subject_%s_event2-LRPpow-trials-L_high.csv' % sub_id_EEG))
    learning_low_t.to_csv(os.path.join(proj_folder, 'EEG_data_clean','subject_%s_event2-LRPpow-trials-L_low.csv' % sub_id_EEG))
    
    # plot
    plt.figure(figsize=(18, 18))
    grid = gridspec.GridSpec(2, 2, hspace=.3, wspace=.2)

    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[0, 1])
    ax3 = plt.subplot(grid[1, 0])
    ax4 = plt.subplot(grid[1, 1])

    sns.heatmap(learning_high, ax=ax1, cbar_kws={'label': '%'});
    ax1.set_xticks(np.linspace(0, learning_high.shape[1], 12));
    ax1.set_xticklabels(np.linspace(-.2, 2, 12));
    ax1.set_yticklabels(frequencies[::-1]);
    ax1.set_title('Learning, High Mean')
    ax1.set_xlabel('Response locked (s)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(np.linspace(0, learning_high.shape[1], 12)[1], color='black', linestyle='dotted');

    sns.heatmap(learning_low, ax=ax2, cbar_kws={'label': '%'});
    ax2.set_xticks(np.linspace(0, learning_low.shape[1], 12));
    ax2.set_xticklabels(np.linspace(-.2, 2, 12));
    ax2.set_yticklabels(frequencies[::-1]);
    ax2.set_title('Learning, Low Mean')
    ax2.set_xlabel('Response locked (s)')
    ax2.set_ylabel('Frequency')
    ax2.axvline(np.linspace(0, learning_low.shape[1], 12)[1], color='black', linestyle='dotted');
    
    sns.heatmap(description_high, ax=ax3, cbar_kws={'label': '%'});
    ax3.set_xticks(np.linspace(0, description_high.shape[1], 12));
    ax3.set_xticklabels(np.linspace(-.2, 2, 12));
    ax3.set_yticklabels(frequencies[::-1]);
    ax3.set_title('Description, High Mean')
    ax3.set_xlabel('Response locked (s)')
    ax3.set_ylabel('Frequency')
    ax3.axvline(np.linspace(0, description_high.shape[1], 12)[1], color='black', linestyle='dotted');
    
    sns.heatmap(description_low, ax=ax4, cbar_kws={'label': '%'});
    ax4.set_xticks(np.linspace(0, description_low.shape[1], 12));
    ax4.set_xticklabels(np.linspace(-.2, 2, 12));
    ax4.set_yticklabels(frequencies[::-1]);
    ax4.set_title('Description, Low Mean')
    ax4.set_xlabel('Response locked (s)')
    ax4.set_ylabel('Frequency')
    ax4.axvline(np.linspace(0, description_low.shape[1], 12)[1], color='black', linestyle='dotted');

    #sns.despine()
    plt.savefig(os.path.join(proj_folder, 'plot_event2-LRPpow','subject_%s-LRP.pdf' % sub_id_EEG), dpi=300);
    plt.clf()
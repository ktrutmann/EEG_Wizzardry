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


decim = 2
frequencies = np.arange(7, 80, 2)  # define frequencies of interest
n_cycles = frequencies / frequencies[0]
print(len(frequencies))
zero_mean = False  # don't correct morlet wavelet to be of mean zero
# To have a true wavelet zero_mean should be True but here for illustration
# purposes it helps to spot the evoked response.

from mne.time_frequency import tfr_morlet


# Bias towards Specific Option:
def LRP(eeg_epochs, behavior, block, option, n_trials=98, z=1.96):
    # select block
    eeg_epochs = eeg_epochs.copy()
    eeg_epochs.drop(np.logical_not(np.array(behavior.event==block, dtype=bool)))
    behavior = behavior.copy()
    behavior = behavior[behavior.event==block].reset_index()
    
    if option == 'S':
        # select specific
        eeg_epochs.drop(np.logical_not(np.array(behavior.means=='high', dtype=bool)))
        behavior = behavior[behavior.means=='high']
        print('n events: %s' % eeg_epochs.events.shape[0])
    else:
        # select reference
        eeg_epochs.drop(np.logical_not(np.array(behavior.means=='low', dtype=bool)))
        behavior = behavior[behavior.means=='low']
        print('n events: %s' % eeg_epochs.events.shape[0])
    
    # left LRP
    eeg_epochs_l = eeg_epochs.copy()
    eeg_epochs_l.drop(np.logical_not(np.array(behavior.specific_key=='q', dtype=bool)))
    print('Left LRP...')
    print('n events: %s' % eeg_epochs.events.shape[0])
    
    eeg_epochs_l_tfr = tfr_morlet(eeg_epochs_l,
                                  frequencies,
                                  n_cycles=n_cycles, # The number of cycles globally or for each frequency.
                                  decim=decim, # To reduce memory usage, decimation factor after time-frequency decomposition.
                                  average=False, # If True average across Epochs.
                                  zero_mean=zero_mean, # Make sure the wavelet has a mean of zero.
                                  return_itc=False) # The inter-trial coherence (ITC)
    eeg_epochs_l_tfr.apply_baseline(mode='percent', baseline=(-200, 0))
    
    ch_index_r = np.where(np.array(eeg_epochs_l_tfr.ch_names) == 'C4')[0][0]
    ch_index_l = np.where(np.array(eeg_epochs_l_tfr.ch_names) == 'C3')[0][0]
    
    left_LRP = eeg_epochs_l_tfr.data[:, ch_index_r, :, :] - eeg_epochs_l_tfr.data[:, ch_index_l, :, :]
    left_LRP_df = pd.DataFrame([])
    for i in np.arange(left_LRP.shape[0]):
        temp = pd.DataFrame(left_LRP[i, :, :], index=eeg_epochs_l_tfr.freqs, columns=eeg_epochs_l_tfr.times)
        temp['epoch'] = i
        left_LRP_df = left_LRP_df.append(temp)
    
    mean_left_LRP = np.mean(left_LRP, axis=0)
    
    # right LRP
    eeg_epochs_r = eeg_epochs.copy()
    eeg_epochs_r.drop(np.logical_not(np.array(behavior.specific_key=='p', dtype=bool)))
    print('Right LRP...')
    print('n events: %s' % eeg_epochs.events.shape[0])
    
    eeg_epochs_r_tfr = tfr_morlet(eeg_epochs_r,
                                  frequencies,
                                  n_cycles=n_cycles, # The number of cycles globally or for each frequency.
                                  decim=decim, # To reduce memory usage, decimation factor after time-frequency decomposition.
                                  average=False, # If True average across Epochs.
                                  zero_mean=zero_mean, # Make sure the wavelet has a mean of zero.
                                  return_itc=False) # The inter-trial coherence (ITC)
                                  
    eeg_epochs_r_tfr.apply_baseline(mode='percent', baseline=(-200, 0))
    
    ch_index_r = np.where(np.array(eeg_epochs_r_tfr.ch_names) == 'C4')[0][0]
    ch_index_l = np.where(np.array(eeg_epochs_r_tfr.ch_names) == 'C3')[0][0]
    
    right_LRP = eeg_epochs_r_tfr.data[:, ch_index_l, :, :] - eeg_epochs_r_tfr.data[:, ch_index_r, :, :]
    right_LRP_df = pd.DataFrame([])
    for i in np.arange(right_LRP.shape[0]):
        temp = pd.DataFrame(right_LRP[i, :, :], index=eeg_epochs_r_tfr.freqs, columns=eeg_epochs_r_tfr.times)
        temp['epoch'] = i
        right_LRP_df = right_LRP_df.append(temp)
        
    mean_right_LRP = np.mean(right_LRP, axis=0)
    
    # merge
    mean_LRP = (mean_left_LRP+mean_right_LRP)/2.
    mean_LRP_df = pd.DataFrame(mean_LRP)
    mean_LRP_df.index = eeg_epochs_l_tfr.freqs
    mean_LRP_df.columns = eeg_epochs_l_tfr.times
    
    LRP_df = (left_LRP_df+right_LRP_df)/2.
    
    return mean_LRP_df, LRP_df
    

for pp in np.arange(n):
    sub_id_EEG = sub_id_EEG_array[pp]
    sub_id_beh = sub_id_beh_array[pp]
    print("")
    print("Start with participant %s-%s..." % (sub_id_EEG, sub_id_beh))
    
    data_sub = data[data.participant==sub_id_beh]
    print("N trials = %s" % data_sub.shape[0])
    
    epochs = mne.read_epochs(os.path.join(proj_folder, 'EEG_data_clean', 'subject_%s_event2-epo.fif' % sub_id_EEG))
    
    learning_S_LRP = LRP(epochs, data_sub, block='choice_learning', option='S')
    learning_R_LRP = LRP(epochs, data_sub, block='choice_learning', option='R')
    description_S_LRP = LRP(epochs, data_sub, block='choice_description', option='S')
    description_R_LRP = LRP(epochs, data_sub, block='choice_description', option='R')
    
    #save to file:
    mean_LRP = pd.DataFrame([])
    
    temp = learning_S_LRP[0].copy()
    temp['Cor_option'] = 'S'
    temp['event'] = 'learning'
    mean_LRP = mean_LRP.append(temp)
    
    temp = learning_R_LRP[0].copy()
    temp['Cor_option'] = 'R'
    temp['event'] = 'learning'
    mean_LRP = mean_LRP.append(temp)
    
    temp = description_S_LRP[0].copy()
    temp['Cor_option'] = 'S'
    temp['event'] = 'description'
    mean_LRP = mean_LRP.append(temp)
    
    temp = description_R_LRP[0].copy()
    temp['Cor_option'] = 'R'
    temp['event'] = 'description'
    mean_LRP = mean_LRP.append(temp)
    
    
    trial_LRP = pd.DataFrame([])
    
    temp = learning_S_LRP[1].copy()
    temp['Cor_option'] = 'S'
    temp['event'] = 'learning'
    trial_LRP = trial_LRP.append(temp)
    
    temp = learning_R_LRP[1].copy()
    temp['Cor_option'] = 'R'
    temp['event'] = 'learning'
    trial_LRP = trial_LRP.append(temp)
    
    temp = description_S_LRP[1].copy()
    temp['Cor_option'] = 'S'
    temp['event'] = 'description'
    trial_LRP = trial_LRP.append(temp)
    
    temp = description_R_LRP[1].copy()
    temp['Cor_option'] = 'R'
    temp['event'] = 'description'
    trial_LRP = trial_LRP.append(temp)
    
    mean_LRP.to_csv(os.path.join(proj_folder, 'EEG_data_clean','subject_%s_event2-LRPpow-average.csv' % sub_id_EEG))
    trial_LRP.to_csv(os.path.join(proj_folder, 'EEG_data_clean','subject_%s_event2-LRPpow-trial.csv' % sub_id_EEG))
    
    
    # plot
    plt.figure(figsize=(18, 18))
    grid = gridspec.GridSpec(2, 2, hspace=.3, wspace=.2)

    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[0, 1])
    ax3 = plt.subplot(grid[1, 0])
    ax4 = plt.subplot(grid[1, 1])

    sns.heatmap(learning_S_LRP[0], ax=ax1, cbar_kws={'label': '%'});
    ax1.set_xticks(np.linspace(0, learning_S_LRP[0].shape[1], 12));
    ax1.set_xticklabels(np.linspace(-.2, 2, 12));
    ax1.set_yticklabels(learning_S_LRP[0].index.astype(int)[::-1]);
    ax1.set_title('Learning, Option S')
    ax1.set_xlabel('Response locked (s)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(np.linspace(0, learning_S_LRP[0].shape[1], 12)[1], color='black', linestyle='dotted');

    sns.heatmap(learning_R_LRP[0], ax=ax2, cbar_kws={'label': '%'});
    ax2.set_xticks(np.linspace(0, learning_R_LRP[0].shape[1], 12));
    ax2.set_xticklabels(np.linspace(-.2, 2, 12));
    ax2.set_yticklabels(learning_R_LRP[0].index.astype(int)[::-1]);
    ax2.set_title('Learning, Option R')
    ax2.set_xlabel('Response locked (s)')
    ax2.set_ylabel('Frequency')
    ax2.axvline(np.linspace(0, learning_R_LRP[0].shape[1], 12)[1], color='black', linestyle='dotted');
    
    sns.heatmap(description_S_LRP[0], ax=ax3, cbar_kws={'label': '%'});
    ax3.set_xticks(np.linspace(0, description_S_LRP[0].shape[1], 12));
    ax3.set_xticklabels(np.linspace(-.2, 2, 12));
    ax3.set_yticklabels(description_S_LRP[0].index.astype(int)[::-1]);
    ax3.set_title('Description, Option S')
    ax3.set_xlabel('Response locked (s)')
    ax3.set_ylabel('Frequency')
    ax3.axvline(np.linspace(0, description_S_LRP[0].shape[1], 12)[1], color='black', linestyle='dotted');
    
    sns.heatmap(description_R_LRP[0], ax=ax4, cbar_kws={'label': '%'});
    ax4.set_xticks(np.linspace(0, description_R_LRP[0].shape[1], 12));
    ax4.set_xticklabels(np.linspace(-.2, 2, 12));
    ax4.set_yticklabels(description_R_LRP[0].index.astype(int)[::-1]);
    ax4.set_title('Description, Option R')
    ax4.set_xlabel('Response locked (s)')
    ax4.set_ylabel('Frequency')
    ax4.axvline(np.linspace(0, description_R_LRP[0].shape[1], 12)[1], color='black', linestyle='dotted');

    #sns.despine()
    plt.savefig(os.path.join(proj_folder, 'plot_event2-LRPpow','subject_%s-LRP.pdf' % sub_id_EEG), dpi=300);
    plt.clf()
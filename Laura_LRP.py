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


# Bias towards Specific Option:
def LRP(eeg_epochs, behavior, block, option, n_trials=98, z=1.96):
    # select block
    eeg_epochs = eeg_epochs.copy()
    cond = np.array(np.logical_or(np.logical_not(behavior.event==block), behavior.outlier), dtype=bool)
    eeg_epochs.drop(cond)
    behavior = behavior.copy()
    behavior = behavior[np.logical_and(behavior.event==block, np.logical_not(behavior.outlier))].reset_index()
    
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
    
    df = eeg_epochs_l.to_data_frame().reset_index()
    group_mean = df.groupby(['condition', 'time']).mean().reset_index()
    group_var = df.groupby(['condition', 'time']).var().reset_index()
    
    left_LRP = (group_mean['C4']-group_mean['C3'])
    left_LRP_var = (group_var['C4']+group_var['C3'])
    
    # right LRP
    eeg_epochs_r = eeg_epochs.copy()
    eeg_epochs_r.drop(np.logical_not(np.array(behavior.specific_key=='p', dtype=bool)))
    print('Right LRP...')
    print('n events: %s' % eeg_epochs.events.shape[0])
    
    df = eeg_epochs_r.to_data_frame().reset_index()
    group_mean = df.groupby(['condition', 'time']).mean().reset_index()
    group_var = df.groupby(['condition', 'time']).var().reset_index()
    
    right_LRP = (group_mean['C3']-group_mean['C4'])
    right_LRP_var = (group_var['C3']+group_var['C4'])
    
    out = group_mean[['time']].copy()
    out['average_LRP'] = (left_LRP+right_LRP)/2
    out['sd_LRP'] = np.sqrt(left_LRP_var+right_LRP_var)/2
    out['95_CI_up'] = out['average_LRP'] + z*((np.sqrt(left_LRP_var+right_LRP_var)/2)/np.sqrt(n_trials))
    out['95_CI_lo'] = out['average_LRP'] - z*((np.sqrt(left_LRP_var+right_LRP_var)/2)/np.sqrt(n_trials))
    
    return out
    

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
    
    plt.figure(figsize=(20, 8))
    grid = gridspec.GridSpec(1, 2, wspace=.1)

    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[0, 1], sharey=ax1)

    ax1.plot(learning_S_LRP.time, learning_S_LRP.average_LRP, label='S')
    ax1.plot(learning_R_LRP.time, learning_R_LRP.average_LRP, label='R')
    ax1.axhline(0, color='black', linestyle='--', alpha=.7)
    ax1.axvline(0, color='black', linestyle='--', alpha=.7)
    ax1.legend(frameon=True, title='Correct option:', loc=2)

    ax1.fill_between(learning_S_LRP.time,
                     learning_S_LRP['95_CI_lo'], 
                     learning_S_LRP['95_CI_up'], 
                     where=learning_S_LRP['95_CI_lo'] <= learning_S_LRP['95_CI_up'],
                     alpha=.3, 
                     zorder=5, 
                     antialiased=True,
                     color=sns.color_palette()[0]);

    ax1.fill_between(learning_R_LRP.time,
                     learning_R_LRP['95_CI_lo'], 
                     learning_R_LRP['95_CI_up'], 
                     where=learning_R_LRP['95_CI_lo'] <= learning_R_LRP['95_CI_up'],
                     alpha=.3, 
                     zorder=5, 
                     antialiased=True,
                     color=sns.color_palette()[1]);

    ax1.set_xlim(-200, 2000)
    ax1.set_title('Learning block')
    ax1.set_ylabel('micro V')
    ax1.set_xlabel('ms (stimulus locked)');


    ax2.plot(description_S_LRP.time, description_S_LRP.average_LRP)
    ax2.plot(description_R_LRP.time, description_R_LRP.average_LRP)
    ax2.axhline(0, color='black', linestyle='--', alpha=.7)
    ax2.axvline(0, color='black', linestyle='--', alpha=.7)

    ax2.fill_between(description_S_LRP.time,
                     description_S_LRP['95_CI_lo'], 
                     description_S_LRP['95_CI_up'], 
                     where=description_S_LRP['95_CI_lo'] <= description_S_LRP['95_CI_up'],
                     alpha=.3, 
                     zorder=5, 
                     antialiased=True,
                     color=sns.color_palette()[0]);

    ax2.fill_between(description_R_LRP.time,
                     description_R_LRP['95_CI_lo'], 
                     description_R_LRP['95_CI_up'], 
                     where=description_R_LRP['95_CI_lo'] <= description_R_LRP['95_CI_up'],
                     alpha=.3, 
                     zorder=5, 
                     antialiased=True,
                     color=sns.color_palette()[1]);

    ax2.set_xlim(-200, 2000)
    ax2.set_title('Description block')
    ax2.set_xlabel('ms (stimulus locked)');

    sns.despine()
    plt.savefig(os.path.join(proj_folder, 'plot_event2-LRP','subject_%s-LRP.pdf' % sub_id_EEG), dpi=300);
    plt.clf()
    
    learning_S_LRP['Cor_option'] = 'S'
    learning_R_LRP['Cor_option'] = 'R'
    learning_S_LRP = learning_S_LRP.append(learning_R_LRP)
    learning_S_LRP['event'] = 'learning'
    
    description_S_LRP['Cor_option'] = 'S' 
    description_R_LRP['Cor_option'] = 'R'
    description_S_LRP = description_S_LRP.append(description_R_LRP)
    description_S_LRP['event'] = 'description'
    
    learning_S_LRP = learning_S_LRP.append(description_S_LRP)
    learning_S_LRP['participant_eeg'] = sub_id_EEG
    learning_S_LRP['participant_beh'] = sub_id_beh
    
    learning_S_LRP.to_csv(os.path.join(proj_folder, 'EEG_data_clean','subject_%s_event2-LRP.csv' % sub_id_EEG))
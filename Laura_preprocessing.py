import numpy as np
import pandas as pd
import mne
import os

#%matplotlib inline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
#import eeg_plotting as ep

sns.set_context("poster") 
sns.set_style("white")
sns.set_palette(sns.husl_palette(4, h=.5, l=.65, s=.9))

proj_folder = '/Network/Servers/psychoserverx.psycho.unibas.ch/Volumes/datacenter/nfs_homepoint/lfontanesi/Python/EEG_experiment'
#proj_folder = os.getcwd()

sub_id = 39
filepath_EEG = 'EEG_data/Participant_%s.bdf' % sub_id
filepath_mne = '/usr/local/anaconda/lib/python3.5/site-packages'
#filepath_mne = '/usr/local/lib/python2.7/site-packages'



print('LOAD EEG DATA:') 
raw = mne.io.read_raw_edf(os.path.join(proj_folder, filepath_EEG), preload=True)



print('Fixing channels names and montage:')
for a in raw.ch_names[:-1]:
    raw.rename_channels({a:a[2:]})

montage = mne.channels.read_montage(kind = 'biosemi64',
                                    path = filepath_mne + '/mne/channels/data/montages')
raw.set_montage(montage)



print('Drop empty channels:')
print('N of channels: %s' % len(raw.ch_names))
raw.drop_channels(raw.ch_names[64+9:len(raw.ch_names)-1])
print('N of channels: %s' % len(raw.ch_names))



print('Add mean temples-electrodes and reference EOG channels:')
raw_temp = raw.copy().pick_channels(['FT9', 'PO9'])
data = raw_temp._data
data[..., np.array(raw_temp.ch_names) == 'PO9', :] += data[..., np.array(raw_temp.ch_names) == 'FT9', :]
data[..., np.array(raw_temp.ch_names) == 'PO9', :] /= 2
raw_temp.rename_channels({'PO9': 'ref'})
print (raw_temp.ch_names)
raw_temp.drop_channels(['FT9'])
print (raw_temp.ch_names)
raw.add_channels([raw_temp], force_update_info=True)



print('Re-define EOG and EMG channels:')
mne.set_bipolar_reference(raw, 
                          anode=['PO10', 'HeRe', 'HeLi', 'VeUp'], 
                          cathode=['FT10', 'ref', 'VeDo', 'EMG1a'], 
                          ch_name=None, 
                          copy=False)



print('Mark MISC, EOG and EMG channels:')
raw.set_channel_types({'FT9':'misc',
                       'PO9':'misc',
                       'PO10-FT10':'eog',
                       'HeRe-ref':'eog',
                       'HeLi-VeDo':'emg',
                       'VeUp-EMG1a':'emg'})
                       
                       
                       
print('Define event id:')
events_id = dict(show_options=2,
                 start_choice=4,
                 end_choice=8,
                 feedback=16,
                 feedback_null=32,
                 end_trial=64)
                 
                 
                 
print('Plot raw to file:')
fig = raw.plot(order=np.arange(0, 20), show=False)
fig.savefig(os.path.join(proj_folder, 'plot_raw', 'subject_%s_a.pdf' % sub_id));

fig = raw.plot(order=np.arange(20, 40), show=False)
fig.savefig(os.path.join(proj_folder, 'plot_raw', 'subject_%s_b.pdf' % sub_id));

fig = raw.plot(order=np.arange(40, 60), show=False)
fig.savefig(os.path.join(proj_folder, 'plot_raw', 'subject_%s_c.pdf' % sub_id));

fig = raw.plot(order=np.arange(60, 71), show=False)
fig.savefig(os.path.join(proj_folder, 'plot_raw', 'subject_%s_d.pdf' % sub_id));       




print('START PREPROCESSING...')  
picks = mne.pick_types(raw.info, eeg=True, eog=True, emg=True)

print('Re-referencing:')
raw, _ = mne.io.set_eeg_reference(raw, ['FT9', 'PO9'])

print('NOTCH FILTERING (for power-line noise):')
raw.notch_filter(np.arange(50, 451, 50), picks=picks, filter_length='auto', phase='zero')
print('BANDPASS FILTERING:')
raw.filter(.1, 100, picks=picks, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero')


print('Find bad channels:')
events = mne.find_events(raw, min_duration=2/raw.info['sfreq'])

tmin, tmax = -.2, 2
picks = mne.pick_types(raw.info, eeg=True)

epochs = mne.Epochs(raw, 
                    events=events, 
                    event_id=[2, 4, 16], 
                    tmin=tmin,
                    tmax=tmax,
                    preload=True, 
                    picks=picks)
                    
df = epochs.to_data_frame()
df.reset_index(inplace=True)

group = df.groupby(['condition', 'epoch'])
mean = group.mean()
std = group.std()

a = mean.std()
a = a[1:]
print('standard deviation of mean across epochs:')
print(np.mean(a), np.std(a))
print('higher than 40:')
print(a[a>40].index)

for i in a[a>40].index:
    raw.info['bads'].append(i)
print(np.array(raw.info['bads']), len(raw.info['bads']))

if len(raw.info['bads']) > 0:
    raw.interpolate_bads(reset_bads=True)
    
    epochs = mne.Epochs(raw, 
                        events=events, 
                        event_id=[2, 4, 16], 
                        tmin=tmin,
                        tmax=tmax,
                        preload=True, 
                        picks=picks)
                    
    df = epochs.to_data_frame()
    df.reset_index(inplace=True)

    group = df.groupby(['condition', 'epoch'])
    mean = group.mean()
    std = group.std()

    a = mean.std()
    a = a[1:]
    print('standard deviation of mean across epochs:')
    print(np.mean(a), np.std(a))
    print('higher than 40:')
    print(a[a>40].index)

    for i in a[a>40].index:
        raw.info['bads'].append(i)
    print(np.array(raw.info['bads']), len(raw.info['bads']))
    
print('Plot filtered data to file:')
fig = raw.plot(order=np.arange(0, 20), show=False)
fig.savefig(os.path.join(proj_folder, 'plot_raw_filtered', 'subject_%s_a.pdf' % sub_id));

fig = raw.plot(order=np.arange(20, 40), show=False)
fig.savefig(os.path.join(proj_folder, 'plot_raw_filtered', 'subject_%s_b.pdf' % sub_id));

fig = raw.plot(order=np.arange(40, 60), show=False)
fig.savefig(os.path.join(proj_folder, 'plot_raw_filtered', 'subject_%s_c.pdf' % sub_id));

fig = raw.plot(order=np.arange(60, 71), show=False)
fig.savefig(os.path.join(proj_folder, 'plot_raw_filtered', 'subject_%s_d.pdf' % sub_id));



print('ICA for event 2, stimulus locked, with baseline, no threshold:')

tmin, tmax = -.2, 2
picks = mne.pick_types(raw.info, eeg=True, emg=True, eog=True, exclude='bads')
baseline = (-.2, 0.0)

epochs = mne.Epochs(raw, 
                    events=events, 
                    event_id=[2], 
                    tmin=tmin,
                    tmax=tmax,
                    preload=True,
                    baseline=baseline,
                    picks=picks)
                    
from mne.preprocessing import ICA, create_eog_epochs

n_components = 25  # if float, select n_components by explained variance of PCA
method = 'fastica'  # for comparison with EEGLAB try "extended-infomax" here
decim = 3  # we need sufficient statistics, not all time points -> saves time

ica = ICA(n_components=n_components, method=method)
print(ica)

ica.fit(epochs, picks=None, decim=decim)
print(ica)

eog_inds, scores = ica.find_bads_eog(epochs)

for comp in eog_inds:
    a = ica.plot_properties(epochs, picks=comp, figsize=[15, 10], 
                            psd_args={'fmax': 35.}, image_args={'sigma': 1.});
    a[0].savefig(os.path.join(proj_folder, 'plot_ICA_excluded', 'subject_%s_event2-%s.pdf' % (sub_id, comp)), dpi=300)
    
ica.apply(epochs, exclude=eog_inds)

epochs.save(os.path.join(proj_folder, 'EEG_data_clean', 'subject_%s_event2-epo.fif' % sub_id))
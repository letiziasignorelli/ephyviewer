import numpy as np
import os
import pathlib
import pywt
import scipy
from scipy import stats
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from ephyviewer import mkQApp, MainViewer, TraceViewer, TimeFreqViewer
from ephyviewer import InMemoryAnalogSignalSource
from ephyviewer import InMemoryTimeFreqSource

def compute_spect(data, mid_f, tstep, fs, exp_cor = 1, t_smooth = 2, zscore=False):
    wavelet_name = 'cmor2.0-2.0'

    # Calculate equivalent scale range
    min_freq = mid_f[0]  # Hz
    max_freq = mid_f[1]  # Hz
    freq_resolution = tstep  # Hz

    # Frequency vector
    frequencies = np.arange(min_freq, max_freq + freq_resolution, freq_resolution)

    scales = pywt.central_frequency(wavelet_name) / (frequencies / fs)

    # Compute CWT
    mwt_m, freqs_m = pywt.cwt(data, scales, wavelet_name, sampling_period=1/fs)

    if np.iscomplexobj(mwt_m):
        powers_m = abs(mwt_m)

    # Exponential correction
    freqs_m_exp = freqs_m**exp_cor
    correction = np.tile(freqs_m_exp[:, np.newaxis], (1, powers_m.shape[1]))
    correction = correction / correction[0, 0]

    powers_m_corrected = powers_m * correction

    logpowers_m = np.log10(powers_m_corrected)**2

    # Gaussian smoothing (along the time axis)
    sigma = t_smooth * fs # The 'sigma' parameter for the gaussian filter is in samples
    powers_m_smooth = gaussian_filter(logpowers_m, sigma=[0, sigma], mode='reflect')
    
    if zscore:
        powers_m_smooth = stats.zscore(powers_m_smooth)
    
    return freqs_m, powers_m_smooth


def bmfilter(raw_signal, filter_type, filter_range, fs):
    filter_order = 10
    filter_type = filter_type.lower()
    bandpass_string = 'bandpass'
    lowpass_string = 'lowpass'
    highpass_string = 'highpass'
    bandstop_string = 'bandstop'

    if filter_type == bandpass_string or filter_type == bandstop_string:
        if len(filter_range) != 2:
            print('Need two numbers for bandpass or bandstop filter')
            return


    current_filter = scipy.signal.butter(N = filter_order, Wn = filter_range, btype = filter_type, fs = fs, output = 'sos' )

    filtered_signal = scipy.signal.sosfiltfilt(current_filter, raw_signal)

    return filtered_signal

def downsample(input_data, downsampling_factor, downsampling_frequency, fs):
    
    downsampled_trace_size = int(round(len(input_data) / downsampling_factor))
    
    filtered_trace = bmfilter(input_data, filter_type = 'lowpass', filter_range = downsampling_frequency, fs = fs )
    downsampled_trace = np.array(scipy.signal.resample(filtered_trace, downsampled_trace_size))
    downsampled_trace = np.reshape(downsampled_trace, (downsampled_trace_size, 1 ))    
    
    return downsampled_trace[:,0] 

# Define parameters for time and frequency analysis
segment_duration = 10  # seconds (for a 10-second segment)
animal_list = ['69505_4']
root_directory = r'Z:\NEW_Data\Animals\data'
new_fs = 200  # New sampling frequency after downsampling

# Load data and extract a 10-second segment
current_rat = animal_list[0]
current_path = os.path.join(root_directory, current_rat, r'Preprocessing\Downsample')
current_file = pathlib.Path(os.path.join(current_path, '2024-03-21_15_downsampled_ls_1200.npz'))

# Load data
downsampled_bundle = np.load(current_file)
fs = downsampled_bundle['fs']
cortex_ecog = downsampled_bundle['ecog_traces']
par_channel = downsampled_bundle['par_idx']
data = cortex_ecog[:, par_channel]

# Downsample data
downsampling_factor = int(fs / new_fs)
data_downsampled = downsample(data, downsampling_factor, new_fs, int(fs))
data_downsampled = data_downsampled[:segment_duration * new_fs]

# Create frequency parameters
mid_f = [0.5, 50]  # Frequency range of interest
tstep = 0.5
lb = 3
ub = 10
scaling = 0.5
norm = None

# Calculate spectrogram for the 10-second segment
freqs_m, powers_m_smooth = compute_spect(data_downsampled, mid_f, lb, ub, tstep, scaling, norm, new_fs, exp_cor=0.5, t_smooth=0, zscore=False)

# Visualization using ephyviewer
app = mkQApp()

# Create trace viewer
source1 = InMemoryAnalogSignalSource(np.expand_dims(data_downsampled, axis=1), new_fs, 0, channel_names=['ECoG'])
trace_view = TraceViewer(source=source1, name='ECoG Trace')
trace_view.auto_scale()

# Create time-frequency viewer for raw computation
tfr_view = TimeFreqViewer(source=source1, name='Real-Time TFR')
tfr_view.params['show_axis'] = True
tfr_view.params['timefreq', 'deltafreq'] = .5
tfr_view.params['timefreq', 'f_start'] = .5
tfr_view.params['timefreq', 'f_stop'] = 50
tfr_view.params['colormap'] = 'jet'
tfr_view.auto_scale()

# Create precomputed time-frequency viewer using spectrogram
source2 = InMemoryTimeFreqSource(spectrogram=powers_m_smooth, frequencies=freqs_m, sample_rate=new_fs, t_start=0)
tfr_precomputed_view = TimeFreqViewer(source=source2, name='Precomputed TFR')
tfr_precomputed_view.params['timefreq', 'f_start'] = freqs_m[0]
tfr_precomputed_view.params['timefreq', 'f_stop'] = freqs_m[-1]
tfr_precomputed_view.params['colormap'] = 'jet'
tfr_precomputed_view.auto_scale()

# Create a main viewer window
win = MainViewer(debug=True, show_auto_scale=True)

# Add views to the main window
win.add_view(trace_view)
win.add_view(tfr_view)
win.add_view(tfr_precomputed_view)

win.show()
app.exec_()

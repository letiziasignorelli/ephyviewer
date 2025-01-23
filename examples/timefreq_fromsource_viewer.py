from ephyviewer import mkQApp, MainViewer, TraceViewer, TimeFreqViewer
from ephyviewer import InMemoryAnalogSignalSource
import numpy as np
import pywt  # Ensure that pywt is installed

# Create a main Qt application (for event loop)
app = mkQApp()

# Create fake 3-channel time-domain signals with 100,000 samples at 1000 Hz
sigs = np.random.rand(1000, 3)
sample_rate = 200
t_start = 0.01

min_freq = 0.01
max_freq = 100
freq_resolution = 1

frequencies = np.arange(min_freq, max_freq + freq_resolution, freq_resolution)

scales = pywt.central_frequency('cmor2.0-2.0') / (frequencies / sample_rate)

# Compute CWTs and stack them together
cwt_list = []
for chan in range(sigs.shape[1]):
    cwt_coeff, _ = pywt.cwt(sigs[:, chan], scales, 'cmor2.0-2.0', sampling_period=1/sample_rate)
    
    # Use only the real part of the CWT coefficients and reverse their order for plotting
    cwt_real = np.abs(cwt_coeff)
    
    logpowers_m = np.log10(cwt_real)**2
    
    cwt_list.append(logpowers_m)


# Convert list of 2D arrays into a 3D array (freqs x time x channels)
cwt_array = np.stack(cwt_list, axis=-1)

# Ensure the shape matches expected (freqs x time x channels)
assert cwt_array.shape == (len(scales), sigs.shape[0], sigs.shape[1]), "CWT shape mismatch"

# Create the main window that can contain several viewers
win = MainViewer(debug=True, show_auto_scale=True)

# Create a data source for the TraceViewer
source1 = InMemoryAnalogSignalSource(sigs, sample_rate, t_start)

# Create a TraceViewer for the time-domain signal
view1 = TraceViewer(source=source1, name='trace')
view1.params['scale_mode'] = 'same_for_all'
view1.auto_scale()

# Standard TimeFreqViewer instantiation with the time-domain source
view2 = TimeFreqViewer(source=source1, name='tfr - time-domain')
view2.params['show_axis'] = True
view2.params['timefreq', 'deltafreq'] = 1
view2.params['timefreq', 'normalisation'] = -1
view2.by_channel_params['ch0', 'visible'] = True

# Create a TimeFreqViewer using the from_numpy method with precomputed CWT spectrogram
view3 = TimeFreqViewer.from_numpy(
    sigs=cwt_array,
    sample_rate=sample_rate,
    t_start=t_start,
    name='tfr - pre',
    frequencies=frequencies,
)

# Add both viewers to the main window
win.add_view(view1)
win.add_view(view2)
win.add_view(view3)

# Show main window and run Qapp
win.show()
app.exec_()
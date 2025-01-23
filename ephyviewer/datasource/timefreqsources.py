# -*- coding: utf-8 -*-
#~ from __future__ import (unicode_literals, print_function, division, absolute_import)

import numpy as np
from .sourcebase import BaseDataSource


class BaseTimeFreqSource(BaseDataSource):
    type = 'TimeFreq'
    
    def __init__(self):
        super().__init__()
    
    def get_freq_length(self):
        raise(NotImplementedError)
        
    def get_sample_length(self):
        raise(NotImplementedError)
    
    def get_shape(self):
        return (self.get_freq_length(), self.get_sample_length(), self.nb_channel)
    
    def get_chunk(self, i_start=None, i_stop=None):
        raise(NotImplementedError)

    def time_to_index(self, t):
        return int((t-self.t_start)*self.sample_rate)

    def index_to_time(self, ind):
        return float(ind/self.sample_rate) + self.t_start
    


class InMemoryTimeFreqSource(BaseTimeFreqSource):
    def __init__(self, spectrogram, frequencies, sample_rate, t_start, channel_names=None):
        super().__init__()

        # If only one channel, expand dims to make it 3D (freq x time x channels)
        if spectrogram.ndim == 2:
            spectrogram = spectrogram[:, :, np.newaxis]
            
        # Validate that the number of frequency bins matches the frequency length
        if spectrogram.shape[0] != len(frequencies):
            raise ValueError("The first dimension of the spectrogram must match the length of the frequencies array.")

        self.spectrogram = spectrogram
        self._frequencies = frequencies
        self.sample_rate = float(sample_rate)
        self._t_start = float(t_start)
        self._t_stop = self.spectrogram.shape[1] / self.sample_rate + float(t_start)
        self.channel_names = channel_names

        if channel_names is None:
            self.channel_names = ['Channel {:3}'.format(c) for c in range(self.spectrogram.shape[2])]

    @property
    def nb_channel(self):
        return self.spectrogram.shape[2]
 
    def get_channel_name(self, chan=0):
        return self.channel_names[chan]

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop

    @property
    def frequencies(self):
        return self._frequencies
    
    def get_freq_length(self):
        return self.spectrogram.shape[0]
    
    def get_sample_length(self):
        return self.spectrogram.shape[1]
    
    def get_chunk(self, i_start=None, i_stop=None):
        return self.spectrogram[:, i_start:i_stop, :]


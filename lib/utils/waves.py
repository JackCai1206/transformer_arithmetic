import string
import numpy as np

class WaveGenerator():
    def __init__(self, n_components=10, length=256, n_levels=10, chars='0123456789'):
        self.n_components = n_components
        self.length = length
        self.n_levels = n_levels
        self.chars = chars
        self.frequencies = np.random.standard_normal(n_components)
        self.weights = np.random.standard_normal(n_components)
        self.phases = np.random.standard_normal(n_components)
        self.c = 0

    def sample_wave(self):
        wave = np.zeros(self.length)
        phases_offset = self.c
        self.c += 1
        for i in range(self.n_components):
            wave += self.weights[i] * np.sin(2 * np.pi * self.frequencies[i] * np.arange(self.length) + self.phases[i]+phases_offset)
        return wave

    def sample(self):
        wave = self.sample_wave()
        min_amp = np.min(wave)
        max_amp = np.max(wave)
        amp_range = max_amp - min_amp
        step_size = amp_range / (self.n_levels - 1)
        quantized_amplitude = np.round((wave - min_amp) / step_size).astype(int)
        string = ''.join([self.chars[level] for level in quantized_amplitude])
        return string, wave


class WaveGenerator2():
    def __init__(self, periods=[2, 3], length=256, chars='$' + string.digits):
        self.n_components = len(periods)
        self.length = length
        self.chars = chars
        self.chars.replace('\n', '')
        # self.periods = np.sort(np.random.choice(np.arange(1, length//2), n_components, replace=False))
        self.periods = periods
        # self.phases = np.random.choice(np.arange(1, length//2), n_components, replace=True)
        
    def sample(self):
        string = np.zeros(self.length, dtype=int)
        char_idx = np.arange(1, len(self.chars))
        np.random.shuffle(char_idx)
        for i in range(self.n_components):
            phase = np.random.choice(np.arange(0, self.periods[i]))
            tgt_idx = string[phase::self.periods[i]]
            tgt_idx[char_idx[i] > tgt_idx] = char_idx[i]
        return ''.join([self.chars[level] for level in string])

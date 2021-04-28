import numpy as np
import spikeextractors as se

# ---------------------------------------------------------------- #
def add_artificial_spikes(X):
    spike_frame_channel_array = []

    for i in range(0, len(X)):
        frame = np.random.randint(10, len(X[0])-10)
        spike_frame_channel_array.append([frame, i])
        X[i][frame]-= 2000

    spike_frame_channel_array = sorted(spike_frame_channel_array)

    return X, spike_frame_channel_array

# ---------------------------------------------------------------- #
def create_simulated_recording(size, num_frames = 1000, sampling_frequency = 30000, seed = 0):
    #TODO if centered at 0, 0: two channels at pos 0 if even number
    # channel_pos = [int(coord-(size-1)/2) for coord in range(0, size)]
    channel_pos = [coord for coord in range(0, size)]
    geom = []
    for k in channel_pos:
        for j in channel_pos:
            geom.append([j, k, 0])

    geom = np.asarray(geom)
    channel_ids = np.arange(0, size*size)
    num_channels = len(channel_ids)

    X = np.random.RandomState(seed=seed).normal(0, 1, (num_channels, num_frames))
    X = (X * 100).astype(int)
    X, spike_frame_channel_array = add_artificial_spikes(X)

    RX = se.NumpyRecordingExtractor(timeseries=X, sampling_frequency=sampling_frequency, geom=geom)

    return geom, RX, spike_frame_channel_array

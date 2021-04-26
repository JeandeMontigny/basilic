import spikeextractors as se

import numpy as np

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
def create_simulated_mea_rec(size, num_frames = 1000, sampling_frequency = 30000, seed = 0):
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

# ---------------------------------------------------------------- #
def make_padded_channels(geom, num_pads = 1):
    sorted_xs = np.unique(np.sort(geom[:,0]))
    sorted_ys = np.unique(np.sort(geom[:,1]))
    # sorted_zs = np.unique(np.sort(geom[:,2]))

    padded_channels_list = []
    observed_channels_list = []
    corresponding_channels_ids = []

    recording_channel_id = -1
    padded_channel_id = -1
    for i in range(min(sorted_xs)-len(sorted_xs)*num_pads, max(sorted_xs)+len(sorted_xs)*num_pads+1):
        for j in range(min(sorted_ys)-len(sorted_ys)*num_pads, max(sorted_ys)+len(sorted_ys)*num_pads+1):
            # for k in range(-num_pads, num_pads+1):
            padded_channel_id+=1
            padded_channels_list.append([i, j, 0])
            if (i in sorted_xs) and (j in sorted_ys):
                recording_channel_id+=1
                observed_channels_list.append(1)
                corresponding_channels_ids.append([recording_channel_id, padded_channel_id])
            else:
                observed_channels_list.append(0)

    padded_channels = np.asarray(padded_channels_list)
    observed_channels = np.asarray(observed_channels_list)

    return padded_channels, observed_channels, corresponding_channels_ids

# ---------------------------------------------------------------- #
def get_neighbours_channels(padded_channels, observed_channels, spike_channel_radius = 40):
    from collections import defaultdict

    neighbouring_channels = defaultdict(list)
    recording_channels_ids = []

    for i in range(0, len(padded_channels)):
        if (observed_channels[i] == 1):
            recording_channels_ids.append(i)

    all_channels_ids = np.arange(0, len(padded_channels))

    for i, channel in enumerate(recording_channels_ids):
        channel_ids_copy = np.copy(all_channels_ids)
        closest_channels = np.asarray(sorted(all_channels_ids, key=lambda channel_id: np.linalg.norm(padded_channels[channel_id] - padded_channels[channel])))
        for close_channel in closest_channels:
            if np.linalg.norm(padded_channels[close_channel] - padded_channels[channel]) < spike_channel_radius:
                neighbouring_channels[channel].append(close_channel)

    return neighbouring_channels

# ---------------------------------------------------------------- #
def extract_waveforms(recording, spike_frame_channel_array, padded_channels, neighbouring_channels,
                      corresponding_channels_ids, snippet_len=60):
    waveforms_list = []
    channel_ids_list = []
    for spikes in spike_frame_channel_array:
        waveforms = []
        channel_ids = []
        spike_times = spikes[0]
        spike_channel_id = spikes[1]
        padded_channel_id = corresponding_channels_ids[spike_channel_id][1]
        real_padded_channel_ids = [id[1] for id in corresponding_channels_ids]

        for channel_id in neighbouring_channels.get(padded_channel_id):
            channel_ids.append(channel_id)
            # if fake channel
            if channel_id not in real_padded_channel_ids:
                waveforms.append(np.zeros(snippet_len))
            else:
                rec_id = [id[0] for id in corresponding_channels_ids if id[1] == channel_id]
                waveforms.append(recording.get_traces(rec_id[0])[0][int(spike_times-(snippet_len/2)):int(spike_times+(snippet_len/2))])

        waveforms_list.append(waveforms)
        channel_ids_list.append(channel_ids)

    return waveforms_list, channel_ids_list

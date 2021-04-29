import spikeextractors as se

import numpy as np

# ---------------------------------------------------------------- #
def get_recording_data(recording_file):
    recording = se.MEArecRecordingExtractor(recording_file)
    sorting = se.MEArecSortingExtractor(recording_file)
    geom = np.asarray(recording.get_channel_locations())

    spike_times = []
    for  unit_id in sorting.get_unit_ids():
        spike_train = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_times.extend(spike_train)
    spike_times = sorted(spike_times)

    spike_frame_channel_array = []
    for i, spike_time in enumerate(spike_times):
        if i % (int(len(spike_times)/5)) == 0:
            print(int(float(i)/len(spike_times)*100), '%', end="\r")
        snippets = np.squeeze(recording.get_snippets(channel_ids=None, reference_frames=[spike_time], snippet_len=10),0)
        min_channel_id = np.argmin(np.min(snippets, 1))
        spike_frame_channel_array.append([spike_time, min_channel_id])
    print("100 %", end="\r")

    return geom, recording, spike_frame_channel_array

# ---------------------------------------------------------------- #
def make_padded_channels(geom, num_pads = 1):
    sorted_xs = np.unique(np.sort(geom[:,0]))
    buffer_xs = sorted_xs[-1] + (-sorted_xs[0]) + np.unique(np.diff(sorted_xs))[0]
    sorted_ys = np.unique(np.sort(geom[:,1]))
    buffer_ys = sorted_ys[-1] + (-sorted_ys[0]) + np.unique(np.diff(sorted_ys))[0]

    # sorted_zs = np.unique(np.sort(geom[:,2]))
    # buffer_zs = sorted_zs[-1] + (-sorted_zs[0]) + np.unique(np.diff(sorted_zs))[0]
    padded_channels_list = list(geom)
    observed_channels_list = list(np.ones(len(geom)))
    for i in range(-num_pads, num_pads+1):
        for j in range(-num_pads, num_pads+1):
            if (i,j) != (0,0):
                buffer_channel_x = geom[:,0] + buffer_xs*i
                buffer_channel_y = geom[:,1] + buffer_ys*j
                geom_copy = np.copy(geom)
                geom_copy[:,0] = buffer_channel_x
                geom_copy[:,1] = buffer_channel_y
                padded_channels_list = padded_channels_list + list(geom_copy)
                observed_channels_list = observed_channels_list + list(np.zeros(len(geom)))
    padded_channels = np.asarray(padded_channels_list)
    observed_channels = np.asarray(observed_channels_list)
    return padded_channels, observed_channels

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
def extract_waveforms(recording, spike_frame_channel_array, padded_channels, observed_channels,
                      neighbouring_channels, wf_frames_before_spike=30, wf_frames_after_spike=30):
    waveforms_list = []
    channel_ids_list = []
    for spikes in spike_frame_channel_array:
        waveforms = []
        channel_ids = []
        spike_time = spikes[0]
        spike_channel_id = spikes[1]
        real_neighbour_channel_ids = [channel_id for channel_id in neighbouring_channels.get(spike_channel_id) if observed_channels[channel_id] == 1]
        real_waveforms = recording.get_snippets(reference_frames=[spike_time],
                                                snippet_len=(wf_frames_before_spike, wf_frames_after_spike),
                                                channel_ids=real_neighbour_channel_ids)[0]
        real_channels_appended = 0
        for channel_id in neighbouring_channels.get(spike_channel_id):
            channel_ids.append(channel_id)
            # if fake channel
            if observed_channels[channel_id] == 0:
                waveforms.append(np.zeros(wf_frames_before_spike+wf_frames_after_spike))
            else:
                waveforms.append(real_waveforms[real_channels_appended])
                real_channels_appended += 1

        waveforms_list.append(waveforms)
        channel_ids_list.append(channel_ids)

    return waveforms_list, channel_ids_list

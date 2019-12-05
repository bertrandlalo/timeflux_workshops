# -*- coding: utf-8 -*-
"""EEG utilities
    Created on december 2018
    @authors: Raphaëlle Bertrand-Lalo, David Ojeda

    Module containing utils functions to load, convert, process and plot EEG data.
"""

import logging

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

logging.getLogger("matplotlib").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import mne
import seaborn as sns
import numpy as np
import pandas as pd

import yaml


def estimate_rate(data):
    """ Estimate nominal sampling rate of a DataFrame.
    This function checks if the index are correct, that is monotonic and regular
    (the jitter should not exceed twice the nominal timespan)
    Notes
    -----
    This function does not take care of jitters in the Index and consider that the rate as the 1/Ts
    where Ts is the average timespan between samples.
    Parameters
    ----------
    data: pd.DataFrame
        DataFrame with index corresponding to timestamp (either DatetimeIndex or floats)
    Returns
    -------
    rate: nominal rate of the DataFrame
    """
    # check that the index is monotonic
    # if not data.index.is_monotonic:
    # raise Exception('Data index should be monotonic')
    if data.shape[0] < 2:
        raise Exception('Sampling rate requires at least 2 points')

    if isinstance(data.index, (pd.TimedeltaIndex, pd.DatetimeIndex)):
        delta = data.index - data.index[0]
        index_diff = np.diff(delta) / np.timedelta64(1, 's')
    elif np.issubdtype(data.index, np.number):
        index_diff = np.diff(data.index)
    else:
        raise Exception('Dataframe index is not numeric')

    average_timespan = np.mean(index_diff)

    return 1 / average_timespan


def load_standalone_graph(path):
    # Load a graph
    with open(path, 'r') as stream:
        try:
            graph = yaml.safe_load(stream)['graphs'][0]
        except yaml.YAMLError as exc:
            print(exc)
    graph_standalone = graph.copy()
    # Get rid of online specific nodes and edges (zmq, lsl, safeguards)
    graph_standalone['nodes'] = [node for node in graph['nodes'] if
                                 node['module'] not in ['timeflux.nodes.lsl', 'timeflux.nodes.zmq']]
    nodes_id = [node['id'] for node in graph_standalone['nodes']]

    def keep_edge(edge):
        return (edge['source'].split(':')[0] in nodes_id) and (edge['target'].split(':')[0] in nodes_id)

    graph_standalone['edges'] = [edge for edge in graph_standalone['edges'] if keep_edge(edge)]
    return graph_standalone


import numpy as np
import mne


def pandas_to_mne(data, events=None, montage_kind='standard_1005', unit_factor=1e-6, bad_ch=[]):
    ''' Convert a pandas Dataframe into mne raw object 
    Parameters
    ----------
    data: Dataframe with index=timestamps, columns=eeg channels
    events: array, shape = (n_events, 3) with labels on the third axis. 
    unit_factor: unit factor to apply to get Voltage
    bad_ch: list of channels to reject 

    Returns
    -------
    raw: raw object
    '''
    n_chan = len(data.columns)

    X = data.copy().values
    times = data.index

    ch_names = list(data.columns)
    ch_types = ['eeg'] * n_chan
    montage = mne.channels.read_montage(montage_kind) if montage_kind is not None else None
    sfreq = estimate_rate(data)
    X *= unit_factor

    if events is not None:
        events_onsets = events.index
        events_labels = events.label.values
        event_id = {mk: (ii + 1) for ii, mk in enumerate(np.unique(events_labels))}
        ch_names += ['stim']
        ch_types += ['stim']

        trig = np.zeros((len(X), 1))
        for ii, m in enumerate(events_onsets):
            ix_tr = np.argmin(np.abs(times - m))
            trig[ix_tr] = event_id[events_labels[ii]]

        X = np.c_[X, trig]
    else:
        event_id = None

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq, montage=montage)
    info["bads"] = bad_ch
    raw = mne.io.RawArray(data=X.T, info=info, verbose=False)
    picks = mne.pick_channels(raw.ch_names, include=[], exclude=["stim"] + bad_ch)
    return raw

from copy import deepcopy
import torch


def fedavg_aggregate(client_states, client_sizes):

    global_state = deepcopy(client_states[0])

    total_samples = sum(client_sizes)

    for key in global_state.keys():

        global_state[key] = sum(
            client_states[i][key] *
            (client_sizes[i] / total_samples)
            for i in range(len(client_states))
        )

    return global_state
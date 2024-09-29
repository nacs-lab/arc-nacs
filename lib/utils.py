import numpy as np
import heapq
from matplotlib import pyplot as plt
#import arc

eVToGHz = 2.417989242 * 1e14 / 1e9
JToGHz = 1.509190179 * 1e33 / 1e9

def get_maximum_overlap_eigenstate(evals, evecs, basis_states, n1 = 0, n2 = 0, l1 = 0, l2 = 0, j1 = 0, j2 = 0, **kwargs):
    # first find basis state corresponding to the n1, n2, l1, l2, j1, j2
    if "state" in kwargs:
        state = kwargs["state"]
        n1 = state[0]
        l1 = state[1]
        j1 = state[2]
        n2 = state[4]
        l2 = state[5]
        j2 = state[6]
    main_idx = -1
    for idx, val in enumerate(basis_states):
        if val[0] == n1 and val[1] == l1 and val[2] == j1 and val[4] == n2 and val[5] == l2 and val[6] == j2:
            main_idx = idx
            break

    res_evals = []
    res_comps = []
    res_idxs = []
    for idx, evec in enumerate(evecs):
        comp = np.abs(evec[main_idx, :])**2
        this_idx = np.argmax(comp)
        res_evals.append(evals[idx][this_idx])
        res_comps.append(comp[this_idx])
        res_idxs.append(this_idx)

    return res_evals, res_comps, res_idxs

def find_highest_n_comps(evec, n, basis_states):
    comps = np.abs(evec)**2

    largest_with_indices = heapq.nlargest(n, enumerate(comps), key=lambda x: x[1])

    # Separate the indices and values
    indices, largest_elements = zip(*largest_with_indices)

    return [basis_states[idx] for idx in indices], largest_elements

def find_states_within(atom, n, l, j, m = 0.5, dn = 5, dl = 3, dE = 10, B = 0):
    # dE is in units of GHz
    # B is in units of G
    if B == 0:
        myE = atom.getEnergy(n, l, j) * eVToGHz
    else:
        myE = atom.getEnergy(n, l, j) * eVToGHz + atom.getZeemanEnergyShift(l, j, m, B * 1e-4) * JToGHz
    res = []
    for this_n in range(n - dn, n + dn + 1):
        for this_l in range(max(0, l - dl), l + dl + 1):
            if this_l >= this_n:
                continue
            if this_l == 0:
                j_vals = [0.5]
            else:
                j_vals = [this_l - 0.5, this_l + 0.5]
            for this_j in j_vals:
                if B == 0:
                    thisE = atom.getEnergy(this_n, this_l, this_j) * eVToGHz
                    if np.abs(myE - thisE) < dE:
                        res.append([this_n, this_l, this_j, 0, thisE - myE])
                else:
                    for this_m in np.linspace(-this_j, this_j, this_j * 2 + 1):
                        thisE = atom.getEnergy(this_n, this_l, this_j) * eVToGHz + atom.getZeemanEnergyShift(this_l, this_j, this_m, B * 1e-4) * JToGHz
                        if np.abs(myE - thisE) < dE:
                            res.append([this_n, this_l, this_j, this_m, thisE - myE])
    return res

def visualize_states(states, states2 = None, ignore_j = True):
    plt.figure()
    for state in states:
        n = state[0]
        l = state[1]
        j = state[2]
        m = state[3]
        energy = state[4]
        plt.hlines(energy, xmin = l, xmax = l + 0.7, color = 'blue')
        if ignore_j:
            plt.text(l + 0.7, energy, str(n))
        else:
            plt.text(l + 0.7, energy, str(n) + ',' + str(j))
    if states2 is not None:
        for state in states2:
            n = state[0]
            l = state[1]
            j = state[2]
            m = state[3]
            energy = state[4]
            plt.hlines(energy, xmin = -l - 0.7, xmax = -l, color = 'orange')
            if ignore_j:
                plt.text(-l - 1, energy, str(n))
            else:
                plt.text(-l - 1, energy, str(n) + ',' + str(j))
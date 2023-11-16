
import numpy as np
# load GraphINVENT-specific functions
from Parameters import Parameters as C




def get_feature_vector_indices():

    idc = [C.n_atom_types, C.n_formal_charge]

    # indices corresponding to implicit H's and chirality are optional (below)
    if not C.use_explicit_H and not C.ignore_H:
        idc.append(C.n_imp_H)

    if C.use_chirality:
        idc.append(C.n_chirality)

    return np.cumsum(idc).tolist()


def one_of_k_encoding(x, allowable_set):

    if x not in set(allowable_set):  # use set for speedup over list
        raise Exception(
            f"Input {x} not in allowable set {allowable_set}. "
            f"Add {x} to allowable set in `features.py` and run again."
        )

    one_hot_generator = (int(x == s) for s in allowable_set)

    return one_hot_generator

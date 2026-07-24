import numpy as np
import mne


def r_value(x1, x2):
    """
    returns r value.
    x1 - x2

    Parameters
    ==========
    x1 : array-like, shape of (n_epochs, n_ch, n_samples)
        epoch data of class 1, should be target
    x2 : array-like, shape of (n_epochs, n_ch, n_samples)
        epoch data of class 2, should be non-target
    """

    x1 = np.array(x1)
    x2 = np.array(x2)

    N1 = x1.shape[0]
    N2 = x2.shape[0]

    X = np.append(x1, x2, axis=0)

    r = np.mean(x1, axis=0) - np.mean(x2, axis=0)
    r = r / np.std(X, axis=0, ddof=1)
    r = r * (np.sqrt(N1 * N2) / (N1 + N2))

    return r


def signed_square_r(x1, x2):
    """
    returns signed square r value.
    x1 - x2

    Parameters
    ==========
    x1 : array-like, shape of (n_epochs, n_ch, n_samples)
        epoch data of class 1, should be target
    x2 : array-like, shape of (n_epochs, n_ch, n_samples)
        epoch data of class 2, should be non-target
    """

    r = r_value(x1, x2)
    signed_r2 = r * np.absolute(r)

    return signed_r2


def evoked_signed_square_r(epochs1, epochs2):
    data_1 = epochs1.get_data(units="uV")
    data_2 = epochs2.get_data(units="uV")

    info = epochs1.info
    tmin = epochs1.tmin
    baseline = epochs1.baseline

    signed_r2 = signed_square_r(data_1, data_2)

    signed_r2_evoked = mne.EvokedArray(signed_r2, info=info, tmin=tmin)

    return signed_r2_evoked

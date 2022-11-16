import numpy as np


# pre-process data
def pre_process(X_train, y_train, X_val, y_val, X_test, y_test):
    ## train set

    # remove pressure offset
    for i in range(len(y_train)):
        y_train[i, 0, :, :] -= np.mean(y_train[i, 0, :, :])  # remove offset
        y_train[i, 0, :, :] -= y_train[i, 0, :, :] * X_train[i, 2, :, :]  # pressure * mask

    # make dimensionless
    for i in range(len(y_train)):
        v_norm = (np.max(np.abs(X_train[i, 0, :, :])) ** 2 + np.max(np.abs(X_train[i, 1, :, :])) ** 2) ** 0.5
        y_train[i, 0, :, :] /= v_norm ** 2
        y_train[i, 1, :, :] /= v_norm
        y_train[i, 2, :, :] /= v_norm

    # normalize to -1...1 range
    max_inputs_0 = np.max(np.abs(X_train[:, 0, :, :]))
    max_inputs_1 = np.max(np.abs(X_train[:, 1, :, :]))

    max_targets_0 = np.max(np.abs(y_train[:, 0, :, :]))
    max_targets_1 = np.max(np.abs(y_train[:, 1, :, :]))
    max_targets_2 = np.max(np.abs(y_train[:, 2, :, :]))

    X_train[:, 0, :, :] *= (1.0 / max_inputs_0)
    X_train[:, 1, :, :] *= (1.0 / max_inputs_1)

    y_train[:, 0, :, :] *= (1.0 / max_targets_0)
    y_train[:, 1, :, :] *= (1.0 / max_targets_1)
    y_train[:, 2, :, :] *= (1.0 / max_targets_2)

    ## val and test set

    # remove pressure offset
    for i in range(len(y_val)):
        y_val[i, 0, :, :] -= np.mean(y_val[i, 0, :, :])  # remove offset
        y_val[i, 0, :, :] -= y_val[i, 0, :, :] * X_val[i, 2, :, :]  # pressure * mask

    for i in range(len(y_test)):
        y_test[i, 0, :, :] -= np.mean(y_test[i, 0, :, :])  # remove offset
        y_test[i, 0, :, :] -= y_test[i, 0, :, :] * X_test[i, 2, :, :]  # pressure * mask

    # make dimensionless
    for i in range(len(y_val)):
        v_norm = (np.max(np.abs(X_val[i, 0, :, :])) ** 2 + np.max(np.abs(X_val[i, 1, :, :])) ** 2) ** 0.5
        y_val[i, 0, :, :] /= v_norm ** 2
        y_val[i, 1, :, :] /= v_norm
        y_val[i, 2, :, :] /= v_norm

    for i in range(len(y_test)):
        v_norm = (np.max(np.abs(X_test[i, 0, :, :])) ** 2 + np.max(np.abs(X_test[i, 1, :, :])) ** 2) ** 0.5
        y_test[i, 0, :, :] /= v_norm ** 2
        y_test[i, 1, :, :] /= v_norm
        y_test[i, 2, :, :] /= v_norm

    # normalize with max values from train set
    X_val[:, 0, :, :] *= (1.0 / max_inputs_0)
    X_val[:, 1, :, :] *= (1.0 / max_inputs_1)

    y_val[:, 0, :, :] *= (1.0 / max_targets_0)
    y_val[:, 1, :, :] *= (1.0 / max_targets_1)
    y_val[:, 2, :, :] *= (1.0 / max_targets_2)

    X_test[:, 0, :, :] *= (1.0 / max_inputs_0)
    X_test[:, 1, :, :] *= (1.0 / max_inputs_1)

    y_test[:, 0, :, :] *= (1.0 / max_targets_0)
    y_test[:, 1, :, :] *= (1.0 / max_targets_1)
    y_test[:, 2, :, :] *= (1.0 / max_targets_2)

    return X_train, y_train, X_val, y_val, X_test, y_test

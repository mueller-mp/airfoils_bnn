'''
script that runs training and validation
'''
import click
from tensorflow.keras.losses import mae
import math
import os.path
import matplotlib.pyplot as plt
from matplotlib import cm
from BNN_functional import Bayes_DfpNet, bayesian_mars_moon, extend_to_val
from data import pre_process
from tensorflow.keras.optimizers import Adam
import numpy as np
from os import listdir
import random
import tensorflow as tf
import pandas as pd


# parameters obtained from commandline
@click.command()
@click.option("--datadir", default = "../data/")
@click.option("--model_type", default = 'bayesian-unet', help = 'bayesian-mars-moon or bayesian-unet')
@click.option("--batch_size", default = 64, type = click.INT)
@click.option("--lrg", default = 0.005, type = click.FLOAT)
@click.option("--epochs", default = 40, type = click.INT)
@click.option("--kl_pref", default = 100, type = click.FLOAT)
@click.option("--expo", default = 3, type = click.FLOAT)
@click.option("--dropout", default = 0., type = click.FLOAT)
@click.option("--flipout", default = True, type = click.BOOL)
@click.option("--folder", default = '')
@click.option("--spatial_dropout", default = True, type = click.BOOL)
@click.option("--save_model", default = False, type = click.BOOL)
@click.option("--seed", default = 405060, type = click.INT)
@click.option("--val_training", default = False, type = click.BOOL,
              help = 'BOOL indicating if train-flag should be used at inference time')
def main(datadir, model_type, batch_size, lrg, epochs, kl_pref, expo, dropout, flipout, folder, spatial_dropout,
         save_model, seed, val_training):
    # Initializations of Seeds
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # load train and val data
    files = listdir(datadir + 'train/')
    random.shuffle(files, )
    frac_train_test = 0.9
    split_idx = int(len(files) * frac_train_test)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    X_train = np.empty((len(train_files), 3, 128, 128))
    y_train = np.empty((len(train_files), 3, 128, 128))

    X_val = np.empty((len(val_files), 3, 128, 128))
    y_val = np.empty((len(val_files), 3, 128, 128))

    for i, file in enumerate(train_files):
        npfile = np.load(datadir + 'train/' + file)
        d = npfile['a']
        X_train[i] = d[0:3]
        y_train[i] = d[3:6]

    for i, file in enumerate(val_files):
        npfile = np.load(datadir + 'train/' + file)
        d = npfile['a']
        X_val[i] = d[0:3]
        y_val[i] = d[3:6]

    print("Number of train data loaded:", len(X_train))

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train), seed = 46168531,
                                                                             reshuffle_each_iteration = False).batch(
        batch_size, drop_remainder = False)

    # load test data
    test_files = listdir(datadir + 'test/')
    random.shuffle(test_files, )

    X_test = np.empty((len(test_files), 3, 128, 128))
    y_test = np.empty((len(test_files), 3, 128, 128))

    for i, file in enumerate(test_files):
        npfile = np.load(datadir + 'test/' + file)
        d = npfile['a']
        X_test[i] = d[0:3]
        y_test[i] = d[3:6]

    print("Number of test data loaded:", len(X_test))

    # pre-process all data
    X_train, y_train, X_val, y_val, X_test, y_test = pre_process(X_train.copy(), y_train.copy(), X_val.copy(),
                                                                 y_val.copy(), X_test.copy(), y_test.copy())

    # move axis to channels_last for convenience
    X_train = np.moveaxis(X_train, 1, -1)
    X_val = np.moveaxis(X_val, 1, -1)
    y_train = np.moveaxis(y_train, 1, -1)
    y_val = np.moveaxis(y_val, 1, -1)
    X_test = np.moveaxis(X_test, 1, -1)
    y_test = np.moveaxis(y_test, 1, -1)

    def computeLR(i, epochs, minLR, maxLR):
        if i < epochs * 0.5:
            return maxLR
        e = (i / float(epochs) - 0.5) * 2.
        # rescale second half to min/max range
        fmin = 0.
        fmax = 6.
        e = fmin + e * (fmax - fmin)
        f = math.pow(0.5, e)
        return minLR + (maxLR - minLR) * f

    if model_type == 'bayesian-unet':
        model = Bayes_DfpNet(expo = expo, flipout = flipout, dropout = dropout,
                             kl_scaling = kl_pref * len(X_train) / batch_size, spatial_dropout = spatial_dropout,
                             extend_to_val = val_training)
    elif model_type == 'bayesian-mars-moon':
        model = bayesian_mars_moon(in_shape = (128, 128, 3), out_filters = 3, reg_filters = 32, flipout = flipout,
                                   dropout = dropout, kl_scaling = kl_pref * len(X_train) / batch_size, bn = True,
                                   spatial_dropout = spatial_dropout)
    else:
        raise Exception('model_type has to be either bayesian-mars-moon or bayesian-unet, but is {}'.format(model_type))
    optimizer = Adam(learning_rate = lrg, beta_1 = 0.5, beta_2 = 0.9999)

    num_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    print('Our model has {} parameters.'.format(num_params))

    kl_losses = []
    mae_losses = []
    total_losses = []
    val_maes = []
    val_stds = []
    idxs = np.array(range(len(X_train)))
    np.random.shuffle(idxs)
    idxs = np.resize(idxs, (int(len(X_train) / batch_size) + 1, batch_size))
    stop = False
    epoch = 0
    while not stop:
        currLr = computeLR(epoch, epochs, 0.5 * lrg, lrg)
        if currLr < lrg:
            tf.keras.backend.set_value(optimizer.lr, currLr)

        # iterate through training data
        kl_sum = 0
        mae_sum = 0
        total_sum = 0
        for batch_idxs in idxs:
            # forward pass and loss computation
            with tf.GradientTape() as tape:
                # inputs, targets = traindata
                inputs, targets = X_train[batch_idxs, ...], y_train[batch_idxs, ...]
                prediction = model(inputs, training = True)
                loss_mae = tf.reduce_mean(mae(prediction, targets))
                kl = sum(model.losses)
                loss_value = kl + tf.cast(loss_mae, dtype = 'float32')
            # backpropagate gradients and update parameters
            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # store losses per batch
            kl_sum += kl
            mae_sum += tf.reduce_mean(loss_mae)
            total_sum += tf.reduce_mean(loss_value)

        # store losses per epoch
        kl_losses += [kl_sum / len(dataset)]
        mae_losses += [mae_sum.numpy() / len(dataset)]
        total_losses += [total_sum.numpy() / len(dataset)]

        # validation
        outputs = model(X_val, training = False)
        val_maes += [tf.reduce_mean(mae(y_val, outputs)).numpy()]

        # if epoch < 3 or epoch % 20 == 0:
        print('Finnished epoch {}...'.format(epoch))
        print('total loss: {}'.format(total_losses[-1]))
        print('KL loss: {}'.format(kl_losses[-1]))
        print('MAE loss: {}'.format(mae_losses[-1]))
        print('Validation MAE loss: {}'.format(val_maes[-1]))
        print('----------------------------')

        epoch += 1
        # stop if max number of epochs is reached
        if epoch + 1 == epochs:
            stop = True

    for val_training in [True, False]:
        savedir = "../runs/" + folder + "{}_bsize_{}_lrG_{}_epochs_{}_klpref_{}_spatialDropout_{}_dropout_{}_flipout_{}_valtrainin_{}/".format(
            model_type, batch_size, lrg, epochs, kl_pref, spatial_dropout, dropout, flipout, val_training)

        if not os.path.exists(savedir):
            os.makedirs(savedir)

        # turn dropout on / off
        extend_to_val(model, val_training)

        # validation
        outputs = model(X_val, training = False)
        val_maes += [tf.reduce_mean(mae(y_val, outputs)).numpy()]

        if save_model:
            model.save(savedir + 'model_saved')

        fig, axs = plt.subplots(ncols = 3, nrows = 1, figsize = (20, 4))
        axs[0].plot(kl_losses, color = 'red')
        axs[0].set_title('KL Loss (Train)')
        axs[1].plot(mae_losses, color = 'blue', label = 'train')
        axs[1].plot(val_maes, color = 'green', label = 'val')
        axs[1].legend()
        axs[1].set_title('MAE Loss')
        axs[2].plot(total_losses, label = 'Total', color = 'black')
        axs[2].plot(kl_losses, label = 'KL', color = 'red')
        axs[2].plot(mae_losses, label = 'MAE', color = 'blue')
        axs[2].set_title('Total Train Loss')
        axs[2].legend()
        plt.savefig(savedir + 'Loss', bbox_inches = 'tight')

        # test
        reps = 20
        preds = np.zeros(shape = (reps,) + X_test.shape)
        for rep in range(reps):
            preds[rep, :, :, :, :] = model(X_test, training = False)
        preds_mean = np.mean(preds, axis = 0)
        preds_std = np.std(preds, axis = 0)

        test_mae_mean = tf.reduce_mean(mae(preds_mean, y_test))
        test_std_mean = tf.reduce_mean(preds_std)

        # store summary statistics in df and append to old df if exists
        df = pd.DataFrame({'model_type': [model_type], 'epochs': [epochs], 'early_stopping_epoch': [epoch],
                           'batch_size': [batch_size], 'lrG': [lrg], 'flipout': [flipout], 'kl_pref': [kl_pref],
                           'dropout': [dropout], 'spatial_dropout': [spatial_dropout], 'val_mae': [val_maes[-1]],
                           'train_mae': [mae_losses[-1]], 'train_loss': [total_losses[-1]],
                           'train_losses_total': [total_losses], 'train_losses_mae': [mae_losses],
                           'val_losses_total': [val_maes], 'test_mae_mean': [test_mae_mean],
                           'test_std_mean': [test_std_mean], 'val_train': [val_training]})

        if os.path.isfile("../runs/" + folder + "df_summary.pkl"):
            df_old = pd.read_pickle("../runs/" + folder + "df_summary.pkl")
            df_new = pd.concat([df_old, df], ignore_index = True)
            df_new.to_pickle("../runs/" + folder + "df_summary.pkl")
        else:
            df.to_pickle("../runs/" + folder + "df_summary.pkl")

        np.savez_compressed(savedir + 'mae_std_per_sample', mae = np.mean(np.abs(preds_mean - y_test), axis = (1, 2)),
                            std = np.mean(preds_std, axis = (1, 2)),
                            rel_error = np.divide(np.mean(np.abs(preds_mean - y_test), axis = (1, 2)),
                                                  np.mean(np.abs(y_test), axis = (1, 2))))

        # plot MAE vs Std Dev per sample
        fig = plt.figure(figsize = (6, 4))
        plt.scatter(np.mean(np.abs(preds_mean - y_test), axis = (1, 2))[:, 0], np.mean(preds_std, axis = (1, 2))[:, 0],
                    c = 'black')
        plt.xlabel('Mean MAE')
        plt.ylabel('Std. Dev. MAE')
        plt.savefig(savedir + 'mae_vs_std_per_sample.png', bbox_inches = 'tight')

        # plot repeated samples from posterior for some observations
        def plot_BNN_predictions(target, preds, pred_mean, pred_std, num_preds=5, channel=0):
            if num_preds > len(preds):
                print(
                    'num_preds was set to {}, but has to be smaller than the length of preds. Setting it to {}'.format(
                        num_preds, len(preds)))
                num_preds = len(preds)

            # transpose and concatenate the frames that are to plot
            to_plot = np.concatenate((target[:, :, channel].transpose().reshape(128, 128, 1),
                                      preds[0:num_preds, :, :, channel].transpose(),
                                      pred_mean[:, :, channel].transpose().reshape(128, 128, 1),
                                      pred_std[:, :, channel].transpose().reshape(128, 128, 1)), axis = -1)
            fig, axs = plt.subplots(nrows = 1, ncols = to_plot.shape[-1], figsize = (20, 4))
            for i in range(to_plot.shape[-1]):
                label = 'Target' if i == 0 else ('Avg Pred' if i == (num_preds + 1) else (
                    'Std Dev (normalized)' if i == (num_preds + 2) else 'Pred {}'.format(i)))
                cmap = cm.viridis if i == (num_preds + 2) else cm.magma
                frame = np.flipud(to_plot[:, :, i])
                min = np.min(frame);
                max = np.max(frame)
                frame -= min;
                frame /= (max - min)
                axs[i].imshow(frame, cmap = cmap)
                axs[i].axis('off')
                axs[i].set_title(label)

        # plot repeated samples
        for obs_idx in [5, 15, 25, 31, 32]:
            plot_BNN_predictions(y_test[obs_idx, ...], preds[:, obs_idx, :, :, :], preds_mean[obs_idx, ...],
                                 preds_std[obs_idx, ...])
            plt.savefig(savedir + 'Sample_idx_{}.png'.format(obs_idx), bbox_inches = 'tight')

        IDXS = [1, 3, 8, 31, 32]
        CHANNEL = 0
        fig, axs = plt.subplots(nrows = len(IDXS), ncols = 3, sharex = True, sharey = True,
                                figsize = (9, len(IDXS) * 3))
        for i, idx in enumerate(IDXS):
            axs[i][0].imshow(np.flipud(X_test[idx, :, :, CHANNEL].transpose()), cmap = cm.magma)
            axs[i][1].imshow(np.flipud(preds_mean[idx, :, :, CHANNEL].transpose()), cmap = cm.magma)
            axs[i][2].imshow(np.flipud(preds_std[idx, :, :, CHANNEL].transpose()), cmap = cm.viridis)
            axs[i][0].axis('off')
            axs[i][1].axis('off')
            axs[i][2].axis('off')
        axs[0][0].set_title('Shape', size = 40)
        axs[0][1].set_title('Avg. Pred', size = 40)
        axs[0][2].set_title('Std. Dev', size = 40)
        plt.tight_layout()
        plt.savefig(savedir + 'different_shapes.png', bbox_inches = 'tight')

        IDXS = [1, 3, 8, 31, 32]
        CHANNEL = 0
        fig, axs = plt.subplots(nrows = len(IDXS), ncols = 4, sharex = True, sharey = True,
                                figsize = (12, len(IDXS) * 3))
        for i, idx in enumerate(IDXS):
            axs[i][0].imshow(np.flipud(X_test[idx, :, :, CHANNEL].transpose()), cmap = cm.magma)
            axs[i][1].imshow(np.flipud(preds_mean[idx, :, :, CHANNEL].transpose()), cmap = cm.magma)
            axs[i][2].imshow(np.flipud(preds_std[idx, :, :, CHANNEL].transpose()), cmap = cm.viridis)
            axs[i][3].imshow(np.flipud(y_test[idx, :, :, CHANNEL].transpose()), cmap = cm.magma)
            axs[i][0].axis('off')
            axs[i][1].axis('off')
            axs[i][2].axis('off')
            axs[i][3].axis('off')
        axs[0][0].set_title('Shape', size = 35)
        axs[0][1].set_title('Avg Pred', size = 35)
        axs[0][2].set_title('Std. Dev', size = 35)
        axs[0][3].set_title('Target', size = 35)
        plt.savefig(savedir + 'different_shapes_w_target.png', bbox_inches = 'tight')

        IDXS = [1, 3, 8, 31, 32]
        CHANNEL = 0
        fig, axs = plt.subplots(nrows = len(IDXS), ncols = 4, sharex = True, sharey = True,
                                figsize = (12, len(IDXS) * 3))
        for i, idx in enumerate(IDXS):
            axs[i][0].imshow(np.flipud(X_test[idx, :, :, CHANNEL].transpose()), cmap = cm.magma)
            axs[i][1].imshow(np.flipud(preds_mean[idx, :, :, CHANNEL].transpose()), cmap = cm.magma)
            axs[i][2].imshow(np.flipud(preds_std[idx, :, :, CHANNEL].transpose()), cmap = cm.viridis)
            axs[i][3].imshow(np.flipud(y_test[idx, :, :, CHANNEL].transpose()), cmap = cm.magma)
            axs[i][0].axis('off')
            axs[i][1].axis('off')
            axs[i][2].axis('off')
            axs[i][3].axis('off')
        axs[0][0].set_title('Shape', size = 45)
        axs[0][1].set_title('Avg Pred', size = 45)
        axs[0][2].set_title('Std. Dev', size = 45)
        axs[0][3].set_title('Target', size = 45)
        plt.tight_layout()
        plt.savefig(savedir + 'different_shapes_w_target_tight.png', bbox_inches = 'tight')

        fig, axs = plt.subplots(nrows = 18, ncols = 15, figsize = (20, 20 * 18 / 15))
        fig.subplots_adjust(wspace = 0, hspace = 0)
        for k, (im1, im2, im3) in enumerate(zip(y_test, preds_mean, preds_std)):
            i = k // 15
            j = k % 15
            axs[3 * i, j].imshow(np.flipud(im1[:, :, 0].transpose()), cmap = cm.magma)
            axs[3 * i + 1, j].imshow(np.flipud(im2[:, :, 0].transpose()), cmap = cm.magma)
            axs[3 * i + 2, j].imshow(np.flipud(im3[:, :, 0].transpose()), cmap = cm.viridis)

            axs[3 * i, j].axvline(x = .0, ymin = 0.0, ymax = 1., linewidth = 4, color = 'k')
            axs[3 * i + 1, j].axvline(x = .0, ymin = 0.0, ymax = 1., linewidth = 4, color = 'k')
            axs[3 * i + 2, j].axvline(x = .0, ymin = 0.0, ymax = 1., linewidth = 4, color = 'k')

            axs[3 * i, j].axhline(y = 0., xmin = 0.0, xmax = 1., linewidth = 4, color = 'k')

            axs[3 * i, j].axis('off')
            axs[3 * i + 1, j].axis('off')
            axs[3 * i + 2, j].axis('off')

        plt.subplots_adjust(wspace = 0, hspace = 0)
        plt.savefig(savedir + 'all_test_samples.png', bbox_inches = 'tight')

        print('mean of y-test: {}'.format(np.mean(y_test)))


if __name__ == "__main__":
    main()

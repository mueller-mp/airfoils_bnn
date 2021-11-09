'''
script that runs training and valdiation
'''
import click
from tensorflow.keras.losses import mae
import math
import os.path
import matplotlib.pyplot as plt
from matplotlib import cm
from BNN_functional import Bayes_DfpNet, bayesian_mars_moon
from tensorflow.keras.optimizers import Adam
import numpy as np
from os import listdir
import random
#from random import shuffle
import tensorflow as tf
import pandas as pd

# parameters obtained from commandline
@click.command()
@click.option("--datadir", default="../data/")
@click.option("--model_type", default='bayesian-unet', help='bayesian-mars-moon or bayesian-unet')
@click.option("--batch_size", default=64, type=click.INT)
@click.option("--lrg", default=0.005, type=click.FLOAT)
@click.option("--epochs", default=40, type=click.INT)
@click.option("--kl_pref", default=100, type=click.FLOAT)
@click.option("--dropout", default=0., type=click.FLOAT)
@click.option("--flipout", default=True, type=click.BOOL)
@click.option("--folder", default='')
@click.option("--spatial_dropout", default=True,type=click.BOOL)
@click.option("--save_model", default=False,type=click.BOOL)
@click.option("--seed", default=405060, type=click.INT)
def main(datadir,
         model_type,
         batch_size,
         lrg,
         epochs,
         kl_pref,
         dropout,
         flipout,
         folder,
         spatial_dropout,
         save_model,
         seed,
):
    savedir = "../runs/"+folder+"{}_bsize_{}_lrG_{}_epochs_{}_klpref_{}_spatialDropout_{}_dropout_{}_flipout_{}/"\
        .format(model_type, batch_size, lrg, epochs, kl_pref, spatial_dropout, dropout, flipout)

    # use model in training mode only if flipout is not used
    if flipout:
        val_training = False
        if dropout > 0.:
            print('Both flipout and dropout used - using flipout layers for BNN, turning dropout off during validation')
    else:
        val_training = True

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Initializations of Seeds
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # load train and val data
    files = listdir(datadir+'train/')
    random.shuffle(files,)
    frac_train_test = 0.9
    split_idx = int(len(files)*frac_train_test)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    X_train  = np.empty((len(train_files), 3, 128, 128))
    y_train = np.empty((len(train_files), 3, 128, 128))

    X_val  = np.empty((len(val_files), 3, 128, 128))
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

    # move axis to channels_last for convenience
    X_train = np.moveaxis(X_train,1,-1)
    X_val = np.moveaxis(X_val,1,-1)
    y_train = np.moveaxis(y_train,1,-1)
    y_val = np.moveaxis(y_val,1,-1)

    print("Number of train data loaded:", len(X_train) )

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train),
        seed=46168531, reshuffle_each_iteration=False).batch(batch_size, drop_remainder=False)

    # load test data
    test_files = listdir(datadir + 'test/')
    random.shuffle(files_test, )

    X_test = np.empty((len(test_files), 3, 128, 128))
    y_test = np.empty((len(test_files), 3, 128, 128))

    for i, file in enumerate(test_files):
        npfile = np.load(datadir + 'test/' + file)
        d = npfile['a']
        X_test[i] = d[0:3]
        y_test[i] = d[3:6]

    # move axis to channels_last for convenience
    X_test = np.moveaxis(X_test, 1, -1)
    y_test = np.moveaxis(y_test, 1, -1)

    print("Number of test data loaded:", len(X_test))

    dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(len(X_test),
                                                                             seed=46168531,
                                                                             reshuffle_each_iteration=False).batch(
        batch_size, drop_remainder=False)

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

    if model_type=='bayesian-unet':
        model=Bayes_DfpNet(expo=3,flipout=flipout, dropout=dropout, kl_scaling=kl_pref*len(X_train)/batch_size, spatial_dropout=spatial_dropout)
    elif model_type=='bayesian-mars-moon':
        model=bayesian_mars_moon(in_shape=(128,128,3), out_filters = 3, reg_filters = 32, flipout=flipout, dropout=dropout,
                                 kl_scaling=kl_pref*len(X_train)/batch_size, bn=True, spatial_dropout=spatial_dropout)
    else:
        raise BaseException('model_type has to be either bayesian-mars-moon or bayesian-unet, but is {}'.format(model_type))
    optimizer = Adam(learning_rate=lrg, beta_1=0.5,beta_2=0.9999)

    num_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    print('Our model has {} parameters.'.format(num_params))

    kl_losses = []
    mae_losses = []
    total_losses = []
    val_maes = []
    val_stds = []
    idxs = np.array(range(len(X_train)))
    np.random.shuffle(idxs)
    idxs = np.resize(idxs,(int(len(X_train)/batch_size)+1,batch_size))
    stop = False
    epoch=0
    while not stop:
    #for epoch in range(epochs):
        # compute learning rate - decay is implemented
        currLr = computeLR(epoch, epochs, 0.5 * lrg, lrg)
        if currLr < lrg:
            tf.keras.backend.set_value(optimizer.lr, currLr)

        # iterate through training data
        kl_sum = 0
        mae_sum = 0
        total_sum = 0
        #for i, traindata in enumerate(dataset, 0):
        for batch_idxs in idxs:
            # forward pass and loss computation
            with tf.GradientTape() as tape:
                # inputs, targets = traindata
                inputs, targets = X_train[batch_idxs,...], y_train[batch_idxs,...]
                prediction = model(inputs, training=True)
                loss_mae = tf.reduce_mean(mae(prediction, targets))
                kl = sum(model.losses)
                loss_value = kl + tf.cast(loss_mae, dtype='float32')
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
        outputs = model(X_val, training=val_training)
        val_maes += [tf.reduce_mean(mae(y_val, outputs)).numpy()]

        if epoch < 3 or epoch % 20 == 0:
            print('Finnished epoch {}...'.format(epoch))
            print('total loss: {}'.format(total_losses[-1]))
            print('KL loss: {}'.format(kl_losses[-1]))
            print('MAE loss: {}'.format(mae_losses[-1]))
            print('Validation MAE loss: {}'.format(val_maes[-1]))
            print('----------------------------')

        epoch += 1
        # stop if max number of epochs is reached
        if epoch+1 == epochs:
            stop = True

        # early stopping if no improvement is made on validation error for 3 consecutive epochs
        if epoch>=4:
            if not any([i<val_maes[-4] for i in val_maes[-3:]]):
                stop = True

    if save_model:
        model.save(savedir+'model_saved')


    fig,axs=plt.subplots(ncols=3,nrows=1,figsize=(20,4))
    axs[0].plot(kl_losses,color='red')
    axs[0].set_title('KL Loss (Train)')
    axs[1].plot(mae_losses,color='blue',label='train')
    axs[1].plot(val_maes,color='green',label='val')
    axs[1].legend()
    axs[1].set_title('MAE Loss')
    axs[2].plot(total_losses,label='Total',color='black')
    axs[2].plot(kl_losses,label='KL',color='red')
    axs[2].plot(mae_losses,label='MAE',color='blue')
    axs[2].set_title('Total Train Loss')
    axs[2].legend()
    plt.savefig(savedir+'Loss',bbox_inches='tight')

    # test
    reps=20
    preds=np.zeros(shape=(reps,)+X_test.shape)
    for rep in range(reps):
        preds[rep,:,:,:,:]=model(X_test,training=val_training)
    preds_mean=np.mean(preds,axis=0)
    preds_std=np.std(preds,axis=0)

    test_mae_mean = tf.reduce_mean(mae(preds_mean,y_test))
    test_std_mean = tf.reduce_mean(preds_std)

    # store summary statistics in df and append to old df if exists
    df=pd.DataFrame({'model_type':[model_type],'epochs':[epochs], 'early_stopping_epoch':[epoch], 'batch_size':[batch_size],'lrG':[lrg],'flipout':[flipout], 'kl_pref':[kl_pref],
                     'dropout':[dropout], 'spatial_dropout':[spatial_dropout], 'val_mae':[val_maes[-1]],'train_mae':[mae_losses[-1]], 'train_loss':[total_losses[-1]],
                     'train_losses_total':[total_losses], 'train_losses_mae':[mae_losses], 'val_losses_total':[val_maes], 'test_mae_mean':[test_mae_mean], 'test_std_mean':[test_std_mean]})

    if os.path.isfile("../runs/"+folder+"df_summary.pkl"):
        df_old = pd.read_pickle("../runs/"+folder+"df_summary.pkl")
        df_new = pd.concat([df_old, df], ignore_index=True)
        df_new.to_pickle("../runs/"+folder+"df_summary.pkl")
    else:
        df.to_pickle("../runs/"+folder+"df_summary.pkl")

    # plot repeated samples from posterior for some observations
    def plot_BNN_predictions(target, preds, pred_mean, pred_std, num_preds=5,channel=0):
      if num_preds>len(preds):
        print('num_preds was set to {}, but has to be smaller than the length of preds. Setting it to {}'.format(num_preds,len(preds)))
        num_preds = len(preds)

      # transpose and concatenate the frames that are to plot
      to_plot=np.concatenate((target[:,:,channel].transpose().reshape(128,128,1),preds[0:num_preds,:,:,channel].transpose(),
                              pred_mean[:,:,channel].transpose().reshape(128,128,1),pred_std[:,:,channel].transpose().reshape(128,128,1)),axis=-1)
      fig, axs = plt.subplots(nrows=1,ncols=to_plot.shape[-1],figsize=(20,4))
      for i in range(to_plot.shape[-1]):
        label='Target' if i==0 else ('Avg Pred' if i == (num_preds+1) else ('Std Dev (normalized)' if i == (num_preds+2) else 'Pred {}'.format(i)))
        frame = np.flipud(to_plot[:,:,i])
        min=np.min(frame); max = np.max(frame)
        frame -= min; frame /=(max-min)
        axs[i].imshow(frame)
        axs[i].axis('off')
        axs[i].set_title(label)

    obs_idx=5
    plot_BNN_predictions(y_test[obs_idx,...],preds[:,obs_idx,:,:,:],preds_mean[obs_idx,...],preds_std[obs_idx,...])
    plt.savefig(savedir + 'Sample_idx_{}.png'.format(obs_idx), bbox_inches='tight')

    obs_idx=15
    plot_BNN_predictions(y_test[obs_idx,...],preds[:,obs_idx,:,:,:],preds_mean[obs_idx,...],preds_std[obs_idx,...])
    plt.savefig(savedir + 'Sample_idx_{}.png'.format(obs_idx), bbox_inches='tight')

    obs_idx=25
    plot_BNN_predictions(y_test[obs_idx,...],preds[:,obs_idx,:,:,:],preds_mean[obs_idx,...],preds_std[obs_idx,...])
    plt.savefig(savedir + 'Sample_idx_{}.png'.format(obs_idx), bbox_inches='tight')


    IDXS = [1,3,8]
    CHANNEL = 0
    fig, axs = plt.subplots(nrows=len(IDXS),ncols=3,sharex=True, sharey = True, figsize = (9,len(IDXS)*3))
    for i, idx in enumerate(IDXS):
      axs[i][0].imshow(np.flipud(X_test[idx,:,:,CHANNEL].transpose()), cmap=cm.magma)
      axs[i][1].imshow(np.flipud(preds_mean[idx,:,:,CHANNEL].transpose()), cmap=cm.magma)
      axs[i][2].imshow(np.flipud(preds_std[idx,:,:,CHANNEL].transpose()), cmap=cm.viridis)
    axs[0][0].set_title('Shape')
    axs[0][1].set_title('Avg Pred')
    axs[0][2].set_title('Std. Dev')
    plt.savefig(savedir+'different_shapes.png', bbox_inches='tight')

    IDXS = [1,3,8]
    CHANNEL = 0
    fig, axs = plt.subplots(nrows=len(IDXS),ncols=4,sharex=True, sharey = True, figsize = (12,len(IDXS)*3))
    for i, idx in enumerate(IDXS):
      axs[i][0].imshow(np.flipud(X_test[idx,:,:,CHANNEL].transpose()), cmap=cm.magma)
      axs[i][1].imshow(np.flipud(preds_mean[idx,:,:,CHANNEL].transpose()), cmap=cm.magma)
      axs[i][2].imshow(np.flipud(preds_std[idx,:,:,CHANNEL].transpose()), cmap=cm.viridis)
      axs[i][3].imshow(np.flipud(y_test[idx,:,:,CHANNEL].transpose()), cmap=cm.magma)

    axs[0][0].set_title('Shape')
    axs[0][1].set_title('Avg Pred')
    axs[0][2].set_title('Std. Dev')
    axs[0][3].set_title('Target')
    plt.savefig(savedir+'different_shapes_w_target.png', bbox_inches='tight')

if __name__ == "__main__":
    main()

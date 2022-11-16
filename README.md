# airfoils_bnn
This repository contains code that allows to train Bayesian Neural Networks to predict the Reynolds-averaged Navier-Stokes (short: RANS) flows around airfoils. Two versions of BNNs are implemented: Flipout-BNNs, leveraging tensorflow-probability and dropout BNNs. 
### Data
Please refer to https://github.com/thunil/Deep-Flow-Prediction on how to get the data.

### Training
By executing the __train.py__ file the BNNs can be trained and evaluated. In order to specify parts of the training, parameters (like e.g. the number of epochs, or the location of the data) can be passed via click arguments:

```python3 train.py --datadir=../data/train/ --epochs=25```

In order to perform a loop over all three model types (flipout, dropout and non-bayesian networks), the helper file __train_loop.py__ can be used:

```python3 train_loop.py --dropouts=0.01,0.05 --flipouts=100,1000```

The above code snippet will train a non bayesian network and two dropout BNNs with dropout rates 0.01 and 0.05. 

### Evaluation
Loss plots and examples showing repeated samples and the corresponding uncertainty distribution, are stored in a folder named *runs/*. Numerical results, like loss values, standard deviations etc. are additionally stored in a pickled dataframe, if the __train_loop.py__ file is executed. 

### Some Preliminary Results

##### Spatial Dropout vs. normal Dropout
Comparing spatial dropout to normal dropout, one finds that there is a qualitative difference between the two: Spatial dropout smoothes out the smaller variations that are visible in the regular dropout scheme. This is also visible in the resulting uncertainty (note that the uncertainty plots show normalized uncertainty, i.e. it it not possible to compare the magnitudes by colorscheme, but only the distribution across the image).
![alt text](https://github.com/muellerm-95/airfoils_bnn/blob/1b0d919ee03c95e18c320b16ef619b0143abbec6/runs/grid6/spatial_do_True_example.png)
![alt text](https://github.com/muellerm-95/airfoils_bnn/blob/1b0d919ee03c95e18c320b16ef619b0143abbec6/runs/grid6/spatial_do_False_example.png)

##### New Shapes - Flipout 
 <img src="https://github.com/muellerm-95/airfoils_bnn/blob/1b0d919ee03c95e18c320b16ef619b0143abbec6/runs/grid6/bayesian-unet_bsize_64_lrG_0.005_epochs_40_klpref_100.0_spatialDropout_True_dropout_0.0_flipout_True/different_shapes.png" width="400" height="400">
 
##### New Shapes - Dropout 

<img src="https://github.com/muellerm-95/airfoils_bnn/blob/1b0d919ee03c95e18c320b16ef619b0143abbec6/runs/grid6/bayesian-unet_bsize_64_lrG_0.005_epochs_40_klpref_1.0_spatialDropout_True_dropout_0.1_flipout_False/different_shapes.png" width="400" height="400">

### Requirements
The networks were trained successfully on the servus gpu machines with the following specifications:

python 3.6

tensorflow 2.5.0

tensorflow probability 0.12.2

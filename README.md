# airfoils_bnn
This repository contains code that allows to train Bayesian Neural Networks to predict the Reynolds-averaged Navier-Stokes (short: RANS) flows around airfoils. Two versions of BNNs are implemented: Flipout-BNNs, leveraging tensorflow-probability and dropout BNNs. 
### Training
By executing the __train_functional.py__ file the BNNs can be trained and evaluated. In order to specify parts of the training, parameters (like e.g. the number of epochs, or the location of the data) can be passed via click arguments:

```python3 train_functional.py --datadir=../data/train/ --epochs=25```

In order to perform a loop over all three model types (flipout, dropout and non-bayesian networks), the helper file __train_loop.py__ can be used:

```python3 train_master.py --dropouts=0.01,0.05 --flipouts=100,1000```

The above code snippet will train a non bayesian network, two flipout BNNs with Kl-prefactors 100 and 1000 and two dropout BNNs with dropout rates 0.01 and 0.05. 

### Results
The results, consisting of loss plots and examples showing repeated samples and the corresponding uncertainty distribution, will be stored in a folder named *runs/*. Numerical results, like loss values, standard deviations etc. are additionally stored in a pickled dataframe, if the __train_loop.py__ file is executed.

### Requirements
The networks were trained successfully on the servus gpu machines with the following specifications:

python 3.6

tensorflow 2.5.0

tensorflow probability 0.12.2

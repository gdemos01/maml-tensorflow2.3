# Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
This repository contains a TensorFlow 2.3 implementation of the few-shot supervised learning MAML algorithm
introduced introduced in [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (Finn et al., ICML 2017)
](https://arxiv.org/abs/1703.03400) 

## Requirments
- Python 3.*
- TensorFlow 2.3+

## How to use
- The checkpoints repository contains the pre-trained weights for the NN and MAML models (trained on 70000 sinusoids).
- You can test the models without training by using the ``--load_weights`` argument.
- Please keep in mind that the default weights will be overwritten when you train a model without specifying a model name.

#### Arguments
- `--train_nn` - Trains NN Regressor
- `--train_maml ` - Trains MAML
- `--adapt_nn ` - Adapts NN Regressor to new samples
- `--adapt_maml ` - Adaps MAML to new samples
- `--compare_models ` - Compares NN and MAML models after adapting to new samples
- `--step_size ` - The learning rate of the NN Regressor (default: 0.01)
- `--train_size ` - The number of Sinusoids to use for training (default: 70000)
- `--nn_model_name ` - The name of the NN Regressor model (used for storing/loading weights)
- `--maml_model_name ` - The name of the MAML model (used for storing/loading weights)
- `--load_weights ` - Used for loading weights of pretrained models when adapting/comparing
- `--K ` - Number of samples to use for training/testing per sinusoid
- `--alpha ` - Innear step size (learning rate) of the MAML algorithm

#### Train Neural Network Regressor
Trains NN Regressor and saves the weights in /checkpoints using the defined model name

`> python Controller.py --train_nn --train_size 70000 --step_size 0.01 --K 10 --nn_model_name example_name`

#### Train MAML
Trains MAML and saves the wieghts in /checkpoints using the defined model name

`> python Controller.py --train_maml --train_size 70000 --alpha 0.01 --K 10 --maml_model_name example_name`

#### Adapt Neural Network Regressor
Loads the weights of pre-trained NN Regressor and adapts the model to a new Task.
Outputs five examples of adaptation for (0,1,10) gradient steps

`> python Controller.py --adapt_nn --load_weights --nn_model_name default_nn --K 10 --step_size 0.02`

#### Adapt MAML
Loads the weights of pre-trained MAML and adapts the model to a new task.
Outputs five examples of adaptation for (0,1,10) gradient steps

`> python Controller.py --adapt_maml --load_weights --nn_model_name default_maml --K 10`

#### Compare Models
Loads the weights of pre-trained NN Regressor and MAML, adapts for 10 gradients steps and outputs 5
examples (errors and curves).

`> python Controller.py --compare_models --load_weights --K 10 --nn_model_name default_nn --maml_model_name default_maml --step_size 0.02`


## Results

### Adapting NN Regressor - K (5,10,20) - Step Size: 0.01

<img src="https://github.com/gdemos01/maml-tensorflow2.3/blob/main/Results/nn_K5.png">
<img src="https://github.com/gdemos01/maml-tensorflow2.3/blob/main/Results/nn_K10.png">
<img src="https://github.com/gdemos01/maml-tensorflow2.3/blob/main/Results/nn_K20.png">

### Adapting MAML - K (5,10,20)

<img src="https://github.com/gdemos01/maml-tensorflow2.3/blob/main/Results/maml_K5.png">
<img src="https://github.com/gdemos01/maml-tensorflow2.3/blob/main/Results/maml_K10.png">
<img src="https://github.com/gdemos01/maml-tensorflow2.3/blob/main/Results/maml_K20.png">


### Comparing NNR with MAML - K (10) - Step Size: 0.01

<img src="https://github.com/gdemos01/maml-tensorflow2.3/blob/main/Results/comparison_0.01_error.png">
<img src="https://github.com/gdemos01/maml-tensorflow2.3/blob/main/Results/comparison_0.01_curve.png">

### Comparing NNR with MAML - K (10) - Step Size: 0.02

<img src="https://github.com/gdemos01/maml-tensorflow2.3/blob/main/Results/comparison_0.02_error.png">
<img src="https://github.com/gdemos01/maml-tensorflow2.3/blob/main/Results/comparison_0.02_curve.png">

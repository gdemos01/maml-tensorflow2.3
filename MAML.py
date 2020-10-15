import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from DataHandler import DataHandler
import os

class MAML:

    """
        MAML - Model-Agnostic Meta-Learning
        The constructor of this class is responsible for building and compiling the Sequential model (Neural Network)
        as described in the paper, initializing the meta-optimizer and loading the model weights (if checkpoint exists)
    """
    def __init__(self,model_name='default_MAML',load=False):
        self.buildModel()
        self.optimizer = tf.keras.optimizers.Adam() # meta-optimizer
        self.model_name = model_name
        self.checkpoint_loc = os.path.join('checkpoints', self.model_name)
        if load:
            print("> Loading MAML weights")
            self.model.load_weights(self.checkpoint_loc)
        self.compileModel()

    """
        This method builds a model with 40 neurons in each of the two hidden layers.
        The activation function used is ReLU
    """
    def buildModel(self):
        self.model = Sequential()
        self.model.add(Dense(40, activation="relu", input_shape=(1,)))
        self.model.add(Dense(40, activation="relu"))
        self.model.add(Dense(1))

    """
        Compiles the model with the appropriate optimizer and loss function
    """
    def compileModel(self):
        self.model.compile(optimizer=self.optimizer, loss=self.lossFunction)

    """
        Loss function is a simple Mean Square Error between the real and predicted values
    """
    def lossFunction(self, labels, predictions):
        # loss = keras.losses.MeanSquaredError(predictions,labels) -- causes issues with graph
        return keras_backend.mean(keras.losses.mean_squared_error(predictions, labels))

    """
        This method executes a forward pass of the model using input x (model prediction).
        It uses the lossFunction to calculate the loss and returns both the loss and the predictions
    """
    def forward(self,x,y):
        predictions = self.model(x)
        loss = self.lossFunction(y,predictions)
        return loss, predictions

    """
        This is the implementation of "Algorithm 2 MAML for Few-Shot Supervised Learning".
        It trains the model and plots the error over time after iterating through all the Tasks in p(T)
        a = 0.01 - Step size hyperparameter for inner-error calculation and gradient decent
        The numbers denote the step of Algorithm 2 we are currently in.
    """
    def trainMAML(self,epochs,training_set,a = 0.01):
        print("> Training MAML for ",epochs," steps")
        for step in range(epochs): # 2
            totalError = 0
            losses = []
            tasks = training_set # 3 Ti~p(T)
            for i, sinusoid in enumerate(tasks): # 4
                x,y = sinusoid.sampleCurve() # 5 - samples are drawn when the dataset is created
                with tf.GradientTape() as test_tape:
                    with tf.GradientTape() as train_tape:
                        inner_loss, _ = self.forward(x,y) # 6 Computing loss of Ti

                    # 7 - Creating temporary model to compute θ` - applying gradients
                    gradients = train_tape.gradient(inner_loss, self.model.trainable_variables)

                    tempModel = MAML()
                    tempModel.model.set_weights(self.model.get_weights())
                    tempModel.compileModel()

                    # ISSUE - START: TF2 disconnects the graphs when trying to use built in functions (with np.array)
                    # This seems to occur when we try to calculate and apply the gradients within GradientTape
                    # (e.g. optimizer.apply_gradients produces error)
                    # SOLUTION: Manually apply the gradients
                    # CREDIT for possible solutions:
                    # 1) https://stackoverflow.com/questions/62345351/gradients-returned-by-tape-gradient-is-none-in-custom-training-loop
                    # 2) https://stackoverflow.com/questions/58856784/nested-gradient-tape-in-function-tf2-0
                    # 3) https://stackoverflow.com/questions/56916313/tensorflow-2-0-doesnt-compute-the-gradient/56917148#56917148
                    z = 0
                    for j in range(len(tempModel.model.layers)):
                        tempModel.model.layers[j].kernel = tf.subtract(self.model.layers[j].kernel,
                                                                  tf.multiply(a, gradients[z]))
                        tempModel.model.layers[j].bias = tf.subtract(self.model.layers[j].bias,
                                                                tf.multiply(a, gradients[z + 1]))
                        z += 2
                    # ISSUE - END
                    # 8 - Sampling new points for fine-tuning
                    x, y = sinusoid.sampleCurve(new_samples=True)

                # 10 Calculating test-error / outer and applying gradients on original θ
                    outer_loss, outer_logits = tempModel.forward(x, y)
                gradients222 = test_tape.gradient(outer_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients222, self.model.trainable_variables))

                # Monitor average loss over time at each step
                totalError += outer_loss
                loss = totalError / (i + 1.0)
                losses.append(loss)
                if (i+1) % 200 == 0:
                    print("> MAML Training - Step: ",(i+1)," loss: ",float(loss)," saving model to ",self.checkpoint_loc)
                    self.model.save_weights(self.checkpoint_loc) # Save model weights to a checkpoint


            # Plot losses after training is finished
            plt.title('Average loss over time')
            plt.plot(losses,'--')
            plt.show()

    """
        Performs a prediction on new datapoints and evaluates the prediction (loss) 
    """
    def test(self,x_test,y_test):
        loss = self.model.evaluate(x_test, y_test)
        prediction = self.model.predict(x_test)
        #print("> Loss after test = ",loss)
        return prediction, loss





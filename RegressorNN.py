import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import os
import matplotlib.pyplot as plt

class RegressorNN:

    """
        A Simple Regressor Neural Network to learn the SINE Function
        The constructor of this class is responsible for building and compiling the Sequential model
        as described in the paper, initializing the optimizer based on the provided step_size
        and loading the model weights (if checkpoint exists)
    """
    def __init__(self,step_size=0.01,model_name='default_nn',load=False):
        print("> Regressor Neural Network Initialized")
        self.buildModel()
        self.optimizer = keras.optimizers.Adam(learning_rate=step_size)
        self.model_name = model_name
        self.checkpoint_loc = os.path.join('checkpoints', self.model_name)
        if load:
            print("> Loading Pretrained NN")
            self.model.load_weights(self.checkpoint_loc)
        self.compileModel()

    """
        This method builds a model with 40 neurons in each of the two hidden layers.
        The activation function used is ReLU
    """
    def buildModel(self):
        self.model = Sequential()
        self.model.add(Dense(40, activation="relu",input_shape=(1,)))
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
    def lossFunction(self,labels, predictions):
        return keras_backend.mean(keras.losses.mean_squared_error(predictions, labels))

    """
        Training the Regressor and saving the weights in a checkpoint
    """
    def train(self,x_train,y_train,steps,sinusoid_id):
        # steps = epochs. steps_size = learning rate -> Terminology used in the paper

        # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #     verbose=1,
        #     monitor='loss',
        #     filepath=self.checkpoint_loc + '_' + str(sinusoid_id),
        #     save_weights_only=True
        # )
        #history = self.model.fit(x_train, y_train, epochs=steps, callbacks=[checkpoint_callback])

        history = self.model.fit(x_train, y_train, epochs=steps,verbose=0)
        loss = history.history['loss']

        return loss

    """
        Adapting the Regressor to new datapoints
    """
    def adapt(self,x_train,y_train,steps):
        history = self.model.fit(x_train, y_train, epochs=steps)
        loss = history.history['loss']
        return loss

    """
        Performs a prediction on new datapoints and evaluates the prediction (loss) 
    """
    def test(self,x_test,y_test):
        loss = self.model.evaluate(x_test,y_test)
        predictions = self.model.predict(x_test)
        return predictions, loss
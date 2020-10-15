import numpy as np

# A simple Sinusoid object to hold information about a SINE curve
class Sinusoid:

    """
        Creates a new SIN Curve based on amplitude and phase
    """
    def __init__(self,amplitude, phase,K, linspace = False):
        self.amplitude = amplitude
        self.phase = phase
        self.K = K
        self.linespace = linspace
        self.x_samples = self.generateXSamples()
        self.y_samples = self.sinWave(self.x_samples)

    """
        Generates a new sample of K datapoints
    """
    def generateXSamples(self):
        if self.linespace:
            return np.linspace(-5.0, 5.0, self.K)
        else:
            return np.random.uniform(-5,5,self.K)

    """
        The SINE function. Assuming period(p) = 1 and vertical shift(c) = 0
        Y = amplitude * sin[p*(X - phase)] + c
    """
    def sinWave(self,x_samples):
        return self.amplitude * np.sin(x_samples - self.phase)

    """
        Returns the X,Y samples of the Sinusoid OR produces new ones when requested
    """
    def sampleCurve(self,new_samples = False):
        if new_samples:
            x_samples = self.generateXSamples()
            y_samples = self.sinWave(x_samples)
        else:
            x_samples = self.x_samples
            y_samples = self.y_samples
        return x_samples[:, None], y_samples[:, None] # TOOK 3 DAYS TO RESOLVE THIS ISSUE :) -.- :)

class DataHandler:

    """
        Class that handles training, test and query datasets
    """
    def __init__(self):
        print("> Data Handler Initiated")

    """
        Creates a training set with training_size Sinusoids and K samples
    """
    def createTrainingSet(self,K, train_size):
        print("> Creating training set consisting of ", train_size, " tasks")
        training_set = []
        for i in range(train_size):
            amplitude = np.random.uniform(0.1,5.0)
            phase = np.random.uniform(0,np.pi)
            training_set.append(Sinusoid(amplitude,phase,K))

        return training_set

    """
        Creates a testing set with training_size Sinusoids and K samples.
        Linespace = True to produce better graphs
    """
    def createTestingSet(self,K, test_size):
        print("> Creating testing set with Sinusoids")

        test_set = []
        for i in range(0, test_size):
            amplitude = np.random.uniform(0.1, 5.0)
            phase = np.random.uniform(0, np.pi)
            test_set.append(Sinusoid(amplitude, phase, K,linspace=True))

        return test_set

    """
        Creates a Query set that is used for fine-tuning the model
        Support/Query sets is terminology used for meta-learners train/test (not meta test)
        Samples K datapoints from a specific task (SIN curve).
    """
    def createQuerySet(self,x_test,y_test,K):
        training_datapoints = np.random.randint(0, len(x_test), K)
        training_datapoints.sort()

        x_train = []
        y_train = []
        for point in training_datapoints:
            x_train.append(x_test[point])
            y_train.append(y_test[point])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        return x_train,y_train
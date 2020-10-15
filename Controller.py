from DataHandler import DataHandler
from RegressorNN import RegressorNN
from MAML import MAML
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plotSinusoids(curves,curve_names,x_train,y_train,x_test,y_test,plt_title):
    print("> Plotting SIN Curves")
    plt.plot(x_test, y_test, label='Ground Truth')
    plt.plot(x_train, y_train, '^', label='Used for Grad')

    for i in range(len(curves)):
        plt.plot(x_test,curves[i],'--',label=curve_names[i])

    plt.title(plt_title)
    plt.legend(loc='best')
    plt.show()

def plotLosses(model_losses,model_names,plot_title):
    print("> Plotting Losses")

    for i in range(len(model_losses)):
        plt.plot(model_losses[i],'-.',label=model_names[i])

    plt.legend(loc='best')
    plt.xlabel("Number of gradient steps")
    plt.ylabel("MSE")
    plt.title(plot_title)
    plt.show()

def trainMAML(maml,training_set):
    maml.trainMAML(1,training_set)

def trainPriorNeuralNetwork(regressorNN,training_set):
    print("> Training Neural Network - Regressor")
    totalError = 0
    losses = []
    for i, sinusoid in enumerate(training_set):
        #print("> Training Sinusoid #", i)
        loss = regressorNN.train(sinusoid.x_samples, sinusoid.y_samples, 1,i)[0]
        totalError += loss
        latest_loss = totalError / (i + 1.0)
        losses.append(latest_loss)
        # Saves model weights only every 1000 sinusoids
        if (i + 1) % 200 == 0:
            print("> NN Regressor Training - Step: ", (i + 1), " loss: ", float(latest_loss), " saving model to ",
                  regressorNN.checkpoint_loc)
            regressorNN.model.save_weights(regressorNN.checkpoint_loc)  # Save model weights to a checkpoint

    plt.title('Average loss over time')
    plt.plot(losses, '--')
    plt.show()
    return losses

def adaptNeuralNetwork(metaRegressor,x_train,y_train,x_test,y_test,K,step_size):
    print("> Adapting Neural Network - Regressor")
    nn_step0,_ = metaRegressor.test(x_test,y_test)
    metaRegressor.adapt(x_train, y_train, 1)  # 1 grad steps
    nn_step1,_ = metaRegressor.test(x_test,y_test)
    metaRegressor.adapt(x_train, y_train, 10)  # 10 grad steps
    nn_step10,_ = metaRegressor.test(x_test,y_test)
    curves = [nn_step0, nn_step1, nn_step10]
    curve_names = ['pre-update', '1 grad step', '10 grad steps']
    plotSinusoids(curves, curve_names, x_train, y_train, x_test, y_test,
                  'pretrained, K=' + str(K) + ' Step Size: ' + str(step_size))

def adaptMAML(maml,x_train,y_train,x_test,y_test,K):
    print("> Adapting MAML")
    maml_step0,_ = np.array(maml.test(x_test,y_test))
    maml.model.fit(x_train, y_train, epochs=1)
    maml_step1,_ = np.array(maml.test(x_test,y_test))
    maml.model.fit(x_train, y_train, epochs=10)
    maml_step10,_ = np.array(maml.test(x_test,y_test))
    curves = [maml_step0, maml_step1, maml_step10]
    curve_names = ['pre-update', '1 grad step', '10 grad steps']
    plotSinusoids(curves, curve_names, x_train, y_train, x_test, y_test, 'MAML, K= ' + str(K))

def compareMetaLearners(maml,neural_network,x_train,y_train,x_test,y_test,K,untrained_loss):

    nn_losses = [untrained_loss]
    maml_losses = [untrained_loss]

    for i in range(11):
        nn_losses.append(neural_network.adapt(x_train, y_train, 1)[0])
        maml_losses.append(maml.model.fit(x_train, y_train,1).history['loss'][0])

    model_losses = [nn_losses, maml_losses]
    model_names = ['pre-trained, step=' + str(step_size), 'MAML']
    plotLosses(model_losses, model_names, 'K-shot Regression, K=' + str(K))

    curves = []
    curve_names = ['pretrained','MAML']
    nn_predictions,_ = neural_network.test(x_test,y_test)
    maml_predictions,_ = maml.test(x_test,y_test)
    curves.append(nn_predictions)
    curves.append(maml_predictions)
    plotSinusoids(curves, curve_names, x_train, y_train, x_test, y_test, 'Comparison 10-grad K=' + str(K))

if __name__ == '__main__':
    print("\n\n\n─────────────────────────▄▀█▀█▀▄"+"+\n"+
            "────────────────────────▀▀▀▀▀▀▀▀▀"+"\n"+
            "─────────█──────────────▄─░░░░░▄"+"\n"+
          "─▄─█────▐▌▌───█──▄─▄───▐▌▌░░░░░▌▌"+"\n"+
          "▐█▐▐▌█▌▌█▌█▌▄█▐▌▐█▐▐▌█▌█▌█░░░░░▌▌"+"\n"+
          " █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█"+"\n"+
          " █░█░░░█ █▀▀ █░░ █▀▀ █▀▀█ █▀▄▀█ █▀▀ ░█"+"\n"+
          " █░█▄█▄█ █▀▀ █░░ █░░ █░░█ █░▀░█ █▀▀ ░█"+"\n"+
          " █░░▀░▀░ ▀▀▀ ▀▀▀ ▀▀▀ ▀▀▀▀ ▀░░░▀ ▀▀▀ ░█"+"\n"+
          " ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀")

    print("\n> Welcome to Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_nn", action="store_true")
    parser.add_argument("--train_maml",action="store_true")
    parser.add_argument("--adapt_nn", action="store_true")
    parser.add_argument("--adapt_maml",action="store_true")
    parser.add_argument("--compare_models",action="store_true")
    parser.add_argument("--steps")
    parser.add_argument("--step_size")
    parser.add_argument("--train_size")
    parser.add_argument("--nn_model_name")
    parser.add_argument("--maml_model_name")
    parser.add_argument("--load_weights",action="store_true")
    parser.add_argument("--K")
    parser.add_argument("--alpha")
    args = parser.parse_args()

    dataHandler = DataHandler()  # Used for creating datasets
    if args.train_nn:
        step_size = float(args.step_size) if args.step_size else 0.01
        model_name = args.nn_model_name if args.nn_model_name else "default_nn"
        load_weights = True if args.load_weights else False
        train_size = int(args.train_size) if args.train_size else 70000
        K = int(args.K) if args.K else 10

        print("> Parameters: K=",K," step_size=",step_size," train_size=",train_size,
              " model_name=",model_name," load_weights=",load_weights)

        regressorNN = RegressorNN(step_size,model_name,load_weights) # Building NN Model
        training_set = dataHandler.createTrainingSet(K,train_size) # Creating training dataset
        trainPriorNeuralNetwork(regressorNN, training_set) # Training Prior Neural Network
    elif args.train_maml:
        model_name = args.maml_model_name if args.maml_model_name else "default_maml"
        load_weights = True if args.load_weights else False
        train_size = int(args.train_size) if args.train_size else 70000
        K = int(args.K) if args.K else 10
        a = int(args.alpha) if args.alpha else 0.01

        print("> Parameters: K=", K, " train_size=", train_size," model_name=", model_name,
              " load_weights=", load_weights," alpha=",a)

        maml = MAML(model_name,load_weights)
        training_set = dataHandler.createTrainingSet(K, train_size)  # Creating training dataset
        maml.trainMAML(1, training_set,a) # Training MAML using Algorithm 2 from Paper
    elif args.adapt_nn:
        step_size = float(args.step_size) if args.step_size else 0.01
        model_name = args.nn_model_name if args.nn_model_name else "default_nn"
        load_weights = True if args.load_weights else False
        K = int(args.K) if args.K else 10

        print("> Parameters: K=",K," step_size=",step_size,
              " model_name=",model_name," load_weights=",load_weights)

        if load_weights:
            # Plots 5 examples of NN Model Adaptation
            for i in range(5):
                print("> Adaptation exapmple ",(i+1),"/ 5")
                testing_set = dataHandler.createTestingSet(600, 1)
                x_test, y_test = testing_set[0].sampleCurve()
                x_train, y_train = dataHandler.createQuerySet(x_test, y_test, K)
                metaRegressor = RegressorNN(step_size, model_name, load_weights)
                adaptNeuralNetwork(metaRegressor,x_train,y_train,x_test,y_test,K,step_size)
        else:
            print("> To run the adaptation experiments you must include the --load_weights argument. "
                  "\n> If you haven't done so already, please train the model with at least 200 tasks "
                  "before trying to run the adaptation experiments :)")
    elif args.adapt_maml:
        model_name = args.maml_model_name if args.maml_model_name else "default_maml"
        load_weights = True if args.load_weights else False
        K = int(args.K) if args.K else 10
        a = int(args.alpha) if args.alpha else 0.01

        print("> Parameters: K=", K," model_name=", model_name, " load_weights=", load_weights, " alpha=", a)

        if load_weights:
            # Plots 5 examples of NN Model Adaptation
            for i in range(5):
                print("> Adaptation exapmple ",(i+1),"/ 5")
                testing_set = dataHandler.createTestingSet(600, 1)
                x_test, y_test = testing_set[0].sampleCurve()
                x_train, y_train = dataHandler.createQuerySet(x_test, y_test, K)

                meta_maml = MAML(model_name, load_weights)
                adaptMAML(meta_maml, x_train, y_train, x_test, y_test, K)
        else:
            print("> To run the adaptation experiments you must include the --load_weights argument. "
                  "\n> If you haven't done so already, please train the model with at least 200 tasks "
                  "before trying to run the adaptation experiments :)")
    elif args.compare_models:
        step_size = float(args.step_size) if args.step_size else 0.01
        nn_model_name = args.nn_model_name if args.nn_model_name else "default_nn"
        maml_model_name = args.maml_model_name if args.maml_model_name else "default_maml"
        load_weights = True if args.load_weights else False
        K = int(args.K) if args.K else 10
        a = int(args.alpha) if args.alpha else 0.01
        print("> Parameters: K=", K, " maml_model_name=", maml_model_name," nn_model_name=",nn_model_name,
              " step_size=",step_size," load_weights=", load_weights, " alpha=", a)

        if load_weights:
            for i in range(5):
                print("> Adaptation exapmple ",(i+1),"/ 5")
                testing_set = dataHandler.createTestingSet(600, 1)
                x_test, y_test = testing_set[0].sampleCurve()
                x_train, y_train = dataHandler.createQuerySet(x_test, y_test, K)

                untrainedModel =RegressorNN() # For Comparisson purposes in terms of error progress
                _, untrained_loss = untrainedModel.test(x_test, y_test)

                metaRegressor = RegressorNN(step_size, nn_model_name, load_weights)
                meta_maml = MAML(maml_model_name, load_weights)
                compareMetaLearners(meta_maml, metaRegressor, x_train, y_train, x_test, y_test, K, untrained_loss)
        else:
            print("> To run the model comparison experiments you must include the --load_weights argument. "
                  "\n> If you haven't done so already, please train the models with at least 200 tasks "
                  "before trying to run the comparison experiments :)")
    else:
        print("> The biggest mistake you can make in life is to waste your time. – Jerry Bruckner")
        print("> P.S. Use an argument next time: --train_nn, --train_maml, --adapt_nn, --adapt_maml, --compare_models")
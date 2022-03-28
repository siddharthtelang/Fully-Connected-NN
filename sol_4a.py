import layers
import numpy as np
import matplotlib.pyplot as plt
import data_generators

class Network(layers.BaseNetwork):
    #TODO: you might need to pass additional arguments to init for prob 2, 3, 4 and mnist
    def __init__(self, data_layer, myParams=None, params=None):
        # you should always call __init__ first 
        super().__init__()
        #TODO: define your network architecture here
        self.module_list = layers.ModuleList()
        if params is not None:
            hidden_units = params["hidden_units"]
            hidden_layers = params["hidden_layers"]
            self.module_list.append(data_layer)

            for i in range(hidden_layers):
                self.module_list.append(layers.Linear(self.module_list[-1], hidden_units))
                self.module_list.append(layers.Bias(self.module_list[-1]))
                self.module_list.append(layers.Relu(self.module_list[-1]))

            self.module_list.append(layers.Linear(self.module_list[-1],1))
            self.module_list.append(layers.Bias(self.module_list[-1]))
            self.module_list.append(layers.Sigmoid(self.module_list[-1]))
            self.set_output_layer(self.module_list[-1])
        
        elif myParams is not None:
            self.module_list.append(data_layer)
            for units in myParams:
                self.module_list.append(layers.Linear(self.module_list[-1], units))
                self.module_list.append(layers.Bias(self.module_list[-1]))
                self.module_list.append(layers.Relu(self.module_list[-1]))
    
            self.module_list.append(layers.Linear(self.module_list[-1],1))
            self.module_list.append(layers.Bias(self.module_list[-1]))
            self.module_list.append(layers.Sigmoid(self.module_list[-1]))
            self.set_output_layer(self.module_list[-1])

class Trainer:
    def __init__(self):
        pass
    
    def define_network(self, data_layer, parameters=None):
        '''
        For prob 2, 3, 4 and mnist:
        parameters is a dict that might contain keys: "hidden_units" and "hidden_layers".
        "hidden_units" specify the number of hidden units for each layer. "hidden_layers" specify number of hidden layers. 
        Note: we might be testing if your network code is generic enough through define_network. Your network code can be even more general, but this is the bare minimum you need to support.
        Note: You are not required to use define_network in setup function below, although you are welcome to.
        '''
        hidden_units = parameters["hidden_units"] #needed for prob 2, 3, 4, mnist
        hidden_layers = parameters["hidden_layers"] #needed for prob 3, 4, mnist
        #TODO: construct your network here
        network = Network(data_layer=data_layer, params=parameters)
        return network
    
    def setup(self, training_data):
        x, y = training_data
        # store training data
        self.training_data = training_data
        #TODO: define input data layer
        self.data_layer = layers.Data(x)

        #TODO: construct the network. you don't have to use define_network.
        hidden_layers = 3
        hidden_units = 20
        params = {"hidden_units": hidden_units, "hidden_layers": hidden_layers}
        self.network = Network(self.data_layer, params=params)

        #TODO: use the appropriate loss function here
        self.loss_layer = layers.CrossEntropy(self.network.get_output_layer(), y)
        #TODO: construct the optimizer class here. You can retrieve all modules with parameters (thus need to be optimized be the optimizer) by "network.get_modules_with_parameters()"
        self.optim = layers.SGDSolver(0.1, self.network.get_modules_with_parameters())
        return self.data_layer, self.network, self.loss_layer, self.optim
    
    def train_step(self):
        # TODO: train the network for a single iteration
        # you have to return loss for the function
        # return loss

        # forward pass
        loss = self.loss_layer.forward()
        # backward pass
        self.loss_layer.backward()
        # update weights
        self.optim.step()
    
        return loss

    def get_num_iters_on_public_test(self):
        #TODO: adjust this number to how much iterations you want to train on the public test dataset for this problem.
        return 20000
    
    def train(self, num_iter):
        train_losses = []
        #TODO: train the network for num_iter iterations. You should append the loss of each iteration to train_losses.

        for i in range(num_iter):
            train_losses.append(self.train_step())

        return train_losses
    
#DO NOT CHANGE THE NAME OF THIS FUNCTION
def main(test=False):

    #setup the trainer
    trainer = Trainer()
    
    #DO NOT REMOVE THESE IF/ELSE
    if not test:
        # Your code goes here.
        dataset = data_generators.data_4a()
        x_train = dataset["train"][0]
        y_train = dataset["train"][1]

        x_test = dataset["test"][0]
        y_test = dataset["test"][1]

        trainer.setup(dataset["train"])
        iter = 10000

        # trainer.train_step()
        loss = trainer.train(iter)
        # print(loss)
        ran = [i for i in range(1, iter+1)]
        plt.title('Loss vs Iterations')
        plt.plot(ran, loss)
        plt.savefig('Loss 4a')
        plt.show()
        print(loss[-1])

        plt.plot(x_test, y_test)
        plt.show()

        trainer.data_layer.set_data(dataset["test"][0])
        pred = trainer.network.forward()
        plt.plot(trainer.data_layer.data, pred)
        plt.show()

        pos = np.where(x_test > 1)
        neg = np.where(x_test < 1)
        pos = x_test[pos]
        neg = x_test[neg]
        x = [1, 1]
        y = [0, 1]
        plt.plot(x, y, color = 'green')
        plt.scatter(pos, len(pos)*[1], color='red')
        plt.scatter(neg, [0]*len(neg), color='blue')

        plt.show()

        print('Accuracy = ', compute_acc_sigmoid(pred, dataset["test"][1]))

        print(trainer.network.get_modules_with_parameters())


    else:
        #DO NOT CHANGE THIS BRANCH! This branch is used for autograder.
        out = {
            'trainer': trainer
        }
        return out

def compute_acc_sigmoid(pred, y):
  y_pred = np.where(pred > 0.5, 1, 0)
  return (y == y_pred).mean()


if __name__ == "__main__":
    main()
    pass

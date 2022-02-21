import layers
import numpy as np
import matplotlib.pyplot as plt
import data_generators

class Network(layers.BaseNetwork):
    #TODO: you might need to pass additional arguments to init for prob 2, 3, 4 and mnist
    def __init__(self, data_layer):
        # you should always call __init__ first 
        super().__init__()
        #TODO: define your network architecture here
        self.linear = layers.Linear(data_layer, 2)
        self.bias = layers.Bias(self.linear)
        self.relu = layers.Relu(self.bias)
        self.linear2 = layers.Linear(self.relu, 1)
        self.bias2 = layers.Bias(self.linear2)
        # For prob 3 and 4:
        # layers.ModuleList can be used to add arbitrary number of layers to the network
        # e.g.:
        # self.MY_MODULE_LIST = layers.ModuleList()
        # for i in range(N):
        #     self.MY_MODULE_LIST.append(layers.Linear(...))
        
        #TODO: always call self.set_output_layer with the output layer of this network (usually the last layer)
        self.set_output_layer(self.bias2)

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
        network = Network(data_layer=data_layer)
        return network
    
    def setup(self, training_data):
        x, y = training_data
        # store training data
        self.training_data = training_data
        #TODO: define input data layer
        self.data_layer = layers.Data(x)
        #TODO: construct the network. you don't have to use define_network.
        self.network = Network(self.data_layer)
        #TODO: use the appropriate loss function here
        self.loss_layer = layers.SquareLoss(self.network.get_output_layer(), y)
        #TODO: construct the optimizer class here. You can retrieve all modules with parameters (thus need to be optimized be the optimizer) by "network.get_modules_with_parameters()"
        self.optim = layers.SGDSolver(0.004, self.network.get_modules_with_parameters())
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
        return 3000
    
    def train(self, num_iter):
        train_losses = []
        #TODO: train the network for num_iter iterations. You should append the loss of each iteration to train_losses.

        for i in range(num_iter):
            train_losses.append(self.train_step())
            # break
        
        return train_losses
    
#DO NOT CHANGE THE NAME OF THIS FUNCTION
def main(test=False):

    #setup the trainer
    trainer = Trainer()
    
    #DO NOT REMOVE THESE IF/ELSE
    if not test:
        # Your code goes here.
        dataset = data_generators.data_2a()
        x_train = dataset["train"][0]
        y_train = dataset["train"][1]

        x_test = dataset["test"][0]
        y_test = dataset["test"][1]

        trainer.setup(dataset["train"])
        iter = 2000
        # trainer.train_step()
        loss = trainer.train(iter)
        # print(loss)
        ran = [i for i in range(1, iter+1)]
        plt.plot(ran, loss)
        plt.show()
        print(loss[-1])

        plt.plot(x_train, y_train)

        # Test results
        trainer.data_layer.set_data(dataset["test"][0])
        pred = trainer.network.forward()
        plt.plot(trainer.data_layer.data, pred)
        plt.show()

        # print(trainer.network.linear.W)
        # print(trainer.network.bias.W)
        # print(trainer.network.linear.forward())
        # print('------------------------------')
        # print(trainer.network.bias.W)
        # print('------------------------------')
        # print(trainer.network.bias.forward())
        # print('-------------------------------')
        # print(trainer.loss_layer.forward())

    else:
        #DO NOT CHANGE THIS BRANCH! This branch is used for autograder.
        out = {
            'trainer': trainer
        }
        return out

if __name__ == "__main__":
    main()
    pass

import layers
import numpy as np
import matplotlib.pyplot as plt
import data_generators
import random

class Network(layers.BaseNetwork):
    #TODO: you might need to pass additional arguments to init for prob 2, 3, 4 and mnist
    def __init__(self, data_layer, params=None):
        # you should always call __init__ first 
        super().__init__()

        self.module_list = layers.ModuleList()
        if params is not None:
            print('Params defined - ', params)
            hidden_units = params["hidden_units"]
            hidden_layers = params["hidden_layers"]
            self.module_list.append(data_layer)

            for i in range(hidden_layers):
                self.module_list.append(layers.Linear(self.module_list[-1], hidden_units))
                self.module_list.append(layers.Bias(self.module_list[-1]))
                self.module_list.append(layers.Relu(self.module_list[-1]))

            self.module_list.append(layers.Linear(self.module_list[-1],10))
            self.module_list.append(layers.Bias(self.module_list[-1]))
            self.module_list.append(layers.Relu(self.module_list[-1]))

            self.set_output_layer(self.module_list[-1])

        else:
            self.linear = layers.Linear(data_layer, 512)
            self.bias = layers.Bias(self.linear)
            self.relu = layers.Relu(self.bias)

            self.linear2 = layers.Linear(self.relu, 256)
            self.bias2 = layers.Bias(self.linear2)
            self.relu2 = layers.Relu(self.bias2)

            self.linear3 = layers.Linear(self.relu2, 128)
            self.bias3 = layers.Bias(self.linear3)
            self.relu3 = layers.Relu(self.bias3)
            
            self.linear5 = layers.Linear(self.relu3, 10)
            self.bias5 = layers.Bias(self.linear5)
            self.relu5 = layers.Relu(self.bias5)

            self.set_output_layer(self.relu5)

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
        self.data_layer = layers.Data(x[:500])

        #TODO: construct the network. you don't have to use define_network.
        hidden_layers = 1
        hidden_units = 784
        params = {"hidden_units": hidden_units, "hidden_layers": hidden_layers}
        self.network = Network(self.data_layer, params=None)

        #TODO: use the appropriate loss function here
        self.loss_layer = layers.CrossEntropySoftMax(self.network.get_output_layer())
        #TODO: construct the optimizer class here. You can retrieve all modules with parameters (thus need to be optimized be the optimizer) by "network.get_modules_with_parameters()"
        self.optim = layers.SGDSolver(0.05, self.network.get_modules_with_parameters())
        return self.data_layer, self.network, self.loss_layer, self.optim
    
    def train_step(self):

        # set the batch
        batch_size = 200
        samples = random.sample(range(1, len(self.training_data[0])), batch_size)
        train = self.training_data[0]
        test = self.training_data[1]

        self.data_layer.set_data(train[samples])
        self.loss_layer.set_data(test[samples])

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
        acc = []
        #TODO: train the network for num_iter iterations. You should append the loss of each iteration to train_losses.

        for i in range(num_iter):
            train_losses.append(self.train_step())
            print('Loss in iteration ', (i+1), ' = ', train_losses[-1])
            pred = self.network.forward()
            acc.append(compute_acc_softmax(pred, self.loss_layer.labels))

        ran = [i for i in range(1, num_iter+1)]
        plt.title('Loss vs Iteration')
        plt.plot(ran, train_losses)
        plt.show()
        plt.title('Training Accuracy')
        plt.plot(ran, acc)
        plt.show()
        return train_losses
    
#DO NOT CHANGE THE NAME OF THIS FUNCTION
def main(test=False):

    #setup the trainer
    trainer = Trainer()
    
    #DO NOT REMOVE THESE IF/ELSE
    if not test:
        # Your code goes here.
        # For prob mnist: example data loading code:
        
        mnist = np.load("mnist.pkl", allow_pickle=True)
        x_train = mnist["training_images"] / 255
        y_train = mnist["training_labels"]
        x_test = mnist["test_images"] / 255
        y_test = mnist["test_labels"]
        
        trainer.setup((x_train, y_train))
        iter = 4000

        # trainer.train_step()
        loss = trainer.train(iter)
        # print(loss)
        ran = [i for i in range(1, iter+1)]
        plt.plot(ran, loss)
        plt.show()
        print(loss[-1])

        trainer.data_layer.set_data(x_test)
        pred = trainer.network.forward()
        acc = compute_acc_softmax(pred, y_test)
        print("Accuracy = ", acc)

        # For prob mnist: you can use this snippet to save the network weight:
        np.savez("mnist_weight.npz", weight=trainer.network.state_dict())
        
        print(trainer.network.get_modules_with_parameters())


    else:
        #DO NOT CHANGE THIS BRANCH! This branch is used for autograder.
        out = {
            'trainer': trainer
        }
        return out

def compute_acc_softmax(pred, y):
  y_pred = np.argmax(pred, axis=-1)
  return (y == y_pred).mean()


if __name__ == "__main__":
    main()
    pass

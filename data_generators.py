import numpy as np

def data_1a():
    x = (np.linspace(-2, 2, 60))[:, np.newaxis]
    y = 7*x + 3 + np.random.randn(60, 1)*0.1
    x_train = x[::2]
    y_train = y[::2]

    x_test = x[1::2]
    y_test = y[1::2]
    return {
        "train": (x_train, y_train),
        "test": (x_test, y_test)
    }


def data_1b():
    #generating data
    x1_all = np.linspace(-2, 2, 60)
    x2_all = np.linspace(-2, 2, 60)
    x1_train = x1_all[::2]
    x2_train = x1_all[::2]
    x_train = np.array(np.meshgrid(x1_train,x2_train)).transpose().reshape(-1, 2)
    A = np.array([[2, -3]])
    b = np.array([5])
    y_train = np.einsum("ij,...j->...i", A, x_train) + np.random.randn(900, 1)*0.1
    #gen test data
    x1_test = x1_all[1::2]
    x2_test = x2_all[1::2]
    x_test = np.array(np.meshgrid(x1_test,x2_test)).transpose().reshape(-1, 2)
    y_test = np.einsum("ij,...j->...i", A, x_test) + np.random.randn(900, 1)*0.1
    return {
        "train": (x_train, y_train),
        "test": (x_test, y_test)
    }



def data_2a():
    #generating data
    x = np.linspace(-2, 2, 30)[:, np.newaxis]
    y = np.sin(x) + np.random.randn(30, 1)*0.02
    x_train = x[::2]
    y_train = y[::2]

    x_test = x[1::2]
    y_test = y[1::2]

    return {
        "train": (x_train, y_train),
        "test": (x_test, y_test)
    }



def data_2b():

    def compute_y(grid):
        y = 1.7* grid[:, 0:1]**2 + 0.6 * grid[:, 1:2] + 2 + np.random.randn(900, 1)*0.1
        return y

    #generating data
    x1_all = np.linspace(-2, 2, 60)[:, np.newaxis]
    x2_all = np.linspace(-2, 2, 60)[:, np.newaxis]
    x1_train = x1_all[::2]
    x2_train = x1_all[::2]
    x_train = np.array(np.meshgrid(x1_train,x2_train)).transpose().reshape(-1, 2)
    y_train = compute_y(x_train)
    #gen test data
    x1_test = x1_all[1::2]
    x2_test = x2_all[1::2]
    x_test = np.array(np.meshgrid(x1_test,x2_test)).transpose().reshape(-1, 2)
    y_test = compute_y(x_test)
    return {
        "train": (x_train, y_train),
        "test": (x_test, y_test)
    }



def data_3a():
    #generating data
    x = np.linspace(-2, 2, 30)[:, np.newaxis]
    y = np.sin(x) + np.random.randn(30, 1)*0.02
    x_train = x[::2]
    y_train = y[::2]

    x_test = x[1::2]
    y_test = y[1::2]
    return {
        "train": (x_train, y_train),
        "test": (x_test, y_test)
    }



def data_3b():

    def compute_y(grid):
        y = 1.7* grid[:, 0:1]**2 + 0.6 * grid[:, 1:2] + 2 + np.random.randn(900, 1)*0.1
        return y

    #generating data
    x1_all = np.linspace(-2, 2, 60)[:, np.newaxis]
    x2_all = np.linspace(-2, 2, 60)[:, np.newaxis]
    x1_train = x1_all[::2]
    x2_train = x1_all[::2]
    x_train = np.array(np.meshgrid(x1_train,x2_train)).transpose().reshape(-1, 2)
    y_train = compute_y(x_train)
    #gen test data
    x1_test = x1_all[1::2]
    x2_test = x2_all[1::2]
    x_test = np.array(np.meshgrid(x1_test,x2_test)).transpose().reshape(-1, 2)
    y_test = compute_y(x_test)
    return {
        "train": (x_train, y_train),
        "test": (x_test, y_test)
    }



def data_4a():
    #generating data
    x = np.linspace(-2, 2, 30)[:, np.newaxis]
    y = np.where(x > 1, 1, 0)
    x_train = x[::2]
    y_train = y[::2]

    x_test = x[1::2]
    y_test = y[1::2]
    return {
        "train": (x_train, y_train),
        "test": (x_test, y_test)
    }


def data_4b():
    #generating data
    x1_all = np.linspace(-2, 2, 60)
    x2_all = np.linspace(-2, 2, 60)
    x1_train = x1_all[::2]
    x2_train = x1_all[::2]
    x_train = np.array(np.meshgrid(x1_train,x2_train)).transpose().reshape(-1, 2)
    A = np.array([[2, -3]])
    y_train = np.where(np.einsum("ij,...j->...i", A, x_train) > 0, 1, 0)
    #gen test data
    x1_test = x1_all[1::2]
    x2_test = x2_all[1::2]
    x_test = np.array(np.meshgrid(x1_test,x2_test)).transpose().reshape(-1, 2)
    y_test = np.where(np.einsum("ij,...j->...i", A, x_test) > 0, 1, 0)
    return {
        "train": (x_train, y_train),
        "test": (x_test, y_test)
    }


def generate_default_data(module_name):
    func_name = module_name.replace("sol_", "data_")
    return globals()[func_name]()
    
if __name__ == "__main__":
    print(generate_default_data("sol_1a"))
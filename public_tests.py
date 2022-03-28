import os
import numpy as np
from utils import run_tests
import data_generators
import copy


answers = {
            'sol_1a':    {'loss_final_thresh': 0.01,  'num_layers': [2] },
            'sol_1b':    {'loss_final_thresh':  0.05,  'num_layers': [2] },
            'sol_2a':    {'loss_final_thresh': 0.005, 'num_layers': [4] },
            'sol_2b':    {'loss_final_thresh':  0.2,  'num_layers': [4] },
            'sol_3a':    {'loss_final_thresh': 0.001, 'num_layers': [7,8,9,10,11,12,13,14,15] },
            'sol_3b':    {'loss_final_thresh': 0.10,   'num_layers': [7,8,9,10,11,12,13,14,15] },
            'sol_4a':    {'acc_final_thresh': 0.9,  'num_layers': [7,8,9,10,11,12,13,14,15] },
            'sol_4b':    {'acc_final_thresh': 0.9,  'num_layers': [7,8,9,10,11,12,13,14,15] },
            'sol_mnist': {'acc_final_thresh': 0.9 ,  'num_layers': [7,8,9,10,11,12,13,14,15] }
          }


def test_network_arch(module_name, test_data, test_answers):
  # print('enter arch')
  num_layers = test_answers['num_layers']
  num_correct, num_total = 0,0
  if module_name == "sol_mnist":
    mnist = np.load("mnist.pkl", allow_pickle=True)
    x_train = mnist["training_images"]
    x_label = mnist["training_labels"]
    dataset = {"train": (x_train, x_label)}
  else:
    dataset = data_generators.generate_default_data(module_name)

  trainer = test_data["trainer"]
  trainer.setup(dataset["train"])
  layers = trainer.network.get_modules_with_parameters()
  #Does network have correct number of layers?
  if len(layers) in num_layers:
    num_correct += 1
  num_total += 1
  # print('exit arch')
  return num_correct, num_total

def compute_acc_softmax(pred, y):
  y_pred = np.argmax(pred, axis=-1)
  return (y == y_pred).mean()

def compute_acc_sigmoid(pred, y):
  y_pred = np.where(pred > 0.5, 1, 0)
  return (y == y_pred).mean()

def compute_mse(pred, y):
  return 1/2 * ((pred - y)**2).mean()

def test_final_mse(module_name, test_data, test_answers):
  # print('test mse enter')
  allowed_module = ["sol_1a", "sol_1b", "sol_2a", "sol_2b", "sol_3a", "sol_3b"]
  if module_name not in allowed_module:
    return 0,0

  thresh = test_answers['loss_final_thresh']
  num_correct, num_total = 0,0
  dataset = data_generators.generate_default_data(module_name)
  trainer = test_data["trainer"]
  trainer.setup(dataset["train"])
  # print('setup ok')
  trainer.train(num_iter=trainer.get_num_iters_on_public_test())
  # print('train ok')

  trainer.data_layer.set_data(dataset["test"][0])
  pred = trainer.network.forward()
  # print('test forward ok')
  #Is final training loss less than some threshold?
  mse = compute_mse(pred, dataset["test"][1])
  print("mse", mse)
  #exit()

  if mse <= thresh:
    num_correct += 1
  num_total += 1
  # print('test mse exit')
  return num_correct, num_total

def test_final_acc(module_name, test_data, test_answers):
  allowed_module = ["sol_4a", "sol_4b"]
  if module_name not in allowed_module:
    return 0,0

  thresh = test_answers['acc_final_thresh']
  num_correct, num_total = 0,0
  dataset = data_generators.generate_default_data(module_name)

  trainer = test_data["trainer"]
  trainer.setup(dataset["train"])
  trainer.train(num_iter=trainer.get_num_iters_on_public_test())
  trainer.data_layer.set_data(dataset["test"][0])
  pred = trainer.network.forward()
  #Is final training loss less than some threshold?
  acc = compute_acc_sigmoid(pred, dataset["test"][1])
  print("acc", acc)
  if acc >= thresh:
    num_correct += 1
  num_total += 1
  return num_correct, num_total

def test_mnist_acc(module_name, test_data, test_answers):
  allowed_module = ["sol_mnist"]
  if module_name not in allowed_module:
    return 0,0

  thresh = test_answers['acc_final_thresh']
  num_correct, num_total = 0,0
  trainer = test_data["trainer"]
  mnist = np.load("mnist_mini.pkl", allow_pickle=True)
  x_train = mnist["training_images"]
  y_train = mnist["training_labels"]
  x_test = mnist["test_images"]
  y_test = mnist["test_labels"]
  trainer.setup((x_train, y_train))
  trainer.network.load_state_dict(np.load("mnist_weight.npz", allow_pickle=True)["weight"])

  trainer.data_layer.set_data(x_test)
  #Is final training loss less than some threshold?
  pred = trainer.network.forward()
  acc = compute_acc_softmax(pred, y_test)
  print("acc", acc)
  if acc >= thresh:
    num_correct += 1
  num_total += 1
  return num_correct, num_total

def perturb_network_loss(network, perturb, loss_layer):
  """
  Returns copy
  """
  network_wph = copy.deepcopy(network)
  loss_wph = copy.deepcopy(loss_layer)
  weights = network_wph.get_modules_with_parameters()
  for i, (w,h) in enumerate(zip(weights,perturb)):
     w.W += h
  loss_wph.in_layer = network_wph
  return loss_wph

def test_gradients(module_name, test_data, test_answers):
  # print('test grad enter')
  disallowed_module = ["sol_mnist"]
  if module_name in disallowed_module:
    return 0,0

  num_correct, num_total = 0,0
  dataset = data_generators.generate_default_data(module_name)
  trainer = test_data["trainer"]
  #Is the gradient numerically accurate?
  if module_name == "sol_1a":
    x_train = np.random.normal(size=(1, 1))
  else:
    x_train = np.random.normal(size=(1, 3))


  y_train = np.array([0])
  data_layer, network, loss_layer, optim =  trainer.setup((x_train, y_train))
  weights = network.get_modules_with_parameters()
  h = [np.random.normal(size=w.W.shape) for w in weights]
  h = [(hh / np.linalg.norm(hh))*np.linalg.norm(x_train) for hh in h]
  neg_h = [-hh for hh in h]

  loss = loss_layer.forward()
  loss_layer.backward()
  grad_w = [w.G.mean(axis=0) for w in weights]

  rel_errors = np.zeros(10)
  for iter in range(10):
    h = [hh/10 for hh in h]
    loss_wph = perturb_network_loss(network, h, loss_layer)
    delta_f = loss_wph.forward() - loss

    delta_f_approx = np.array([(g*hh).sum() for g,hh in zip(grad_w, h)]).sum()
    delta_f = delta_f.sum()
    #Compute relative error
    rel_error = (delta_f - delta_f_approx) / delta_f
    rel_errors[iter] = rel_error
    pass

  abs_rel_error = min(np.abs(rel_errors))
  if abs_rel_error < 1e-6:
    num_correct += 1
  else:
    print('abs_rel_error is more than 1e-6')
  num_total += 1
  # print('test grad exit')
  return num_correct, num_total

if __name__ == "__main__":

  solution_directory = 'solutions'
  #NB: this assumes tests are defined in global namespace and names prefixed with "test_"
  tests = {k:v for k,v in globals().items() if "test_" in k and callable(v)}
  run_tests(solution_directory, tests, answers)


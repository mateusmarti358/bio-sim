import numpy as np

def softmax(x):
  exp_x = np.exp(x - np.max(x))
  return exp_x / np.sum(exp_x, axis=0)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return max(0, x)

def tanh(x):
  return np.tanh(x)

def random_list(size: int):
  output = []

  for _ in range(size):
    output.append(np.random.random())

  return output


class Neuron:
  def __init__(self, inputs, activ_func, weights=None, bias=None):
    self.inputs = inputs

    self.weights = random_list(len(inputs)) if weights is None else weights

    self.bias = np.random.random() if bias is None else bias

    self.activ_func = activ_func

  def mutate(self, mutation_rate):
    for i in range(len(self.weights)):
      self.weights[i] *= np.random.uniform(1, mutation_rate)
      self.weights[i] = max(min(self.weights[i], 10.0), -10.0)

    self.bias *= np.random.normal(1, mutation_rate)
    self.bias = max(min(self.bias, 10.0), -10.0)

  def process(self, inputs, others):
    activation = 0

    for (is_input, input_id), weight in zip(self.inputs, self.weights):
      if is_input:
        activation += inputs[input_id] * weight
        continue

      activation += others[input_id] * weight

    activation += self.bias

    return self.activ_func(activation)

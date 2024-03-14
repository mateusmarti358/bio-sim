class Brain:
  def __init__(self, neurons, outputs):
    self.neurons = neurons

    self.outputs = outputs

  def mutate(self, mutation_rate, generation):
    mutation_rate = mutation_rate * (1 / (1 + generation))

    for neuron in self.neurons:
      neuron.mutate(mutation_rate)

    for output_neuron in self.outputs:
      output_neuron.mutate(mutation_rate)

    return self

  def process(self, inputs):
    neurons = []

    for neuron in self.neurons:
      neurons.append(neuron.process(inputs, neurons))

    outputs = []

    for output_neuron in self.outputs:
      outputs.append(output_neuron.process(inputs, neurons))

    return outputs

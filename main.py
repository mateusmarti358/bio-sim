import pygame

import numpy as np
import random

from brain.neuron import Neuron, relu, softmax, tanh
from brain.brain import Brain

pygame.init()

# Set up the screen
WIDTH, HEIGHT = 1000, 1000
canvas = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

RED = [255, 0, 0]
GREEN = [255, 0, 0]
BLUE = [255, 0, 0]


def random_color():
  return random.choice([RED, GREEN, BLUE])


def random_x():
  return np.random.randint(0, int(WIDTH / 10)) * 10


def random_y():
  return np.random.randint(0, int(HEIGHT / 10)) * 10


def standard_brain():
    return Brain([Neuron([(True, 0), (True, 1), (True, 2)], softmax),
                  Neuron([(True, 0), (True, 1), (True, 2)], softmax)],
                 
                 # Output neurons

                 [Neuron([(True, 0), (True, 1), (True, 2), (False, 0), (False, 1)], relu), # STOP
                  
                  Neuron([(True, 0), (True, 1), (True, 2), (False, 0), (False, 1)], relu), # LEFT
                  Neuron([(True, 0), (True, 1), (True, 2), (False, 0), (False, 1)], relu), # RIGHT

                  Neuron([(True, 0), (True, 1), (True, 2)], relu), # UP
                  Neuron([(True, 0), (True, 1), (True, 2)], relu), # DOWN

                  Neuron([(True, 0), (True, 1), (True, 2), (False, 0), (False, 1)], tanh), # Set Red
                  Neuron([(True, 0), (True, 1), (True, 2), (False, 0), (False, 1)], tanh), # Set Green
                  Neuron([(True, 0), (True, 1), (True, 2), (False, 0), (False, 1)], tanh), # Set Blue
                  ]) 


class Cell:
  def __init__(self, x, y, color, brain=None):
    self.health = 100

    self.x = x
    self.y = y

    self.dx = 0
    self.dy = 0

    self.color = color

    self.osc = 0.0

    self.brain = standard_brain() if brain is None else brain

  def update(self):
    neurons = self.brain.process([self.x, self.y, np.sin(self.osc)])
    action = np.argmax(neurons[0:4])

    if action == 0:
      self.dx = 0
      self.dy = 0

    elif action == 1:
      self.dx = -1
      self.x -= 10

    elif action == 2:
      self.dx = 1
      self.x += 10

    elif action == 3:
      self.dy = -1
      self.y -= 10

    elif action == 4:
      self.dy = 1
      self.y += 10

    if neurons[5] < -0.75:
      self.color[0] -= 1
      self.color[0] %= 256
    elif neurons[5] > 0.75:
      self.color[0] += 1
      self.color[0] %= 256

    if neurons[6] < -0.75:
      self.color[1] -= 1
      self.color[1] %= 256
    elif neurons[6] > 0.75:
      self.color[1] += 1
      self.color[1] %= 256

    if neurons[7] < -0.75:
      self.color[2] -= 1
      self.color[2] %= 256
    elif neurons[7] > 0.75:
      self.color[2] += 1
      self.color[2] %= 256

    self.x = self.x % WIDTH
    self.y = self.y % HEIGHT

    self.osc += 0.1

  def draw(self):
    pygame.draw.rect(canvas, (self.color[0], self.color[1], self.color[2]), (self.x, self.y, 10, 10))


cells = []

brains = []

generation = 1


def next_gen():
  global generation

  brains.clear()
  for cell in cells:
    brains.append(cell.brain)

  cells.clear()

  while len(brains) < 85:
    brains.append(standard_brain())

  while len(brains) > 85:
    brains.pop(np.random.randint(0, len(brains)))
  

  for brain in brains:
    cells.append(Cell(random_x(), random_y(), random_color(), brain.mutate(1.0, generation)))
    for _ in range(2):
      cells.append(Cell(random_x(), random_y(), random_color(), brain.mutate(1.0, generation)))

  print(generation)
  generation += 1

def death_factor(cell):
  return cell.x < WIDTH / 2
def update():
  for cell in cells:
    if cell.health <= 1:
      cells.remove(cell)

    if np.random.random() < 0.3 and death_factor(cell):
      cell.health -= 33

    cell.update()

def draw():
  canvas.fill(BLACK)

  for cell in cells:
    cell.draw()

  pygame.display.update()


if __name__ == '__main__':
  for _ in range(250):
    cells.append(Cell(random_x(), random_y(), random_color()))


  running = True
  paused = False
  clock = pygame.time.Clock()

  ttick = 12
  gen_time = 0

  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_SPACE:
          paused = not paused

    draw()
    
    if paused: continue
    
    update()

    if gen_time > ttick * 500:
      gen_time = 0
      next_gen()

    # Cap the frame rate
    clock.tick(ttick)
    gen_time += ttick * 10

  # Quit Pygame
  pygame.quit()

import numpy as np
import random
import pygame

WIDTH, HEIGHT = 600, 400
TARGET = np.array([500, 200])
POP_SIZE = 50
STEPS = 200

class Agent:
    def __init__(self, brain=None):
        self.pos = np.array([50.0, HEIGHT/2])
        self.step = 0
        self.alive = True
        self.brain = brain if brain is not None else self._random_brain()
    
    def _random_brain(self):
        return np.random.uniform(-1, 1, (STEPS, 2))

    def update(self):
        if self.step < STEPS:
            self.pos += self.brain[self.step]
            self.step += 1

    def fitness(self):
        return 1.0 / (np.linalg.norm(self.pos - TARGET) + 1)

def evolve(pop):
    pop.sort(key=lambda a: a.fitness(), reverse=True)
    survivors = pop[:10]

    new_pop = survivors.copy()
    while len(new_pop) < POP_SIZE:
        parent = random.choice(survivors)
        child_brain = parent.brain + np.random.normal(0, 0.1, parent.brain.shape)
        new_pop.append(Agent(child_brain))
    return new_pop

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

population = [Agent() for _ in range(POP_SIZE)]
generation = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((30, 30, 30))
    pygame.draw.circle(screen, (255, 0, 0), TARGET.astype(int), 8)

    all_done = True
    for agent in population:
        agent.update()
        pygame.draw.circle(screen, (0, 200, 255), agent.pos.astype(int), 3)
        if agent.step < STEPS:
            all_done = False

    pygame.display.flip()

    if all_done:
        generation += 1
        print(f"Generation {generation}")
        population = evolve(population)

pygame.quit()
import neat
import pygame

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0
        # Game loop hier einbauen
        # fitness erhöhen, wenn der Agent länger überlebt
        genome.fitness = fitness

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config-feedforward"
)

population = neat.Population(config)
population.run(eval_genomes, 50)
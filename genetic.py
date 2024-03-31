from neural_network import NeuralNetwork
import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, network_size=[4, 24, 2], mr=0.5, pop_size=100) -> None:
        self.population = []
        self.mr = mr
        
        for i in range(pop_size):
            self.population.append(NeuralNetwork(shape=network_size))
        
    
    def fitness(self, inputs, targets):
        fitness_scores = []
        for network in self.population:
            outputs = network.forward(inputs)
            # Convert outputs to a more probabilistic form if not already
            outputs = np.clip(outputs, 1e-9, 1 - 1e-9)  # Avoid division by zero or log(0)
            # Calculate log loss for binary classification
            log_loss = -np.mean(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs))
            # Convert log loss to fitness score, where lower loss = higher fitness
            fitness_score = 1 / (log_loss + 1e-9)
            fitness_scores.append(fitness_score)
        return fitness_scores
        
    def mutation(self, network):
        for layer in range(len(network.weights)):
            network.weights[layer] += (np.random.randn(*network.weights[layer].shape)-0.5) * self.mr
            network.bias[layer] += (np.random.randn(*network.bias[layer].shape)-0.5) * self.mr
    
    def selection(self, fitness_scores):
        # Example of selection process
        # [0.4, 0.1, 0.2, 0.3] -> [1]
        
        # rand(0, 1) = 0.25
        # 1. 0.25 - 0.1 = 0.15
        # 2. 0.15 - 0.2 = -0.05
        
        # rand(0, 1) = 0.6
        # 1. 0.6 - 0.1 = 0.5
        # 2. 0.5 - 0.2 = 0.3
        # 3. 0.3 - 0.3 = 0.0   
        
        # rand(0, 1) = 0.61
        # 1. 0.61 - 0.1 = 0.51
        # 2. 0.51 - 0.2 = 0.31
        # 3. 0.31 - 0.3 = 0.01
        # 4. 0.01 - 0.4 = -0.39
        
        sum = np.sum(fitness_scores)
        r = random.uniform(0, sum)
        
        for i in range(len(fitness_scores)):
            r -= fitness_scores[i]
            
            if r <= 0:
                return self.population[i]
                
        return self.population[-1]
        
    
    def make_babies(self, inputs, targets):
        new_population = []
        fitness_scores = self.fitness(inputs, targets)
        # Elitism
        new_population.append(self.population[np.argmax(fitness_scores)])
        
        while len(new_population) < len(self.population):
            child = self.selection(fitness_scores)
            self.mutation(child)
            new_population.append(child)
            
        self.population = new_population
        
    def find_best(self, inputs, targets):
        fitness_scores = self.fitness(inputs, targets)
        return self.population[np.argmax(fitness_scores)]

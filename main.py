import numpy as np
from genetic import GeneticAlgorithm

def main():
    # Step 1: Create a Dataset for the OR function
    # Each data point is a tuple of (input Matrix, expected output)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [1]])
    
    # Create a population
    population = GeneticAlgorithm(network_size=[2, 4, 4, 1], mr=0.05, pop_size=100)
    
    # Perform the genetic algorithm
    for i in range(1000):
        if i % 100 == 0:
            print("Generation: ", i)
            best = population.find_best(X, Y)
            pred = best.forward(X)
            print("Best prediction:\n", pred)
            print("Actual:\n", Y)
        
        population.make_babies(X, Y)

if __name__ == "__main__":
    main()


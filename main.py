import numpy
import random
from deap import base
from deap import creator
from deap import algorithms
from deap import tools
from deap.tools import Statistics
import plotly.express as px


class Product:
    def __init__(self, name, space, price):
        self.name = name
        self.space = space
        self.price = price


# [ 0, 1, 1, 0, ..., 1]
def fitness(solution):
    cost = 0
    sum_space = 0
    for i in range(len(solution)):
        if solution[i] == 1:
            cost += prices[i]
            sum_space += spaces[i]
    if sum_space > limit:
        cost = 1

    return cost,


if __name__ == '__main__':
    products_list = [Product('Refrigerator A', 0.751, 999.90), Product('Cell phone', 0.00000899, 2911.12),
                     Product('TV 55', 0.400, 4346.99), Product("TV 50' ", 0.290, 3999.90),
                     Product("TV 42' ", 0.200, 2999.00), Product("Notebook A", 0.00350, 2499.90),
                     Product("Ventilator", 0.496, 199.90), Product("Microwave A", 0.0424, 308.66),
                     Product("Microwave B", 0.0544, 429.90), Product("Microwave C", 0.0319, 299.29),
                     Product("Refrigerator B", 0.635, 849.00), Product("Refrigerator C", 0.870, 1199.89),
                     Product("Notebook B", 0.498, 1999.90), Product("Notebook C", 0.527, 3999.00)]
    spaces = []
    prices = []
    names = []
    for product in products_list:
        print(product.name, '-', product.space, '-', product.price)
        spaces.append(product.space)
        names.append(product.name)
        prices.append(product.price)

    limit = 3  # Maximum capacity of the Truck
    mutation_probability = 0.01
    number_of_generation = 100

    toolbox = base.Toolbox()  # to initialise the genetic algorithm
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))  # The results will be in the range of 0 to 1
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox.register('attr_boo', random.randint, 0, 1)  # boolean attribute, generate random  zeros and ones

    # tools.initRepeat to generate random solution, n=14 is the size we have 14 products
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_boo, n=14)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', fitness)
    toolbox.register('mate', tools.cxOnePoint)  # apply crossover
    toolbox.register('mutate', tools.mutFlipBit, indpb =mutation_probability)  # indpb  is mutation rate
    toolbox.register('select', tools.selRoulette)

    population = toolbox.population(n=20)
    crossover_probability = 1.0

    # We are accessing the values of each one of Generations
    statistic: Statistics = tools.Statistics(key = lambda individual: individual.fitness.values)
    statistic.register('max', numpy.max)  # Max number in a list
    statistic.register('min', numpy.min)  # Min number in a list
    statistic.register('std', numpy.std)  # Min number in a list

    population, info = algorithms.eaSimple(population, toolbox, crossover_probability,
                                           mutation_probability, number_of_generation, statistic)

    best_solutions = tools.selBest(population, k=1) # k is number of the first best solutions

    print(best_solutions[0])
    print(best_solutions[0].fitness)
    best_solution = best_solutions[0]
    for i in range(len(best_solution)):
        if best_solution[i] == 1:
            print("Name: ", names[i], " - Price: ", prices[i])

    print(info.select('max'))
    figure = px.line(x=range(0, number_of_generation+1), y=info.select('max'), title="Genetic Algorithm results")
    figure.show()

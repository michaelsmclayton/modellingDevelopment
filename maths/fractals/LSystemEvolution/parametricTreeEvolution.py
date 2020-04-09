import turtle
import numpy as np
from TreeLSystem import TreeLSystem
import matplotlib.pyplot as plt

# Set evolution parameters
populationSize = 20
generations = 30

# ----------------------------------
# Set up evolution
# ----------------------------------
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
def randomIndividual():
    return {
        'ANGLE1': np.random.rand()*1,
        'ANGLE2': np.random.rand()*1,
        'RATE': .2 + np.random.rand()*.4,
        'MIN': np.random.rand()*25,
        'ITERATIONS': np.random.randint(low=2,high=4)
    }
toolbox.register("attr_assign", randomIndividual)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_assign, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    screen.reset() # Reset screen
    lSystem = TreeLSystem(individual, screenSize) # Make new tree
    result = lSystem.run()
    return result['fitness']
def mate(child1, child2):
    parents = [child1, child2]
    children = parents
    for c in range(2):
        for key in child1[0].keys():
            source = np.random.randint(0,2)
            children[c][0][key] = parents[source][0][key]
            if np.random.rand()<.15: # Mutation
                if key in ('ANGLE1', 'ANGLE2', 'MIN'):
                    randomValue = np.random.randn()*5
                if key == 'RATE':
                    randomValue = np.random.randn()*.2
                if key == 'ITERATIONS':
                    randomValue = np.random.randint(low=1,high=4)-2
                children[c][0][key] += randomValue
    return children
toolbox.register("evaluate", evaluate)
toolbox.register("mate", mate)
# toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)

# ----------------------------------
# Run evolution
# ----------------------------------

# Setup screen
screen = turtle.Screen()
screenSize = screen.screensize()
turtle.tracer(0) # have system draw instantly (pen.speed(0))

# Initialise population
print('Setting up initial population...')
pop = toolbox.population(n=populationSize)

# Get initial fitnesses
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses): # Set fitness of each individual
    ind.fitness.values = [fit]
print("  Evaluated %i individuals" % len(pop))

# Iterate over generations
bestIndividual = None
bestFitness = -1
for g in range(generations):
    print('Generation %s...' % (g))

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))

    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        toolbox.mate(child1, child2)
        del child1.fitness.values # fitness values of the children must be recalculated later
        del child2.fitness.values

    # Evaluate the individuals with an invalid fitness (i.e. fitness has been removed above?)
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind) # Evaluate fitness
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = [fit]
    print("  Evaluated %i individuals" % len(invalid_ind))
    
    # The population is entirely replaced by the offspring
    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]
    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5
    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

    # Save best individual
    if np.max(fits) > 0:
        # print('Best fitness = %s' % (bestFitness))
        # if np.max(fits) > bestFitness:
        #   bestFitness = np.max(fits)
        bestIndividual = pop[np.argmax(fits)]
        screen.clear() # Reset screen
        turtle.tracer(0)
        lSystem = TreeLSystem(bestIndividual, screenSize, display=True) # Make new tree
        result = lSystem.run()
        turtle.setpos([0,-.9*screenSize[1]])
        turtle.write('Generation %s' % (g+1), align="center", font=("Arial", 14, "normal"))
        turtle.getscreen().getcanvas().postscript(file='%s.ps' % (g))

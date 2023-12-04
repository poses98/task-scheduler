import math

import numpy as np
def exercise3(seed=0, tasks=0, resources=0, task_duration=[], task_resource=[], task_dependencies=[]):
    """
    Returns the best solution found by the basic genetic algorithm of exercise 3
    :param seed: used to initialize the random number generator
    :param tasks: number of tasks in the task planning problem with resources
    :param resources: number of resources in the task planning problem with resources
    :param task_duration: list of durations of the tasks
    :param task_resource: list of resources required by each task
    :param task_dependencies: list of dependencies (expressed as binary tuples) between tasks
    :return: list with the start time of each task in the best solution found, or empty list if no solution was found
    """
    print("Test Simple gen Alg")
    # Parameter initialization
    import numpy as np

    pop_size = 100
    elitism = 10
    generations = 100
    p_cross = 1.0
    p_mut = 0.05

    fittest_individual, fittest_fitness, generation, best_fitness, mean_fitness = genetic_algorithm(alphabet, tasks, pop_size, generate_random_individual, knapsack_fitness, generation_stop, elitism, roulette_wheel_selection, one_point_crossover, p_cross, uniform_mutation,p_mut, task_duration=task_duration, task_resource=task_resource, task_dependencies=task_dependencies)

    print("Best Individual:")
    print(fittest_individual)
    print("\nBest Individual's Fitness:" + str(fittest_fitness))

    print("\nExercise 3:")

    import matplotlib.pyplot as plt
    x = np.linspace(0, generations + 1, generations + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(pad=5.0)
    ax1.plot(x, best_fitness)
    ax2.plot(x, mean_fitness)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Mean Fitness')


alphabet = [0,1]

def knapsack_fitness(chromosome, *args, **kwargs):
    fitness = 0
    weights=kwargs['weights'] # kwargs stores a dictionary with extra nominal arguments
    values=kwargs['values']
    max_weight =kwargs['max_weight']
    if np.dot(weights, chromosome) <= max_weight:
        fitness = np.dot(values, chromosome)
    return fitness

def generate_random_individual(alphabet, length, *args, **kwargs):
    indices = np.random.randint(0, len(alphabet), length)
    return np.array(alphabet)[indices]

def roulette_wheel_selection(population, fitness, number_parents, *args, **kwargs):
    population_fitness = sum(fitness)
    chromosome_probabilities = [f/population_fitness for f in fitness]
    indices = np.random.choice(range(len(fitness)), number_parents, p=chromosome_probabilities)
    return [population[i] for i in indices]

def one_point_crossover(parent1, parent2, p_cross=0.9):
    if np.random.random() < p_cross:
        point = np.random.randint(1,len(parent1)-1)
        child1 = np.append(parent1[:point],parent2[point:])
        child2 = np.append(parent2[:point],parent1[point:])
        return child1, child2
    else:
        return parent1, parent2

def uniform_mutation(chromosome, p_mut, alphabet):
    child = np.copy(chromosome)
    random_values = np.random.random(len(chromosome))
    mask = random_values < p_mut
    indices = np.random.randint(0, len(alphabet), size=np.count_nonzero(mask))
    child[mask] = np.array(alphabet)[indices]
    return child

def generation_stop(generation, fitness, best_fitness, mean_fitness, *args, **kwargs):
    max_gen=kwargs['max_gen']
    return generation >= max_gen
def genetic_algorithm(alphabet, length, pop_size, generate_individual, fitness, elitism, selection, crossover, p_cross, mutation, p_mut, *args, **kwargs):
    # Population initialization
    population = [generate_individual(alphabet, length, *args, **kwargs) for _ in range(0,100)]
    offspring_size = pop_size - elitism
    generation = 0
    best_fitness = []
    mean_fitness = []

    # First fitness evaluation
    fitness_values = [fitness(x, *args, **kwargs) for x in population]
    best_fitness.append(np.max(fitness_values))
    mean_fitness.append(np.mean(fitness_values))

    # Main loop, checking stopping criteria
    while not generation != 100:

        # Select elite parents
        if elitism > 0:
            indices = np.argpartition(fitness_values, -elitism)[-elitism:]
            elite = [population[i] for i in indices]

        # Select the parents and perform crossover and mutation
        parents = selection(population, fitness_values, offspring_size if (offspring_size%2==0) else offspring_size+1, *args, **kwargs)
        offspring = []
        for k in range(math.ceil(offspring_size/2)):
            parent1 = parents[2*k]
            parent2 = parents[2*k+1]
            child1, child2 = crossover(parent1, parent2, p_cross, *args, **kwargs)
            child1 = mutation(child1, p_mut, alphabet, *args, **kwargs)
            offspring.append(child1)
            if 2*k+1 < offspring_size:
                child2 = mutation(child2, p_mut, alphabet, *args, **kwargs)
                offspring.append(child2)

        # Build new population (replacing)
        if elitism > 0:
            population[:elitism] = elite
            population[elitism:] = offspring

        # Compute fitness of new population
        fitness_values = [fitness(x, *args, **kwargs) for x in population]
        best_fitness.append(np.max(fitness_values))
        mean_fitness.append(np.mean(fitness_values))
        generation += 1

    # Get the fittest individual
    fittest_index = np.where(fitness_values == np.max(fitness_values))[0][0]
    fittest_individual = population[fittest_index]
    fittest_fitness = fitness_values[fittest_index]

    return fittest_individual, fittest_fitness, generation, best_fitness, mean_fitness



def exercise4(seed=0, tasks=0, resources=0, task_duration=[], task_resource=[], task_dependencies=[]):
    """
    Returns the best solution found by the advanced genetic algorithm of exercise 4
    :param seed: used to initialize the random number generator
    :param tasks: number of tasks in the task planning problem with resources
    :param resources: number of resources in the task planning problem with resources
    :param task_duration: list of durations of the tasks
    :param task_resource: list of resources required by each task
    :param task_dependencies: list of dependencies (expressed as binary tuples) between tasks
    :return: list with the start time of each task in the best solution found, or empty list if no solution was found
    """
    print("Test Advanced gen Alg")
    return []

import checks.checkings as checks
import numpy as np
import math

import upmproblems.rcpsp06


def genetic_algorithm(alphabet, length, pop_size, generate_individual, fitness, elitism, selection, crossover, p_cross, mutation, p_mut, *args, **kwargs):
    chromosome = [kwargs['tasks']]
    for i in range(len(chromosome)):
        chromosome[i] = -1
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
    while not generation_stop(generation, max_gen=kwargs['max_gen']):
        # Select elite parents
        if elitism > 0:
            indices = np.argpartition(fitness_values, -elitism)[-elitism:]
            elite = [population[i] for i in indices]

        # Select the parents and perform crossover and mutation
        parents = selection(population, fitness_values, offspring_size if (offspring_size % 2 == 0) else offspring_size + 1, *args, **kwargs)
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

    return fittest_individual, fittest_fitness, generation, best_fitness, mean_fitness, chromosome

def maximum_time (task_duration):
    max_time = 0
    for i in range(len(task_duration)):
        max_time = max_time + task_duration[i]
    return max_time

alphabet = [0, maximum_time(upmproblems.rcpsp06.get_task_duration())]


def individual_fitness(chromosome, *args, **kwargs):
    fitness = 0
    tasks = kwargs['tasks']
    if checks.checkings.checkDependencies(chromosome=chromosome, tasks=tasks, task_duration=kwargs['task_duration']):
        if checks.checkings.checkResources(chromosome=chromosome, tasks=tasks, max_resources = kwargs['max_resources']):
            fitness = calculate_makespan(chromosome, tasks)
    return fitness


def calculate_makespan(chromosome, tasks):
    latest_end = -1
    for actual_task in range(len(chromosome)):
        task_duration = tasks[actual_task]['task_duration']
        end_time = chromosome[actual_task]+task_duration
        if end_time > latest_end:
            latest_end = end_time
    return latest_end


def generate_random_individual(alphabet, length, *args, **kwargs):
    indices = np.random.randint(0, len(alphabet), length)
    return np.array(alphabet)[indices]


def roulette_wheel_selection(population, fitness, number_parents, *args, **kwargs):
    population_fitness = sum(fitness)
    chromosome_probabilities = [f/population_fitness for f in fitness]
    indices = np.random.choice(range(len(fitness)), number_parents, p=chromosome_probabilities)
    return [population[i] for i in indices]


def one_point_crossover(parent1, parent2, p_cross):
    if np.random.random() < p_cross:
        point = np.random.randint(1, len(parent1)-1)
        child1 = np.append(parent1[:point], parent2[point:])
        child2 = np.append(parent2[:point], parent1[point:])
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


def generation_stop(generation, *args, **kwargs):
    max_gen=kwargs['max_gen']
    return generation >= max_gen


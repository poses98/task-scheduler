import checks.checkings as checks
import numpy as np
import math
import upmproblems.rcpsp06


def maximum_time(task_duration):
    max_time = 0
    for i in range(len(task_duration)):
        max_time = max_time + task_duration[i]
    return max_time


alphabet = list(range(maximum_time(upmproblems.rcpsp06.get_task_duration())))


def individual_fitness(chromosome, *args, **kwargs):
    task_duration = kwargs['task_duration']
    task_dependencies = kwargs['task_dependencies']
    task_resource = kwargs['task_resource']
    resources = kwargs['resources']
    if checks.checkDependencies(chromosome, task_duration, task_dependencies):
        if checks.checkResources(chromosome, task_duration, task_resource, resources):
            makespan = calculate_makespan(chromosome, task_duration)
            fitness = 1 / makespan
            return fitness
        else:
            fitness = 2 / calculate_makespan(chromosome, task_duration)
    else:
        fitness = calculate_makespan(chromosome, task_duration)
    return fitness


def calculate_makespan(chromosome, task_duration, *args, **kwargs):
    latest_end = -1
    for actual_task in range(len(chromosome)):
        end_time = chromosome[actual_task] + task_duration[actual_task]
        if end_time > latest_end:
            latest_end = end_time
    return latest_end


def generate_random_individual(alphabet, length, *args, **kwargs):
    indices = np.random.randint(0, len(alphabet), length)
    return np.array(alphabet)[indices]


def roulette_wheel_selection(population, fitness, number_parents, *args, **kwargs):
    population_fitness = sum(fitness)
    if population_fitness == 0:
        indices = np.random.choice(range(len(fitness)), number_parents)
    else:
        chromosome_probabilities = [f / population_fitness for f in fitness]
        indices = np.random.choice(range(len(fitness)), number_parents, p=chromosome_probabilities)
    return [population[i] for i in indices]


def one_point_crossover(parent1, parent2, p_cross, *args, **kwargs):
    if np.random.random() < p_cross:
        point = np.random.randint(1, len(parent1) - 1)
        child1 = np.append(parent1[:point], parent2[point:])
        child2 = np.append(parent2[:point], parent1[point:])
        return child1, child2
    else:
        return parent1, parent2


def uniform_mutation(chromosome, p_mut, alphabet, *args, **kwargs):
    child = np.copy(chromosome)
    random_values = np.random.random(len(chromosome))
    mask = random_values < p_mut
    indices = np.random.randint(0, len(alphabet), size=np.count_nonzero(mask))
    child[mask] = np.array(alphabet)[indices]
    return child


def generation_stop(generation, **kwargs):
    max_gen = kwargs['max_gen']
    return generation >= max_gen


def genetic_algorithm(alphabet, length, pop_size, generate_individual, fitness,
                      generation_stop, selection, crossover, p_cross, mutation, p_mut, *args, **kwargs):
    # genetic_algorithm(alphabet, tasks, pop_size, generate_random_individual, individual_fitness, genetation_stop, roulette_wheel_selection, one_point_crossover, p_cross, uniform_mutation, p_mut, task_duration=task_duration, task_resource=task_resource, task_dependencies=task_dependencies, resources = resources)
    # Population initialization
    population = [generate_individual(alphabet, length, *args, **kwargs) for _ in range(0, 100)]

    offspring_size = pop_size
    generation = 0
    best_fitness = []
    mean_fitness = []

    # First fitness evaluation
    fitness_values = [fitness(x, *args, **kwargs) for x in population]
    best_fitness.append(np.max(fitness_values))
    mean_fitness.append(np.mean(fitness_values))

    # Main loop, checking stopping criteria
    while not generation_stop(generation, max_gen=kwargs['max_gen']):
        # Select the parents and perform crossover and mutation
        parents = selection(population, fitness_values,
                            offspring_size if (offspring_size % 2 == 0) else offspring_size + 1, *args, **kwargs)
        offspring = []

        for k in range(math.ceil(offspring_size / 2)):
            parent1 = parents[2 * k]
            parent2 = parents[2 * k + 1]
            child1, child2 = crossover(parent1, parent2, p_cross, *args, **kwargs)
            child1 = mutation(child1, p_mut, alphabet, *args, **kwargs)
            offspring.append(child1)
            if 2 * k + 1 < offspring_size:
                child2 = mutation(child2, p_mut, alphabet, *args, **kwargs)
                offspring.append(child2)
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

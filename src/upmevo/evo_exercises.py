import numpy as np
import checks.checkings
import matplotlib.pyplot as plt
from checks import checkings as checks
import math

def maximum_time(task_duration):
    max_time = 0
    for i in range(len(task_duration)):
        max_time = max_time + task_duration[i]
    return max_time



def individual_fitness(chromosome, *args, **kwargs):
    task_duration = kwargs['task_duration']
    task_dependencies = kwargs['task_dependencies']
    task_resource = kwargs['task_resource']
    resources = kwargs['resources']
    fitness = 0

    if checks.checkDependencies(chromosome, task_duration, task_dependencies):
        if checks.checkResources(chromosome, task_duration, task_resource, resources):
            makespan = calculate_makespan(chromosome, task_duration)
            fitness = 1 / makespan
            return fitness

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
    individual = np.array(alphabet)[indices]
    while checks.checkDependencies(individual, task_duration=kwargs['task_duration'], task_dependencies=kwargs['task_dependencies']) == False & checks.checkResources(individual,task_duration=kwargs['task_duration'],task_resource=kwargs['task_resource'], resources=kwargs['resources']) == False:
        indices = np.random.randint(0, len(alphabet), length)
        individual = np.array(alphabet)[indices]
    return individual


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



def swap_mutation(individual, p_mutate, *args, **kwargs):
    mutated_individual = individual.copy()

    for i in range(len(mutated_individual)):
        if np.random.random() < p_mutate:
            # Select a random position to swap with
            swap_position = np.random.randint(0, len(mutated_individual))

            # Swap the values
            mutated_individual[i], mutated_individual[swap_position] = (
                mutated_individual[swap_position],
                mutated_individual[i],
            )

    return mutated_individual


def generation_stop(generation, *args, **kwargs):
    max_gen = kwargs['max_gen']
    return generation >= max_gen



def genetic_algorithm(alphabet, length, pop_size, generate_individual, fitness,
                      generation_stop, selection, crossover, p_cross, mutation, p_mut, *args, **kwargs):
    # genetic_algorithm(alphabet, tasks, pop_size, generate_random_individual, individual_fitness, genetation_stop, roulette_wheel_selection, one_point_crossover, p_cross, uniform_mutation, p_mut, task_duration=task_duration, task_resource=task_resource, task_dependencies=task_dependencies, resources = resources)
    # Population initialization
    np.random.seed(kwargs['seed'])
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

def exercise3(seed, tasks, resources, task_duration=[], task_resource=[], task_dependencies=[]):
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
    alphabet = list(range(maximum_time(task_duration)))
    pop_size = 100
    max_gen = 100
    p_cross = 0.9
    p_mut = 0.1
    fittest_individual, fittest_fitness, generation, best_fitness, mean_fitness = genetic_algorithm(alphabet, tasks, pop_size, generate_random_individual, individual_fitness, generation_stop, roulette_wheel_selection, one_point_crossover, p_cross, uniform_mutation, p_mut, task_duration=task_duration, task_resource=task_resource, task_dependencies=task_dependencies, resources=resources, max_gen=max_gen, seed=seed)
    makespan = calculate_makespan(fittest_individual, task_duration)
    print("Best Individual:")
    print(fittest_individual)
    print("\nBest Individual's Fitness:" + str(fittest_fitness))
    print('\n Makespan: ' + str(makespan))

    return fittest_individual


def tournament_selection(population, fitness, number_parents, tournament_size=3, *args, **kwargs):
    selected_parents = []

    for _ in range(number_parents):
        # Randomly select individuals for the tournament
        tournament_indices = np.random.choice(range(len(population)), size=tournament_size, replace=False)

        # Choose the individual with the highest fitness from the tournament
        winner_index = max(tournament_indices, key=lambda i: fitness[i])

        # Add the winner to the selected parents
        selected_parents.append(population[winner_index])

    return selected_parents


def exercise4(seed=1234567890, tasks=0, resources=0, task_duration=[], task_resource=[], task_dependencies=[], *args,
              **kwargs):
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
    np.random.seed(seed)
    # Parameter initialization
    pop_size = 150
    elitism = 10
    generations = 150
    p_cross = 0.9
    p_mut = 0.15

    fittest_individual, fittest_fitness, generation, best_fitness, mean_fitness = adv_genetic_algorithm(
        alphabet=range(sum(task_duration)),
        length=tasks,
        pop_size=pop_size,
        generate_individual=generate_random_individual_w_zero,
        fitness=scheduling_fitness,
        stopping_criteria=generation_stop,
        elitism=elitism,
        selection=tournament_selection,
        crossover=one_point_crossover,
        p_cross=p_cross,
        mutation=uniform_mutation,
        p_mut=p_mut,
        task_duration=task_duration,
        task_resources=task_resource,
        resources=resources,
        task_dependencies=task_dependencies,
        max_gen=generations)

    if kwargs['plot']:
        # Display results
        print("Best Individual:")
        print(fittest_individual)
        print("\nBest Individual's Fitness:" + str(fittest_fitness))

        x = np.linspace(0, generations, generations + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.tight_layout(pad=5.0)
        ax1.plot(x, best_fitness)
        ax2.plot(x, mean_fitness)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Best Fitness')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Mean Fitness')
        plt.show()

    return fittest_fitness, fittest_individual


def adv_genetic_algorithm(alphabet, length, pop_size, generate_individual, fitness, stopping_criteria, elitism,
                          selection, crossover, p_cross, mutation, p_mut, *args, **kwargs):
    # Population initialization
    population = [generate_individual(alphabet, length, *args, **kwargs) for _ in range(pop_size)]
    offspring_size = pop_size - elitism
    generation = 0
    best_fitness = []
    mean_fitness = []

    # First fitness evaluation
    fitness_values = [fitness(x, *args, **kwargs) for x in population]
    # Min fitness value in a population
    best_fitness.append(np.min(fitness_values))
    # Average fitness value in population
    mean_fitness.append(np.mean(fitness_values))

    # Main loop, checking stopping criteria
    while not stopping_criteria(generation, fitness_values, best_fitness, mean_fitness, *args, **kwargs):

        # Select elite parents
        if elitism > 0:
            indices = np.argpartition(fitness_values, elitism)[:elitism]
            elite = [population[i] for i in indices]

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

        # Build new population (replacing)
        if elitism > 0:
            population[:elitism] = elite
            population[elitism:] = offspring

        # Compute fitness of new population
        fitness_values = [fitness(x, *args, **kwargs) for x in population]
        best_fitness.append(np.min(fitness_values))
        mean_fitness.append(np.mean(fitness_values))
        generation += 1

    # Get the fittest individual
    fittest_index = np.where(fitness_values == np.min(fitness_values))[0][0]
    fittest_individual = population[fittest_index]
    fittest_fitness = fitness_values[fittest_index]

    return fittest_individual, fittest_fitness, generation, best_fitness, mean_fitness


def scheduling_fitness(schedule, *args, **kwargs):
    max_resources = kwargs['resources']  # Maximum number of resources available
    task_duration = kwargs['task_duration']
    task_resources = kwargs['task_resources']
    task_number = len(task_duration)
    task_dependencies = kwargs['task_dependencies']
    max_duration = sum(task_duration)
    # Check for valid schedule based on dependencies
    if not checks.checkings.checkDependencies(schedule, task_duration, task_dependencies):
        return max_duration * 2 * task_number
    # Check for valid schedule based on resource constraints
    if not checks.checkings.checkResources(chromosome=schedule, task_duration=task_duration,
                                           task_resource=task_resources, resources=max_resources):
        return max_duration * 3 * task_number
    # Calculate makespan as the fitness value

    makespan = calculate_makespan_adv(schedule, task_duration)
    return makespan


def generate_random_individual_w_zero(alphabet, length, *args, **kwargs):
    indices = np.random.randint(0, len(alphabet), length)
    individual = np.array(alphabet)[indices]

    # Heuristic: Ensure at least one individual is 0
    if 0 not in individual:
        zero_index = np.random.randint(0, length)
        individual[zero_index] = 0

    return individual


def calculate_makespan_adv(chromosome, tasks_duration):
    latest_end = -1
    for actual_task in range(len(chromosome)):
        task_duration = tasks_duration[actual_task]
        end_time = chromosome[actual_task] + task_duration
        if end_time > latest_end:
            latest_end = end_time
    return latest_end

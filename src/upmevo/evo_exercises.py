import math
import numpy as np
import checks.checkings
import matplotlib.pyplot as plt


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
    np.random.seed(0)
    pop_size = 100
    elitism = 10
    generations = 100
    p_cross = 0.9
    p_mut = 0.1
    max_gen = 100

    fittest_individual, fittest_fitness, generation, best_fitness, mean_fitness, chromosome = genetic_algorithm(
        alphabet, tasks, pop_size, generate_random_individual, individual_fitness(), generation_stop, elitism,
        roulette_wheel_selection, one_point_crossover, p_cross, uniform_mutation, p_mut, max_gen=max_gen,
        task_duration=task_duration, task_resource=task_resource, task_dependencies=task_dependencies)

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

    return chromosome


alphabet = [0, 100]


def individual_fitness(chromosome, *args, **kwargs):
    fitness = 0
    tasks = kwargs['tasks']
    if checks.checkings.checkDependencies(chromosome=chromosome, tasks=tasks, task_duration=kwargs['task_duration']):
        if checks.checkings.checkResources(chromosome=chromosome, tasks=tasks, max_resources=kwargs['max_resources']):
            fitness = calculate_makespan(chromosome, tasks)
    return fitness


def calculate_makespan(chromosome, tasks):
    latest_end = -1
    for actual_task in range(len(chromosome)):
        task_duration = tasks[actual_task]['task_duration']
        end_time = chromosome[actual_task] + task_duration
        if end_time > latest_end:
            latest_end = end_time
    return latest_end


def generate_random_individual(alphabet, length, *args, **kwargs):
    indices = np.random.randint(0, len(alphabet), length)
    return np.array(alphabet)[indices]


def roulette_wheel_selection(population, fitness, number_parents, *args, **kwargs):
    population_fitness = sum(fitness)
    chromosome_probabilities = [f / population_fitness for f in fitness]
    indices = np.random.choice(range(len(fitness)), number_parents, p=chromosome_probabilities)
    return [population[i] for i in indices]


def one_point_crossover(parent1, parent2, p_cross, *args, **kwargs):
    print(parent1)
    print(parent2)
    if np.random.random() < p_cross:
        point = np.random.randint(1, len(parent1) - 1)
        child1 = np.append(parent1[:point], parent2[point:])
        child2 = np.append(parent2[:point], parent1[point:])
        return child1, child2
    else:
        return parent1, parent2


def pmx_crossover(parent1, parent2, p_cross, *args, **kwargs):
    if np.random.random() < p_cross:
        size = len(parent1)
        point1 = np.random.randint(0, size - 1)
        point2 = np.random.randint(point1 + 1, size)

        # Initialize the children as copies of the parents
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Perform PMX crossover
        for i in range(point1, point2):
            # Swap the values between the two parents
            temp1, temp2 = child1[i], child2[i]
            child1[i], child2[i] = temp2, temp1

            # Update the mapping for the swapped values
            index1 = np.where(child1 == temp2)[0][0]
            index2 = np.where(child2 == temp1)[0][0]
            child1[index1], child2[index2] = temp1, temp2

        return child1, child2
    else:
        return parent1.copy(), parent2.copy()


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


def genetic_algorithm(alphabet, length, pop_size, generate_individual, fitness, elitism, selection, crossover, p_cross,
                      mutation, p_mut, *args, **kwargs):
    chromosome = [kwargs['tasks']]
    for i in range(len(chromosome)):
        chromosome[i] = -1
    # Population initialization
    population = [generate_individual(alphabet, length, *args, **kwargs) for _ in range(0, 100)]
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
        best_fitness.append(np.max(fitness_values))
        mean_fitness.append(np.mean(fitness_values))
        generation += 1

    # Get the fittest individual
    fittest_index = np.where(fitness_values == np.max(fitness_values))[0][0]
    fittest_individual = population[fittest_index]
    fittest_fitness = fitness_values[fittest_index]

    return fittest_individual, fittest_fitness, generation, best_fitness, mean_fitness, chromosome


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
    print("Advanced Genetic Algorithm")
    np.random.seed(1234567890)
    # Parameter initialization
    pop_size = 150
    elitism = 10
    generations = 100
    p_cross = 1.0
    p_mut = 0.15

    fittest_individual, fittest_fitness, generation, best_fitness, mean_fitness = adv_genetic_algorithm(
        alphabet=range(tasks),
        length=tasks,
        pop_size=pop_size,
        generate_individual=generate_random_individual,
        fitness=scheduling_fitness,
        stopping_criteria=generation_stop,
        elitism=elitism,
        selection=tournament_selection,
        crossover=pmx_crossover,
        p_cross=p_cross,
        mutation=swap_mutation,
        p_mut=p_mut,
        task_duration=task_duration,
        task_resources=task_resource,
        resources=resources,
        task_dependencies=task_dependencies,
        max_gen=generations)
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
            indices = np.argpartition(fitness_values, -elitism)[-elitism:]
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
    task_dependencies = kwargs['task_dependencies']

    # Check for valid schedule based on dependencies
    if not check_dependencies(schedule, task_duration, task_dependencies):
        return 1000
    # Check for valid schedule based on resource constraints
    if not check_resources(schedule, task_duration, task_resource=task_resources, max_resources=max_resources):
        return 1000
    # Calculate makespan as the fitness value
    makespan = calculate_makespan_adv(schedule, task_duration)
    return -makespan


def calculate_makespan_adv(chromosome, tasks_duration):
    latest_end = -1
    for actual_task in range(len(chromosome)):
        task_duration = tasks_duration[actual_task]
        end_time = chromosome[actual_task] + task_duration
        if end_time > latest_end:
            latest_end = end_time
    return latest_end


def check_dependencies(schedule, task_duration, task_dependencies):
    for schedule_index in range(len(schedule)):
        task_start_time = schedule[schedule_index]
        task_id = schedule_index + 1  # Tasks are 1-indexed
        current_task_duration = task_duration[schedule_index]

        # Check if dependencies are satisfied for the current task
        for dependency in task_dependencies:
            dependent_task, current_task = dependency
            if current_task == task_id:
                dependent_time_completed = schedule[dependent_task - 1] + task_duration[dependent_task - 1]
                if dependent_time_completed > task_start_time:
                    return False
    return True



def check_resources(schedule, task_duration, task_resource, max_resources, *args, **kwargs):
    for schedule_index in range(len(schedule)):
        task_index = schedule_index
        current_task_resources = task_resource[task_index]
        current_task_duration = task_duration[task_index]
        task_start_time = schedule[schedule_index]

        for conflictive_index in range(len(schedule)):
            same_index = conflictive_index == schedule_index
            executing_same_time = (task_start_time <= schedule[
                conflictive_index] < task_start_time + current_task_duration) or schedule[
                                      conflictive_index] <= task_start_time < current_task_duration + schedule[
                                      conflictive_index]
            if not same_index and executing_same_time:
                if current_task_resources + task_resource[conflictive_index] > max_resources:
                    return False
    return True

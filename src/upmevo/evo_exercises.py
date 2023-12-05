import numpy as np
from upmevo import *
from upmproblems import rcpsp06 as p6


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

    fittest_individual, fittest_fitness, generation, best_fitness, mean_fitness = genetic_algorithm(alphabet, tasks, pop_size, generate_random_individual, individual_fitness, generation_stop, elitism, roulette_wheel_selection, one_point_crossover, p_cross, uniform_mutation, p_mut, max_gen=max_gen, task_duration=task_duration, task_resource=task_resource, task_dependencies=task_dependencies)

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

    return fittest_individual

print(exercise3(3, p6.get_tasks(), p6.get_resources(), task_duration= p6.get_task_duration(), task_resource=p6.get_task_resource(), task_dependencies=p6.get_task_dependencies()))

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

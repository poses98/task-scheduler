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
    pop_size = 100
    max_gen = 100
    p_cross = 0.9
    p_mut = 0.1

    fittest_individual, fittest_fitness, generation, best_fitness, mean_fitness = genetic_algorithm(alphabet, tasks, pop_size, generate_random_individual, individual_fitness, generation_stop, roulette_wheel_selection, one_point_crossover, p_cross, uniform_mutation, p_mut, task_duration=task_duration, task_resource=task_resource, task_dependencies=task_dependencies, resources=resources, max_gen=max_gen)

    print("Best Individual:")
    print(fittest_individual)
    print("\nBest Individual's Fitness:" + str(fittest_fitness))

    return fittest_individual

exercise3(0, tasks=p6.get_tasks(), resources=p6.get_resources(), task_duration= p6.get_task_duration(), task_resource=p6.get_task_resource(), task_dependencies=p6.get_task_dependencies())


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

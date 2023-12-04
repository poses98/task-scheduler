import checks.checkings as checks


def exercise1(tasks=0, resources=0, task_duration=[], task_resource=[], task_dependencies=[]):
    """
    Returns the best solution found by the branch and bound algorithm of exercise 1
    :param tasks: number of tasks in the task planning problem with resources
    :param resources: number of resources in the task planning problem with resources
    :param task_duration: list of durations of the tasks
    :param task_resource: list of resources required by each task
    :param task_dependencies: list of dependencies (expressed as binary tuples) between tasks
    :return: list with the start time of each task in the best solution found, or empty list if no solution was found
    """
    print("Test B&B")
    return []


def exercise2(tasks=0, resources=0, task_duration=[], task_resource=[], task_dependencies=[]):
    """
    Returns the best solution found by the A* algorithm of exercise 2
    :param tasks: number of tasks in the task planning problem with resources
    :param resources: number of resources in the task planning problem with resources
    :param task_duration: list of durations of the tasks
    :param task_resource: list of resources required by each task
    :param task_dependencies: list of dependencies (expressed as binary tuples) between tasks
    :return: list with the start time of each task in the best solution found, or empty list if no solution was found
    """
    print("Test A*")

    return []

def selectDecision(chromosome):
    best_heuristic = 0
    selected_task = -1
    for i in range(len(chromosome)):
        actual_heuristic = heuristic_function(i, chromosome=chromosome)
        if actual_heuristic > best_heuristic:
            selected_task = chromosome[i]
    return selected_task

def heuristic_function (position, **kwargs):
    heuristic = 0
    if checks.checkDependencies(kwargs= ['chromosome'], task_dependencies=kwargs['tasks_dependencies'], task_duration=kwargs['task_duration']):
        if checks.checkResources(kwargs= ['chromosome'], task_duration=kwargs['task_duration'], task_resource=kwargs['task_resource']):
            heuristic = heuristic_proposal1(kwargs=['chromosome'])
    return heuristic

def heuristic_proposal1 (**kwargs):
    max_duration = -1
    task_duration = kwargs['task_duration']
    for current_task in range(len(kwargs= ['chromosome'])):
        if task_duration[current_task] > max_duration:
            max_duration = task_duration[current_task]
    return max_duration
def astar(chromosome, *args, **kwargs):
    for i in range(len(chromosome)):
        chromosome[i] = -1
    for time in range(len(chromosome)):
        task = selectDecision(chromosome)
        chromosome[task] = time

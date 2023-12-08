def exercise1(tasks=0, resources=0, task_duration=[], task_resource=[], task_dependencies=[]):
    """
    Main function for exercise 1.
    Parameters:
    - tasks: int, number of tasks.
    - resources: int, number of resources.
    - task_duration: list of task durations.
    - task_resource: list of resource requirements for each task.
    - task_dependencies: list of task dependency tuples.
    Returns:
    - list: Availability schedule for each task.
    """

    availability = [-1] * tasks
    achievesDependencies = [True] * tasks
    branch_and_bound(availability, achievesDependencies, tasks=tasks, resources=resources, task_duration=task_duration,
                     task_resource=task_resource, task_dependencies=task_dependencies)
    return availability


"-----------------------------------------------------------------------------------------------------------------------------------"
"Additional methods for exercise 1"


def calculate_makespanBnB(task_duration, *args, **kwargs):
    """
    Calculates the makespan of a schedule, used during execution.
    Parameters:
    - task_duration: list of task durations.
    Returns:
    - int: makespan of the schedule.
    """

    end_time = -1
    for actual_task in range(len(task_duration)):
        end_time += task_duration[actual_task]
    return end_time


def updateAchievedBnB(achievesDependencies, availability, i, task_duration, task_dependencies):
    """
    Updates the list of achieved dependencies for each task.
    Parameters:
    - achievesDependencies: list of booleans indicating whether each task has its dependencies fulfilled.
    - availability: list of task start times.
    - i: current time instant.
    - task_duration: list of task durations.
    - task_dependencies: list of task dependency tuples.
    Returns:
    - list: Updated list of achieved dependencies.
    """

    for task in range(len(achievesDependencies)):
        achievesDependencies[task] = fulfillsDependenciesBnB(i, availability, task_duration, task_dependencies, task)
    return achievesDependencies


def fulfillsDependenciesBnB(i, availability, task_duration, task_dependencies, task):
    """
    Checks if all dependencies of a task are fulfilled at a given time instant.
    Parameters:
    - i: current time instant.
    - availability: list of task start times.
    - task_duration: list of task durations.
    - task_dependencies: list of task dependency tuples.
    - task: index of the current task.
    Returns:
    - boolean: True if all dependencies are fulfilled, False otherwise.
    """

    fulfills = True
    for dependency in task_dependencies:
        (dependent_task, current_task) = dependency
        if current_task == task + 1:
            dependent_task_start_time = availability[dependent_task - 1]
            if dependent_task_start_time == -1:
                fulfills = False
            else:
                if (dependent_task_start_time + task_duration[dependent_task - 1]) > i:
                    fulfills = False
    return fulfills


def checkResourcesBnB(availability, achievesDependencies, task_duration, task_resource, resources, **kwargs):
    """
    Checks resource availability for each time instant in the schedule.
    Parameters:
    - availability: list of task start times.
    - achievesDependencies: list of achieved dependencies for each task.
    - task_duration: list of task durations.
    - task_resource: list of resource requirements for each task.
    - resources: total number of resources.
    Returns:
    - list: Updated availability schedule.
    """

    for instant in range(calculate_makespanBnB(task_duration)):
        used_resources = 0
        for span_task in range(len(achievesDependencies)):
            if availability[span_task] != -1:
                if (instant >= availability[span_task]) & (
                        instant < (availability[span_task] + task_duration[span_task])):
                    used_resources += task_resource[span_task]
                if used_resources > resources:
                    availability = bestCostBnB(instant, availability, task_duration,
                                               task_resource,
                                               resources, used_resources, **kwargs)
    return availability


def bestCostBnB(instant, availability, achievesDependencies, task_duration, task_resource, resources, used_resources,
                **kwargs):
    """
        Finds the task with the best cost and removes it from the schedule if resource usage exceeds the limit.
        Parameters:
        - instant: current time instant.
        - availability: list of task start times.
        - achievesDependencies: list of achieved dependencies for each task.
        - task_duration: list of task durations.
        - task_resource: list of resource requirements for each task.
        - resources: total number of resources.
        - used_resources: current total resources in use.
        Returns:
        - list: Updated availability schedule.
        """

    worst_cost = 1000
    pos = -1
    while used_resources > resources:
        for i in range(len(availability)):
            if availability[i] != -1:
                if (instant >= availability[i]) & (instant < (availability[i] + task_duration[i])):
                    current = calculateCostBnB(i, task_duration, task_resource, **kwargs)
                    if current < worst_cost:
                        worst_cost = current
                        pos = i
        availability[pos] = -1
        used_resources -= task_resource[pos]
    return availability


def dependentTasksBnB(i, task_dependencies):
    """
    Counts the number of tasks dependent on a given task.
    Parameters:
    - i: index of the current task.
    - task_dependencies: list of task dependency tuples.
    Returns:
    - int: Number of dependent tasks.
    """

    counter = 0
    for x in range(len(task_dependencies)):
        for dependency in task_dependencies:
            (dependent_task, current_task) = dependency
            if dependent_task == i + 1:
                counter += 1
    return counter


def calculateCostBnB(i, task_duration, task_resource, **kwargs):
    """
    Calculates the cost of a task based on resource usage and number of dependent tasks.
    Parameters:
    - i: index of the current task.
    - task_duration: list of task durations.
    - task_resource: list of resource requirements for each task.
    - kwargs: additional parameters, including task dependencies.
    Returns:
    - float: Cost of the task.
    """

    tasks = kwargs["tasks"]
    return (task_resource[i] / task_duration[i]) + dependentTasksBnB(i, task_dependencies=kwargs["task_dependencies"])


def modifyAvailabilityBnB(achievesDependencies, availability, best):
    """
    Modifies the availability schedule based on achieved dependencies.
    Parameters:
    - achievesDependencies: list of achieved dependencies for each task.
    - availability: list of task start times.
    - best: current time instant.
    Returns:
    - list: Updated availability schedule.
    """

    for i in range(len(achievesDependencies)):
        if availability[i] == -1:
            if achievesDependencies[i]:
                availability[i] = best
    return availability


def branch_and_bound(availability, achievesDependencies, **kwargs):
    """
    Branch-and-bound algorithm for scheduling tasks.
    Parameters:
    - availability: list of task start times.
    - achievesDependencies: list of achieved dependencies for each task.
    - kwargs: additional parameters, including task durations, resources, etc.
    """

    for i in range(calculate_makespanBnB(task_duration=kwargs['task_duration'])):
        achievesDependencies = updateAchievedBnB(achievesDependencies, availability, i,
                                                 task_duration=kwargs["task_duration"],
                                                 task_dependencies=kwargs["task_dependencies"])
        availability = modifyAvailabilityBnB(achievesDependencies, availability, i)
        availability = checkResourcesBnB(availability, achievesDependencies, task_duration=kwargs['task_duration'],
                                         task_resource=kwargs['task_resource'], resources=kwargs['resources'],
                                         task_dependencies=kwargs["task_dependencies"], tasks=kwargs['tasks'])


def result_makespan(availability_schedule, task_duration):
    """
    Finds the highest value in the availability schedule and its position.
    Parameters:
    - availability_schedule: list of task start times.
    - task_duration: list of task durations.
    """

    highest_value = max(availability_schedule)
    pos = availability_schedule.index(highest_value)
    return highest_value + task_duration[pos]


def calculate_makespan(task_duration, *args, **kwargs):
    end_time = -1
    for actual_task in range(len(task_duration)):
        end_time += task_duration[actual_task]
    return end_time


def checkDependenciesAs(current_fulfillsDependencies, current_availability, index, **kwargs):
    dependency = kwargs['task_dependencies']
    task_duration = kwargs['task_duration']
    new_fulfillsDependencies = list(current_fulfillsDependencies)
    for dependent_task, current_task in dependency:
        if current_task == index + 1:
            dependent_end_time = current_availability[dependent_task - 1] + task_duration[dependent_task - 1]
            if dependent_end_time > current_availability[index]:
                new_fulfillsDependencies[index] = False

    return new_fulfillsDependencies

def checkResourcesAs(avaliability, fulfillsDependencies, task_duration, task_resource, resources, **kwargs):
    for instant in range(calculate_makespan(task_duration)):
        used_resources = 0
        for span_task in range(len(fulfillsDependencies)):
            if avaliability[span_task] != -1:
                if (instant >= avaliability[span_task]) and (
                        instant < (avaliability[span_task] + task_duration[span_task])):
                    used_resources += task_resource[span_task]
                if used_resources > resources:
                    # If resources are exceeded, try to reschedule the task
                    avaliability = bestCost(instant, avaliability, task_duration, task_resource,
                                            resources, used_resources, **kwargs)
    return avaliability


def bestCost(instant, avaliability, task_duration, task_resource, resources, used_resources, **kwargs):
    peor_coste = 1000
    pos = -1
    while used_resources > resources:
        for i in range(len(avaliability)):
            if avaliability[i] != -1:
                if (instant >= avaliability[i]) & (instant < (avaliability[i] + task_duration[i])):
                    actual = calculateCost(i, task_duration, task_resource, **kwargs)
                    if actual < peor_coste:
                        peor_coste = actual
                        pos = i
        avaliability[pos] = -1
        used_resources -= task_resource[pos]
    return avaliability


def dependent_tasks(i, task_dependencies):
    contador = 0
    for dependency in task_dependencies:
        (dependent_task, current_task) = dependency
        if dependent_task == i + 1:
            contador += 1
    return contador


def calculateCost(i, task_duration, task_resource, **kwargs):
    return (task_resource[i] / task_duration[i]) + dependent_tasks(i, task_dependencies=kwargs["task_dependencies"])


def updateAvaliability(fulfillsDependencies, avaliability, best):
    for i in range(len(fulfillsDependencies)):
        if avaliability[i] == -1 and fulfillsDependencies[i]:
            # Schedule the task at the earliest possible time
            avaliability[i] = best
    return avaliability


# Heuristic function
def optimal_heuristic(avaliability, task_duration, task_resource, task_dependencies, resources):
    # Heuristic part 1: Earliest possible start time for unscheduled tasks
    h1 = 0
    for task, start_time in enumerate(avaliability):
        if start_time == -1:
            # Find the earliest time the task can start based on dependencies
            earliest_start = 0
            for dependent_task, current_task in task_dependencies:
                if current_task == task + 1:
                    dependent_end_time = avaliability[dependent_task - 1] + task_duration[dependent_task - 1]
                    earliest_start = max(earliest_start, dependent_end_time)
            h1 += earliest_start

    # Heuristic part 2: Resource constraint penalty
    h2 = 0
    for task, start_time in enumerate(avaliability):
        if start_time == -1:
            # Estimate the additional time needed due to resource constraints
            h2 += task_resource[task] / resources * task_duration[task]

    # Combine the heuristics
    heuristic_value = h1 + h2
    return heuristic_value


def a_star(best_solution, fulfillsDependencies, **kwargs):
    open_set = [(optimal_heuristic(best_solution, kwargs['task_duration'], kwargs['task_resource'],
                                   kwargs['task_dependencies'], kwargs['resources']), best_solution,
                 fulfillsDependencies)]
    best_solution = None
    while open_set:
        # Find the state with the lowest f-cost
        current_state = min(open_set, key=lambda x: x[0])
        open_set.remove(current_state)
        current_f_cost, current_avaliability, current_fulfillsDependencies = current_state

        # Check if the goal has been reached
        if all(current_fulfillsDependencies):
            if best_solution is None or current_f_cost < best_solution[0]:
                best_solution = current_avaliability

        for i in range(calculate_makespan(task_duration=kwargs['task_duration'])):
            new_fulfillsDependencies = checkDependenciesAs(current_fulfillsDependencies, current_avaliability, i,
                                                        task_duration=kwargs["task_duration"],
                                                        task_dependencies=kwargs["task_dependencies"])
            new_disponibilidad = updateAvaliability(new_fulfillsDependencies, current_avaliability, i)
            new_disponibilidad = checkResourcesAs(new_disponibilidad, new_fulfillsDependencies,
                                                  task_duration=kwargs['task_duration'],
                                                  task_resource=kwargs['task_resource'], resources=kwargs['resources'],
                                                  task_dependencies=kwargs["task_dependencies"], tasks=kwargs['tasks'])

            # Skip this state if not all dependencies are met
            if not all(new_fulfillsDependencies):
                continue

            # Calculate the f-cost for the new state
            h_cost = optimal_heuristic(new_disponibilidad, kwargs['task_duration'], kwargs['task_resource'],
                                       kwargs['task_dependencies'], kwargs['resources'])
            f_cost = calculate_makespan(task_duration=kwargs['task_duration'],
                                        disponibilidad=new_disponibilidad) + h_cost

            # Add the new state to the open set
            open_set.append((f_cost, new_disponibilidad, new_fulfillsDependencies))

    return best_solution  # If the open set is empty and the goal was not reached, return None


def exercise2(tasks, resources, task_duration=[], task_resource=[], task_dependencies=[]):
    fulfillsDependencies = [True] * tasks
    best_solution = [-1] * tasks
    best_solution = a_star(best_solution, fulfillsDependencies, tasks=tasks, resources=resources,
                          task_duration=task_duration,
                          task_resource=task_resource, task_dependencies=task_dependencies)
    return best_solution

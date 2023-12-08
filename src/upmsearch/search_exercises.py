
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
                    availability = bestCostBnB(instant, availability, achievesDependencies, task_duration, task_resource,
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


"-----------------------------------------------------------------------------------------------------------------------------------"
"-----------------------------------------------------------------------------------------------------------------------------------"
"TRANSLATED"
"NON-TRANSLATED"
def exercise2(tasks=0, resources=0, task_duration=[], task_resource=[], task_dependencies=[]):
    disponibilidad = [-1] * tasks
    cumpleDependencia = [True] * tasks
    a_star(disponibilidad, cumpleDependencia, tasks=tasks, resources=resources, task_duration=task_duration,
           task_resource=task_resource, task_dependencies=task_dependencies)
    return disponibilidad

def calculate_makespan(task_duration, *args, **kwargs):
    end_time = -1
    for actual_task in range(len(task_duration)):
        end_time += task_duration[actual_task]
    return end_time

def actualizarCumplido(cumpleDependencias, disponibilidad, i, task_duration, task_dependencies):
    for task in range(len(cumpleDependencias)):
        cumpleDependencias[task] = checkDependenciasAs(i, disponibilidad, task_duration, task_dependencies, task)
    return cumpleDependencias

def checkDependenciasAs(i, disponibilidad, task_duration, task_dependencies, task):
    cumple = True
    for dependency in task_dependencies:
        (dependent_task, current_task) = dependency
        if current_task == task + 1:
            dependent_task_start_time = disponibilidad[dependent_task - 1]
            if dependent_task_start_time == -1:
                cumple = False
            else:
                if (dependent_task_start_time + task_duration[dependent_task-1]) > i:
                    cumple = False
    return cumple

def checkResources(disponibilidad, cumpleDependencias, task_duration, task_resource, resources, **kwargs):
    for instant in range(calculate_makespan(task_duration)):
        used_resources = 0
        for span_task in range(len(cumpleDependencias)):
            if disponibilidad[span_task] != -1:
                if (instant >= disponibilidad[span_task]) and (instant < (disponibilidad[span_task] + task_duration[span_task])):
                    used_resources += task_resource[span_task]
                if used_resources > resources:
                    # If resources are exceeded, try to reschedule the task
                    disponibilidad = mejorCoste(instant, disponibilidad, cumpleDependencias, task_duration, task_resource, resources, used_resources, **kwargs)
    return disponibilidad
def mejorCoste(instant, disponibilidad, cumpleDependencias, task_duration, task_resource, resources, used_resources, **kwargs):
    peor_coste = 1000
    pos = -1
    while used_resources > resources:
        for i in range(len(disponibilidad)):
            if disponibilidad[i] != -1:
                if (instant >= disponibilidad[i]) & (instant < (disponibilidad[i] + task_duration[i])):
                    actual = calcularCoste(i, task_duration, task_resource, **kwargs)
                    if actual < peor_coste:
                        peor_coste = actual
                        pos = i
        disponibilidad[pos] = -1
        used_resources -= task_resource[pos]
    return disponibilidad

def tareasDependiente(i, task_dependencies):
    contador = 0
    for dependency in task_dependencies:
        (dependent_task, current_task) = dependency
        if dependent_task == i + 1:
            contador += 1
    return contador

def calcularCoste(i, task_duration, task_resource,**kwargs):
     tasks = kwargs["tasks"]
     return (task_resource[i] / task_duration[i]) + tareasDependiente(i, task_dependencies = kwargs["task_dependencies"])




def modificarDisponibilidad(cumpleDependencias, disponibilidad, mejor):
    for i in range(len(cumpleDependencias)):
        if disponibilidad[i] == -1 and cumpleDependencias[i]:
            # Schedule the task at the earliest possible time
            disponibilidad[i] = mejor
    return disponibilidad

def a_star(disponibilidad, cumpleDependencias, **kwargs):
    open_set = [(optimal_heuristic(disponibilidad, kwargs['task_duration'], kwargs['task_resource'], kwargs['task_dependencies'], kwargs['resources']), disponibilidad, cumpleDependencias)]

    while open_set:
        # Find the state with the lowest f-cost
        current_state = min(open_set, key=lambda x: x[0])
        open_set.remove(current_state)
        current_f_cost, current_disponibilidad, current_cumpleDependencias = current_state

        # Check if the goal has been reached
        if all(current_cumpleDependencias):
            return current_disponibilidad  # Goal reached

        for i in range(calculate_makespan(task_duration=kwargs['task_duration'])):
            new_cumpleDependencias = actualizarCumplido(current_cumpleDependencias, current_disponibilidad, i,
                                                         task_duration=kwargs["task_duration"],
                                                         task_dependencies=kwargs["task_dependencies"])
            new_disponibilidad = modificarDisponibilidad(new_cumpleDependencias, current_disponibilidad, i)
            new_disponibilidad = checkResources(new_disponibilidad, new_cumpleDependencias, task_duration=kwargs['task_duration'], task_resource=kwargs['task_resource'], resources=kwargs['resources'], task_dependencies=kwargs["task_dependencies"], tasks=kwargs['tasks'])

            # Skip this state if not all dependencies are met
            if not all(new_cumpleDependencias):
                continue

            # Calculate the f-cost for the new state
            h_cost = optimal_heuristic(new_disponibilidad, kwargs['task_duration'], kwargs['task_resource'], kwargs['task_dependencies'], kwargs['resources'])
            f_cost = calculate_makespan(task_duration=kwargs['task_duration'], disponibilidad=new_disponibilidad) + h_cost

            # Add the new state to the open set
            open_set.append((f_cost, new_disponibilidad, new_cumpleDependencias))

    return None  # If the open set is empty and the goal was not reached, return None



# Heuristic function
def optimal_heuristic(disponibilidad, task_duration, task_resource, task_dependencies, resources):
    # Heuristic part 1: Earliest possible start time for unscheduled tasks
    h1 = 0
    for task, start_time in enumerate(disponibilidad):
        if start_time == -1:
            # Find the earliest time the task can start based on dependencies
            earliest_start = 0
            for dependent_task, current_task in task_dependencies:
                if current_task == task + 1:
                    dependent_end_time = disponibilidad[dependent_task - 1] + task_duration[dependent_task - 1]
                    earliest_start = max(earliest_start, dependent_end_time)
            h1 += earliest_start

    # Heuristic part 2: Resource constraint penalty
    h2 = 0
    for task, start_time in enumerate(disponibilidad):
        if start_time == -1:
            # Estimate the additional time needed due to resource constraints
            h2 += task_resource[task] / resources * task_duration[task]

    # Combine the heuristics
    heuristic_value = h1 + h2
    return heuristic_value





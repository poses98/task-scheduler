from src.checks.checkings import checkResources
from src.checks.checkings import checkDependencies



def exercise1(tasks=0, resources=0, task_duration=[], task_resource=[], task_dependencies=[]):
    print("Initial State:")
    initial_chromosome = [0] * tasks
    initial_cost = 0
    print(f"Chromosome: {initial_chromosome}, Cost: {initial_cost}")

    # Initialize priority queue for state exploration
    priority_queue = []

    def push(cost, chromosome):
        nonlocal priority_queue
        priority_queue.append((cost, chromosome))
        priority_queue.sort(reverse=True)  # Sort in reverse order to get the smallest cost first

    def pop():
        nonlocal priority_queue
        return priority_queue.pop()

    push(initial_cost, initial_chromosome)

    while priority_queue:
        current_cost, current_chromosome = pop()

        # Check if the current partial schedule is a valid solution
        if len(set(current_chromosome)) == tasks:
            print("\nFinal State (Solution Found):")
            print(f"Chromosome: {current_chromosome}, Cost: {current_cost}")
            return current_chromosome

        # Explore child states (partial schedules)
        for task_index in range(tasks):
            if current_chromosome[task_index] == 0:
                # Task not scheduled, try scheduling it at the earliest possible time
                new_chromosome = current_chromosome.copy()
                new_chromosome[task_index] = max(current_chromosome) + 1

                # Check resource and dependency constraints
                if (checkResources(new_chromosome, task_duration, task_resource, resources) and
                        checkDependencies(new_chromosome, task_dependencies, task_duration)):
                    new_cost = max(new_chromosome) + task_duration[task_index]
                    push(new_cost, new_chromosome)

    # If no solution found
    print("\nNo Solution Found")
    return []

    # A* Search Algorithm for task scheduling with resources

def exercise2(tasks, resources, task_duration, task_resource, task_dependencies):
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





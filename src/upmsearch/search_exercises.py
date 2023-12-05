from src.checks.checkings import checkResources
from src.checks.checkings import checkDependencies
import heapq
import checks


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

    # Define the heuristic function
    def heuristic(state):
        return heuristic_function(state['position'], chromosome=state['chromosome'],
                                  tasks_dependencies=task_dependencies, task_duration=task_duration,
                                  task_resource=task_resource)

    # Define the successors function using the selectDecision function
    def successors(state):
        successors = []
        selected_task = selectDecision(state['chromosome'])
        if selected_task != -1:
            new_state = state.copy()
            new_state['chromosome'][selected_task] = state['time']  # Assign the current time to the selected task
            new_state['time'] += task_duration[selected_task]  # Increment time by the task's duration
            successors.append(new_state)
        return successors

    # Define the goal check function using the provided checks module
    def is_goal_state(state):
        return checks.checkDependencies(chromosome=state['chromosome'], task_dependencies=task_dependencies,
                                        task_duration=task_duration) and checks.checkResources(

            chromosome=state['chromosome'], task_duration=task_duration, task_resource=task_resource)

    # Initialize the priority queue
    frontier = []
    # Initial state: (cost, {'time': 0, 'chromosome': [-1] * tasks})
    initial_state = {'time': 0, 'chromosome': [-1] * tasks}
    heapq.heappush(frontier, (heuristic(initial_state), initial_state))

    while frontier:
        current_cost, current_state = heapq.heappop(frontier)

        # Check if current state is the goal state
        if is_goal_state(current_state):
            return current_state['chromosome']

        # Expand the current state to its successors
        for next_state in successors(current_state):
            next_cost = current_cost + 1  # Assume a uniform cost of 1 for each step
            heapq.heappush(frontier, (next_cost + heuristic(next_state), next_state))

    # If no solution is found, return an empty list
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

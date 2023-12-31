# 1. IMPORTS
import time
"FOR TESTING:"
from src.checks.testing_checkings import testDependenciesUnFullfilled
from src.checks.testing_checkings import testDependenciesFullfilled
from src.checks.testing_checkings import testResourcesFulfilled
from src.checks.testing_checkings import testResourcesUnfulfilled

"FOR MENU:"
from src.menu import menuAlgorithm
from src.menu import menuProblems

"FOR ALGORITHMS:"
# rcpsp06
from src.upmproblems.rcpsp06 import get_tasks as get_tasks_rcpsp06
from src.upmproblems.rcpsp06 import get_resources as get_resources_rcpsps06
from src.upmproblems.rcpsp06 import get_task_duration as get_task_duration_rcpsp06
from src.upmproblems.rcpsp06 import get_task_resource as get_task_resource_rcpsp06
from src.upmproblems.rcpsp06 import get_task_dependencies as get_task_dependencies_rcpsp06
# rcpsp07
from src.upmproblems.rcpsp07 import get_tasks as get_tasks_rcpsp07
from src.upmproblems.rcpsp07 import get_resources as get_resources_rcpsp07
from src.upmproblems.rcpsp07 import get_task_duration as get_task_duration_rcpsp07
from src.upmproblems.rcpsp07 import get_task_resource as get_task_resource_rcpsp07
from src.upmproblems.rcpsp07 import get_task_dependencies as get_task_dependencies_rcpsp07
# rcpsp10
from src.upmproblems.rcpsp10 import get_tasks as get_tasks_rcpsp10
from src.upmproblems.rcpsp10 import get_resources as get_resources_rcpsp10
from src.upmproblems.rcpsp10 import get_task_duration as get_task_duration_rcpsp10
from src.upmproblems.rcpsp10 import get_task_resource as get_task_resource_rcpsp10
from src.upmproblems.rcpsp10 import get_task_dependencies as get_task_dependencies_rcpsp10
# rcpsp30
from src.upmproblems.rcpsp30 import get_tasks as get_tasks_rcpsp30
from src.upmproblems.rcpsp30 import get_resources as get_resources_rcpsp30
from src.upmproblems.rcpsp30 import get_task_duration as get_task_duration_rcpsp30
from src.upmproblems.rcpsp30 import get_task_resource as get_task_resource_rcpsp30
from src.upmproblems.rcpsp30 import get_task_dependencies as get_task_dependencies_rcpsp30

"FOR EVOLUTIONARY ALGORITHMS:"
from src.upmevo.evo_exercises import exercise3 as simpleGenetic
from src.upmevo.evo_exercises import exercise4 as advancedGenetic
import sys
from statistics import mean, stdev

"FOR SEARCH:"
from src.upmsearch.search_exercises import result_makespan
from src.upmsearch.search_exercises import exercise1 as branchAndBound
from src.upmsearch.search_exercises import exercise2 as aStar



# 2. VERIFICATION OF CHECKDEPENDENCIES AND CHECKRESOURCES
def checkingAlg():
    print("Testing that the method checkDependencies works correctly, at least for rcpsp06 (for example)")
    testDependenciesFullfilled()
    testDependenciesUnFullfilled()

    print()

    print("Testing that the method checkResources works correctly, at least for rcpsp06 (for example)")
    testResourcesFulfilled()
    testResourcesUnfulfilled()
    print("-----------------------------------------------------------------------------------------------")


# 3. TASKSCHEDULER (CHOOSING
def taskScheduler():
    choiceAlg = -1
    while choiceAlg != 0:
        choiceAlg = menuAlgorithm()
        if choiceAlg != 0:
            choiceProb = menuProblems()
            # First, the date needed is assigned based on the previous choice madr
            match choiceProb:
                case 1:
                    tasks = get_tasks_rcpsp06()
                    resources = get_resources_rcpsps06()
                    task_duration = get_task_duration_rcpsp06()
                    task_resource = get_task_resource_rcpsp06()
                    task_dependencies = get_task_dependencies_rcpsp06()
                case 2:
                    tasks = get_tasks_rcpsp07()
                    resources = get_resources_rcpsp07()
                    task_duration = get_task_duration_rcpsp07()
                    task_resource = get_task_resource_rcpsp07()
                    task_dependencies = get_task_dependencies_rcpsp07()
                case 3:
                    tasks = get_tasks_rcpsp10()
                    resources = get_resources_rcpsp10()
                    task_duration = get_task_duration_rcpsp10()
                    task_resource = get_task_resource_rcpsp10()
                    task_dependencies = get_task_dependencies_rcpsp10()
                case 4:
                    tasks = get_tasks_rcpsp30()
                    resources = get_resources_rcpsp30()
                    task_duration = get_task_duration_rcpsp30()
                    task_resource = get_task_resource_rcpsp30()
                    task_dependencies = get_task_dependencies_rcpsp30()
                case _:
                    print("A problem has occured with the data")

            # Then, the algorithm starts (depending on the previous choice made), with its data already prepared
            match choiceAlg:
                case 1:
                    start_time = time.time()
                    result = branchAndBound(tasks, resources, task_duration, task_resource, task_dependencies)
                    makespan = result_makespan(result, task_duration)
                    print(f"Makespan: {makespan}")
                    end_time = time.time()
                    print(result)
                    elapsed_time = end_time - start_time
                    print(f"Execution time: {elapsed_time:.30f} seconds")
                case 2:
                    start_time = time.time()
                    result = aStar(tasks, resources, task_duration, task_resource, task_dependencies)
                    makespan = result_makespan(result, task_duration)
                    print(f"Makespan: {makespan}")
                    end_time = time.time()
                    print(result)
                    elapsed_time = end_time - start_time
                    print(f"Execution time: {elapsed_time:.30f} seconds")
                case 3:
                    seed = 1
                    simpleGenetic(seed, tasks, resources, task_duration, task_resource, task_dependencies)
                case 4:
                    seed = 0
                    advancedGenetic(seed, tasks, resources, task_duration, task_resource, task_dependencies, plot=True)
                case 5:
                    n_ex = 31
                    fitness_ex = []
                    sys.stdout.write(f"Progress: {0}/{n_ex-1}")
                    for i in range(n_ex):
                        seed = 1234567 + i * i ^ 2
                        fittest_fitness, fittest_individual = advancedGenetic(seed, tasks, resources, task_duration, task_resource, task_dependencies, plot=False)
                        fitness_ex.append(fittest_fitness/10)
                        sys.stdout.flush()
                        sys.stdout.write("\r")
                        sys.stdout.write(f"Progress: {i}/{n_ex-1}")
                    print("\nMBF: " + str(mean(fitness_ex)*10) + " " + u"\u00B1" + " " + str(stdev(fitness_ex)*10))

    print("Execution completed. Thank you")

print("\Checking checks")
checkingAlg()
print('\nTask Scheduler')
taskScheduler()




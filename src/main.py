"FOR TESTING:"
from src.checks.testing_checkings import testDependenciesUnFullfilled
from src.checks.testing_checkings import testDependenciesFullfilled
from src.checks.testing_checkings import testResourcesFulfilled
from src.checks.testing_checkings import testResourcesUnfulfilled
"FOR MENU:"
from src.menu import menuAlgorithm
from src.menu import menuProblems
"FOR ALGORITHMS:"
#rcpsp06
from src.upmproblems.rcpsp06 import get_tasks as get_tasks_rcpsp06
from src.upmproblems.rcpsp06 import get_resources as get_resources_rcpsps06
from src.upmproblems.rcpsp06 import get_task_duration as get_task_duration_rcpsp06
from src.upmproblems.rcpsp06 import get_task_resource as get_task_resource_rcpsp06
from src.upmproblems.rcpsp06 import get_task_dependencies as get_task_dependencies_rcpsp06
#rcpsp07
from src.upmproblems.rcpsp07 import get_tasks as get_tasks_rcpsp07
from src.upmproblems.rcpsp07 import get_resources as get_resources_rcpsp07
from src.upmproblems.rcpsp07 import get_task_duration as get_task_duration_rcpsp07
from src.upmproblems.rcpsp07 import get_task_resource as get_task_resource_rcpsp07
from src.upmproblems.rcpsp07 import get_task_dependencies as get_task_dependencies_rcpsp07
#rcpsp10
from src.upmproblems.rcpsp10 import get_tasks as get_tasks_rcpsp10
from src.upmproblems.rcpsp10 import get_resources as get_resources_rcpsp10
from src.upmproblems.rcpsp10 import get_task_duration as get_task_duration_rcpsp10
from src.upmproblems.rcpsp10 import get_task_resource as get_task_resource_rcpsp10
from src.upmproblems.rcpsp10 import get_task_dependencies as get_task_dependencies_rcpsp10
#rcpsp30
from src.upmproblems.rcpsp30 import get_tasks as get_tasks_rcpsp30
from src.upmproblems.rcpsp30 import get_resources as get_resources_rcpsp30
from src.upmproblems.rcpsp30 import get_task_duration as get_task_duration_rcpsp30
from src.upmproblems.rcpsp30 import get_task_resource as get_task_resource_rcpsp30
from src.upmproblems.rcpsp30 import get_task_dependencies as get_task_dependencies_rcpsp30
"FOR EVOLUTIONARY ALGORITHMS:"
from src.upmevo.evo_exercises import exercise3 as simpleGenetic
from src.upmevo.evo_exercises import exercise4 as advancedGenetic
"FOR SEARCH:"
from src.upmsearch.search_exercises import exercise1 as branchAndBound
from src.upmsearch.search_exercises import exercise2 as aStar

def checkingAlg():
    print("Testing that the method checkDependencies works correctly, at least for rcpsp06 (for example)")
    testDependenciesFullfilled()
    testDependenciesUnFullfilled()

    print()

    print("Testing that the method checkResources works correctly, at least for rcpsp06 (for example)")
    testResourcesFulfilled()
    testResourcesUnfulfilled()
    print("-----------------------------------------------------------------------------------------------")

    print('\nTask Scheduler')

def taskScheduler():
    choiceAlg = -1
    choiceProb = -1
    while choiceAlg != 0:
        choiceAlg = menuAlgorithm()
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

        #Then, the algorithm starts (depending on the previous choice made), with its data already prepared
        match choiceAlg:
            case 1:
                branchAndBound(tasks, resources, task_duration, task_resource, task_dependencies)
            case 2:
                aStar(tasks, resources, task_duration, task_resource, task_dependencies)
            case 3:
                seed = 0 #HERE, CORREGIR !!
                simpleGenetic(seed, tasks,resources, task_duration, task_resource, task_dependencies)
            case 4:
                seed = 0 #HERE, CORREGIR !!
                advancedGenetic(seed, tasks, resources, task_duration, task_resource, task_dependencies)





print("TESTING CHECKS")
checkingAlg()
taskScheduler()
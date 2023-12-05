from src.checks.checkings import checkDependencies

from src.upmproblems.rcpsp06 import get_task_dependencies as rcpsp06_get_task_dependencies
from src.upmproblems.rcpsp06 import get_task_duration as rcpsp06_get_task_duration

from src.checks.checkings import checkResources
from src.upmproblems.rcpsp06 import get_task_duration, get_task_resource

"""
DEPENDENCIES
---------------------------------------------------------------------------------------------------
"""


# Test case 1: Partial path with fulfilled dependencies for rcpsp06
def testDependenciesFullfilled():
    partial_path_1 = [1, 2, 3, 5]
    result = checkDependencies(partial_path_1, rcpsp06_get_task_dependencies(), rcpsp06_get_task_duration())
    resultDependencies(result)
    return 0


# Test case 2: Partial path with unfulfilled dependencies for rcpsp06
def testDependenciesUnFullfilled():
    partial_path_2 = [1, 2, 4, 5]
    result = checkDependencies(partial_path_2, rcpsp06_get_task_dependencies(), rcpsp06_get_task_duration())
    resultDependencies(result)
    return 0


def resultDependencies(result):
    if result is True:
        print("Test case 1: Dependencies fulfilled for rcpsp06")
    else:
        print("Test case 2: Dependencies not fulfilled for rcpsp06")


"""
RESOURCES
---------------------------------------------------------------------------------------------------
"""


# Test case 1: Partial path with fulfilled resources for rcpsp06
def testResourcesFulfilled():
    partial_path_1 = [1, 2, 3]
    task_duration = get_task_duration()
    task_resource = get_task_resource()
    resources = 4

    result = checkResources(partial_path_1, task_duration, task_resource, resources)
    resultResources(result)
    return 0


# Test case 1: Partial path with fulfilled resources for rcpsp06
def testResourcesUnfulfilled():
    partial_path_2 = [1, 2, 3, 4]  # Assume this path exceeds the available resources
    task_duration = get_task_duration()
    task_resource = get_task_resource()
    resources = 2

    result = checkResources(partial_path_2, task_duration, task_resource, resources)
    resultResources(result)
    return 0


def resultResources(result):
    if result is True:
        print("Test case 1: Resources fulfilled for rcpsp06")
    else:
        print("Test case 2: Resources not fulfilled for rcpsp06")

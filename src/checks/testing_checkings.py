from src.checks.checkings import checkDependencies
from src.upmproblems.rcpsp06 import get_task_dependencies as rcpsp06_get_task_dependencies
"""
DEPENDENCIES
---------------------------------------------------------------------------------------------------
"""
# Test case 1: Partial path with fulfilled dependencies for rcpsp06
def testDependenciesFullfilled():
   partial_path_1 = [1, 2, 3, 5]
   if checkDependencies(partial_path_1, rcpsp06_get_task_dependencies()):
       print("Test case 1: Dependencies fulfilled for rcpsp06")
   else:
       print("Test case 1: Dependencies not fulfilled for rcpsp06")
   return 0


# Test case 2: Partial path with unfulfilled dependencies for rcpsp06
def testDependenciesUnFullfilled():
   partial_path_2 = [1, 2, 4, 5]
   if checkDependencies(partial_path_2, rcpsp06_get_task_dependencies()):
       print("Test case 2: Dependencies fulfilled for rcpsp06")
   else:
       print("Test case 2: Dependencies not fulfilled for rcpsp06")
   return 0


"""
RESOURCES
---------------------------------------------------------------------------------------------------
"""

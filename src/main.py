
print('Task Scheduler')


"DELETE"
from src.checks.testing_checkings import testDependenciesUnFullfilled
from src.checks.testing_checkings import testDependenciesFullfilled
print("Testing that the method checkDependencies works correctly, at least for rcpsp06 (for example)")
testDependenciesFullfilled()
testDependenciesUnFullfilled()
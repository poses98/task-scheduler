def checkDependencies(path, task_dependencies):
   """
   Returns true if all dependencies are fulfilled, false otherwise.
   Parameters:
   - path: List of task indices representing the partial path.
   - task_dependencies: List of tuples representing task dependencies.
   """
   for task, dependent_task in task_dependencies:
       if dependent_task in path and task not in path:
           return False
   return True
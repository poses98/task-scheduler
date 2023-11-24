def checkDependencies(path, task_dependencies):
   """
   Returns true if all dependencies are fulfilled, false otherwise.
   Parameters:
   - path: list of task indices representing the partial path
   - task_dependencies: list of tuples representing task dependencies
   """
   for task, dependent_task in task_dependencies:
       if dependent_task in path and task not in path:
           return False
   return True

def checkResources(path, task_duration, task_resource, resources):
    """
    Returns true if the number of used resources for a partial path does
    not exceed R. False otherwise
    :param path: list of task indices representing the partial path
    :param task_duration: list of task durations
    :param task_resource: list of task resource requirements
    :param resources: total number of available resources (R)
    :return:
    """
    used_resources = [0] * resources

    for task in path:
        task_index = task - 1
        used_resources[task_resource[task_index] - 1] += task_duration[task_index]
         #Check if exceeds the total resources
        if used_resources[task_resource[task_index] - 1] > resources:
            return False
    return True
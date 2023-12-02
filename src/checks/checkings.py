def checkDependencies(chromosome, task_dependencies, task_duration):
   """
   Returns true if all dependencies are fulfilled, false otherwise.
   Parameters:
   - chromosome: list of the starting time of each task, initialized to -1
   - task_dependencies: list of tuples representing task dependencies
   - task_duration: list of the duration of each task.
   """
   for x in range(len(chromosome)):
       task_start_time = chromosome[x]
       task_duration = task_duration[x]
       for dependency in task_dependencies:
           dependent_task, current_task = dependency
           dependent_task_start_time = chromosome[dependent_task]
           dependent_task_finish_time = dependent_task_start_time + task_duration[dependent_task]
           if task_start_time < dependent_task_start_time + dependent_task_finish_time:
               return False
           else:
               if task_start_time != -1 & dependent_task_start_time == -1:
                   return False
   return True

def checkResources(chromosome, task_duration, task_resource, resources):
    """
    Returns true if the number of used resources for a partial path does
    not exceed R. False otherwise
    :param chromosome: list of the starting time of each task, initialized to -1
    :param task_duration: list of task durations
    :param task_resource: list of task resource requirements
    :param resources: total number of available resources (R)
    :return:
    """
    used_resources = 0
    for max_final in range(len(chromosome)):
        latest_init = chromosome[0]
        if chromosome[max_final] > latest_init:
            latest_init = chromosome[max_final]

    for each_instant in range(latest_init):
        for span_task in range(len(chromosome)):
            if each_instant <= span_task+task_duration[span_task]:
                used_resources = used_resources + task_resource[span_task]
    if used_resources > resources:
        return False
    else:
        return True
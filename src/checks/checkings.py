def checkDependencies(chromosome, task_duration, task_dependencies):
   """
   Returns true if all dependencies are fulfilled, false otherwise.
   Parameters:
   - chromosome: list of the starting time of each task, initialized to -1
   - task_dependencies: list of tuples representing task dependencies
   - task_duration: list of the duration of each task.
   """
   fulfilled = True
   for x in range(len(chromosome)):
       task_start_time = chromosome[x]
       for dependency in task_dependencies:
           (dependent_task, current_task) = dependency
           if (current_task == x+1):
            dependent_task_start_time = chromosome[dependent_task-1]
            if task_start_time < (dependent_task_start_time + task_duration[dependent_task-1]):
                fulfilled = False
                print('aquí falla')
                return fulfilled
            else:
                if task_start_time != -1 & dependent_task_start_time == -1:
                    fulfilled = False
                    return fulfilled
   return fulfilled

def calculate_makespan(chromosome, task_duration, *args, **kwargs):
    latest_end = -1
    for actual_task in range(len(chromosome)):
        end_time = chromosome[actual_task]+task_duration[actual_task]
        if end_time > latest_end:
            latest_end = end_time
    return latest_end

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
    fulfilled = True
    for each_instant in range(calculate_makespan(chromosome, task_duration)):
        used_resources = 0
        for span_task in range(len(chromosome)):
            if chromosome[span_task] != -1:
                if (each_instant > chromosome[span_task]) & (each_instant <= (chromosome[span_task] + task_duration[span_task])):
                    used_resources += task_resource[span_task]
        if used_resources > resources:
            fulfilled = False
            return fulfilled
    return fulfilled

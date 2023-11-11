tasks = 10
resources = 6
task_duration = [3, 2, 5, 4, 2, 3, 4, 2, 4, 6]
task_resource = [5, 1, 1, 1, 3, 3, 2, 4, 5, 2]
task_dependencies = [(1, 4), (1, 5), (2, 9), (2, 10), (3, 8), (4, 6),
                     (4, 7), (5, 9), (5, 10), (6, 8), (6, 9), (7, 8)]

def get_tasks():
    return tasks


def get_resources():
    return resources


def get_task_duration():
    return task_duration


def get_task_resource():
    return task_resource


def get_task_dependencies():
    return task_dependencies

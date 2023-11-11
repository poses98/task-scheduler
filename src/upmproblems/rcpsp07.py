tasks = 7
resources = 5
task_duration = [2, 1, 1, 1, 3, 2, 1]
task_resource = [4, 1, 2, 2, 2, 1, 2]
task_dependencies = [(1, 3), (1, 5), (3, 6), (4, 6), (5, 7), (6, 7)]


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

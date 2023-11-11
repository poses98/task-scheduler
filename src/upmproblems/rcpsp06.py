tasks = 6
resources = 4
task_duration = [3, 4, 2, 2, 1, 4]
task_resource = [2, 3, 4, 4, 3, 2]
task_dependencies = [(1, 3), (2, 3), (2, 4), (3, 5), (4, 6)]


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

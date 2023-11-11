tasks = 30
resources = 28
task_duration = [5, 3, 1, 1, 5, 2, 2, 8, 9, 1,
                 4, 2, 4, 1, 1, 3, 7, 6, 1, 8,
                 3, 3, 10, 8, 7, 4, 5, 2, 7, 5]
task_resource = [8, 5, 1, 7, 1, 4, 1, 4, 5, 1,
                 7, 11, 1, 6, 1, 1, 10, 9, 8, 1,
                 1, 1, 1, 8, 8, 1, 1, 9, 1, 3]
task_dependencies = [(1, 4), (2, 5), (2, 16), (2, 25), (3, 9), (3, 13),
                     (3, 18), (4, 11), (4, 15), (4, 17), (5, 6), (5, 7),
                     (5, 14), (6, 8), (6, 10), (6, 19), (7, 21), (8, 22),
                     (9, 20), (10, 12), (11, 16), (11, 30), (12, 28), (13, 26),
                     (14, 18), (14, 28), (15, 25), (15, 26), (16, 26), (16,27),
                     (17, 18), (17, 24), (18, 27), (19, 24), (20, 29), (21, 23),
                     (22, 30), (23, 27), (24, 30), (25, 28), (26, 29), (27, 29)]


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

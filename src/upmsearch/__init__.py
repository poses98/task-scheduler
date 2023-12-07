import src.upmproblems.rcpsp06 as rcpsp06

"lista de caminos que salen"
"definir función coste"
"definir f makespan"
"impr"

def calculate_makespan(task_duration, *args, **kwargs):
    end_time = -1
    for actual_task in range(len(task_duration)):
        end_time += task_duration[actual_task]
    return end_time


"array disponibilidad"


def exercise1(tasks=0, resources=0, task_duration=[], task_resource=[], task_dependencies=[]):
    "inicializando"
    disponibilidad = [-1] * tasks
    cumpleDependencia = [True] * tasks
    branch_and_bound(disponibilidad, cumpleDependencia, tasks=tasks, resources=resources, task_duration=task_duration,
                     task_resource=task_resource, task_dependencies=task_dependencies)

    "comprobar cuáles no disponen de ninguno"

    return disponibilidad


def checkDependenciasBnB(cumpleDependencias, disponibilidad, task_duration, task_dependencies):
    for x in range(len(cumpleDependencias)):
        if disponibilidad[x] == -1:
            task_start_time = disponibilidad[x]
            for dependency in task_dependencies:
                (dependent_task, current_task) = dependency
                if current_task == x + 1:
                    dependent_task_start_time = disponibilidad[dependent_task - 1]
                    if (task_start_time == -1) & (dependent_task_start_time != -1):
                        cumpleDependencias[current_task - 1] = True
                    else:
                        if task_start_time < (dependent_task_start_time + task_duration[dependent_task]):
                            cumpleDependencias[current_task - 1] = False
                        else:
                            if task_start_time != -1 & dependent_task_start_time == -1:
                                cumpleDependencias[current_task] = False
    return cumpleDependencias


def checkResources(disponibilidad, cumpleDependencias, task_duration, task_resource, resources):
    for instant in range(calculate_makespan(task_duration)):
        used_resources = 0
        for span_task in range(len(cumpleDependencias)):
            if disponibilidad[span_task] != -1:
                if (instant >= disponibilidad[span_task]) & (instant < (disponibilidad[span_task] + task_duration[span_task])):
                    used_resources += task_resource[span_task]
                if used_resources > resources:
                    #AÑADIR COSTE PARA SELECCIONAR CUAL DE LAS TAREAS QUE ESTÁN USANDO EL RECURSO QUITAMOS
                    disponibilidad[span_task] = -1
                    used_resources -= task_resource[span_task]
    return cumpleDependencias, disponibilidad

def modificarDisponibilidad(cumpleDependencias, disponibilidad, mejor):
    for i in range(len(cumpleDependencias)):
        if disponibilidad[i] == -1:
            if cumpleDependencias[i]:
                disponibilidad[i] = mejor
    return disponibilidad


def branch_and_bound(disponibilidad, cumpleDependencias, **kwargs):
    for i in range(calculate_makespan(task_duration=kwargs['task_duration'])):
        cumpleDependencias = checkDependenciasBnB(cumpleDependencias, disponibilidad,
                                                  task_duration=kwargs["task_duration"],
                                                  task_dependencies=kwargs["task_dependencies"])
        disponibilidad = modificarDisponibilidad(cumpleDependencias, disponibilidad, i)
        cumpleDependencias, disponibilidad = checkResources(disponibilidad, cumpleDependencias, task_duration=kwargs['task_duration'], task_resource=kwargs['task_resource'], resources=kwargs['resources'])

    "checkDep, el que no dp en true"
    "checkRec, false los que cumplan"
    "si en el array de T/F, seleccionar el de mejor coste"
    "mejor = mejorCoste()"
    "modifDisponibilidad"


"mirar solo los que no son -1"

"poner en qué momento se inicializa "
print(exercise1(rcpsp06.get_tasks(), rcpsp06.get_resources(), rcpsp06.task_duration, rcpsp06.get_task_resource(),
                rcpsp06.get_task_dependencies()))

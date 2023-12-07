import src.upmproblems.rcpsp06 as rcpsp06

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
    return disponibilidad

def actualizarCumplido (cumpleDependencias,disponibilidad, i, task_duration, task_dependencies):
    for task in range(len(cumpleDependencias)):
        cumpleDependencias[task] = checkDependenciasBnB(i,disponibilidad, task_duration, task_dependencies, task)
    return cumpleDependencias


def checkDependenciasBnB(i, disponibilidad, task_duration, task_dependencies, task):

    #[-1, 0, -1, -1, -1, -1] i = 2
    cumple = True
    for dependency in task_dependencies:
        (dependent_task, current_task) = dependency #task_dependencies = [(1, 3), (2, 3), (2, 4), (3, 5), (4, 6)]
        if current_task == task + 1:
            dependent_task_start_time = disponibilidad[dependent_task - 1] #0
            print('Tiempo de inicio de la tarea '+ str(dependent_task)+ ' > ' + str(dependent_task_start_time))
            if dependent_task_start_time == -1:
                cumple = False
            else:
                print('Tiempo de la tarea de la que depende' + str(dependent_task_start_time + task_duration[dependent_task-1]) + ' vs. Instante i ' + str(i))
                if (dependent_task_start_time + task_duration[dependent_task-1]) > i:
                    #task_duration = [3, 4, 2, 2, 1, 4]
                    cumple = False
                    #CASOS QUE INCUMPLEN DEPENDENCIAS: EMPEZAR UNA TAREA EN EL TIEMPO DE LAS QUE DEPENDE, EMPEZAR UNA TAREA SIN HABER SELECCIONADO DE LAS QUE DEPENDE
    return cumple


def checkResources(disponibilidad, cumpleDependencias, task_duration, task_resource, resources, **kwargs):
    for instant in range(calculate_makespan(task_duration)):
        used_resources = 0
        for span_task in range(len(cumpleDependencias)):
            if disponibilidad[span_task] != -1:
                if (instant >= disponibilidad[span_task]) & (instant < (disponibilidad[span_task] + task_duration[span_task])):
                    used_resources += task_resource[span_task]
                if used_resources > resources:
                    disponibilidad = mejorCoste(instant, disponibilidad, cumpleDependencias, task_duration, task_resource, resources, used_resources, **kwargs)
    return disponibilidad

def mejorCoste(instant, disponibilidad, cumpleDependencias, task_duration, task_resource, resources, used_resources, **kwargs):
    "pos del mejor coste" # AÑADIR COSTE PARA SELECCIONAR CUAL DE LAS TAREAS QUE ESTÁN USANDO EL RECURSO QUITAMOS
    peor_coste = 1000
    pos = -1
    while used_resources > resources:
        for i in range(len(disponibilidad)):
            if disponibilidad[i] != -1:
                if (instant >= disponibilidad[i]) & (instant < (disponibilidad[i] + task_duration[i])):
                    actual = calcularCoste(i, task_duration, task_resource, **kwargs)
                    if actual < peor_coste:
                        peor_coste = actual
                        pos = i
        disponibilidad[pos] = -1
        used_resources -= task_resource[pos]
    return disponibilidad

def tareasDependiente(i, task_dependencies):
    contador = 0
    for x in range(len(task_dependencies)):
            for dependency in task_dependencies:
                (dependent_task, current_task) = dependency
                if dependent_task == i + 1:
                    contador += 1
    return contador

def calcularCoste(i, task_duration, task_resource, **kwargs):
    "recursos/tiempo + calcular cuántas tareas dependen de ella/ta"
    tasks = kwargs["tasks"]
    return (task_resource[i]/ task_duration[i]) + tareasDependiente(i, task_dependencies = kwargs["task_dependencies"])

def modificarDisponibilidad(cumpleDependencias, disponibilidad, mejor):
    for i in range(len(cumpleDependencias)):
        if disponibilidad[i] == -1:
            if cumpleDependencias[i]:
                disponibilidad[i] = mejor
    return disponibilidad


def branch_and_bound(disponibilidad, cumpleDependencias, **kwargs):
    for i in range(calculate_makespan(task_duration=kwargs['task_duration'])):
        cumpleDependencias = actualizarCumplido(cumpleDependencias, disponibilidad, i,
                                                  task_duration=kwargs["task_duration"],
                                                  task_dependencies=kwargs["task_dependencies"])
        print(i)
        print(cumpleDependencias)
        disponibilidad = modificarDisponibilidad(cumpleDependencias, disponibilidad, i)
        print(disponibilidad)
        disponibilidad = checkResources(disponibilidad, cumpleDependencias, task_duration=kwargs['task_duration'], task_resource=kwargs['task_resource'], resources=kwargs['resources'], task_dependencies=kwargs["task_dependencies"], tasks=kwargs['tasks'])
        print(disponibilidad)
    "checkDep, el que no dp en true"
    "checkRec, false los que cumplan"
    "si en el array de T/F, seleccionar el de mejor coste"
    "mejor = mejorCoste()"
    "modifDisponibilidad"


"mirar solo los que no son -1"

"poner en qué momento se inicializa "
print(exercise1(rcpsp06.get_tasks(), rcpsp06.get_resources(), rcpsp06.task_duration, rcpsp06.get_task_resource(),
                rcpsp06.get_task_dependencies()))

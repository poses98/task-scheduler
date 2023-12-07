from src.checks.checkings import checkResources
from src.checks.checkings import checkDependencies
import heapq
import checks
import itertools
import src.upmproblems.rcpsp06 as rcpsp06
"lista de caminos que salen"
"definir función coste"
"definir f makespan"
"impr"

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
            if task_start_time < (dependent_task_start_time + task_duration[dependent_task]):
                fulfilled = False

                return fulfilled
            else:
                if task_start_time != -1 & dependent_task_start_time == -1:
                    fulfilled = False
                    return fulfilled
   return fulfilled

def calculate_makespan(task_duration, *args, **kwargs):
    end_time = -1
    for actual_task in range(len(task_duration)):
        end_time += task_duration[actual_task]
    return end_time

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

"array disponibilidad"
def exercise1(tasks=0, resources=0, task_duration=[], task_resource=[], task_dependencies=[]):
  "inicializando"
  disponibilidad = [-1] * tasks
  cumpleDependencia = [True] * tasks
  brand_and_bound(disponibilidad, cumpleDependencia, tasks=tasks, resources=resources, task_duration=task_duration,
                  task_resource=task_resource, task_dependencies=task_dependencies)


  "comprobar cuáles no disponen de ninguno"

  return cumpleDependencia

def checkDependenciasBnB(cumpleDependencias, disponibilidad, task_duration, task_dependencies):
   fulfilled = True
   for x in range(calculate_makespan(task_duration)):
       task_start_time = disponibilidad[x]
       for dependency in task_dependencies:
           (dependent_task, current_task) = dependency
           if (current_task == x+1):
            dependent_task_start_time = disponibilidad[dependent_task-1]
            if task_start_time < (dependent_task_start_time + task_duration[dependent_task]):
                cumpleDependencias[current_task] = False

            else:
                if task_start_time != -1 & dependent_task_start_time == -1:
                    cumpleDependencias[current_task] = False
   return cumpleDependencias


def modificarDisponibilidad(cumpleDependencias, disponibilidad, mejor):
    return 0

def brand_and_bound(disponibilidad, cumpleDependencias, **kwargs):
    for i in range(calculate_makespan(task_duration=kwargs["task_duration"])):

            cumpleDependencias = checkDependenciasBnB(cumpleDependencias, disponibilidad, task_duration=kwargs["task_duration"], task_dependencies=kwargs["task_dependencies"])
            print(cumpleDependencias)
            "checkDep, el que no dp en true"
            "checkRec, false los que cumplan"
            "si en el array de T/F, seleccionar el de mejor coste"
            "mejor = mejorCoste()"
            "modifDisponibilidad"


"mirar solo los que no son -1"



"poner en qué momento se inicializa "
print(exercise1(rcpsp06.get_tasks(), rcpsp06.get_resources(), rcpsp06.task_duration, rcpsp06.get_task_resource(), rcpsp06.get_task_dependencies()))

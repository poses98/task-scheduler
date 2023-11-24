def menuAlgorithm():
    choice = -1
    while choice < 0 or choice > 4:
        print("\nMenu algorithms:")
        print("1. Branch and Bound")
        print("2. A*")
        print("3. Simple Genetic Algorithm")
        print("4. Advanced Genetic Algorithm")
        print("0. Exit")
        choice = int(input("Enter your choice: "))
        if choice < 0 or choice > 4:
            print("Please enter a valid option 0-4.")
    return choice

def menuProblems():
    choice = 10
    while choice < 1 or choice > 4:
        print("\nMenu problems:")
        print("1. rcpsp06")
        print("2. rcpsp07")
        print("3. rcpsp10")
        print("4. rcpsp30")
        choice = int(input("Enter your choice: "))
        if choice < 1 or choice > 3:
            print("Please enter a valid option 0-4.")
    return choice
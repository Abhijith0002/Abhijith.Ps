list = []                                                   # Global variable

def calculate_net_sal():
    global list
    empId = int (input ("Enter the employee Id: "))         # local variable
    empName = int (input ("Enter the employee Name: "))    
    empSalary = int (input ("Enter the employee Salary: "))
    hra = int (input ("Enter the employee HRA: "))
    da = int (input ("Enter the employee DA: "))
    netsalary = empSalary + hra + da
    list  = [empId, empName,empSalary, hra, da, netsalary]

def display():
    print ("Employee Id is: ",list[0])
    print ("Employee Name is: ",list[1])
    print ("Employee Salary is: ",list[2])
    print ("Employee HRA is: ",list[3])
    print ("Employee DA is: ",list[4])
    print ("Employee Net Salary is: ",list[5])

while True:
    print('''
    WELCOME TO EMPAPP2023
    1. Add Employee details
    2. Display details
    3. Exit
    ''')
    a = int(input("Enter your Choice"))
    if (a == 1):
        calculate_net_sal()
    elif (a == 2):
        display()
    elif (a == 3):
        break
    else:
        print ("Invalid input pls try again !!!!!!")
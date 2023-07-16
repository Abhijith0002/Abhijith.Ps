"""
Write a procedure that solves quadratic equations using the quadratic formula: It should take as arguments three numbers a, b, and c.
It should print error messages if a is zero, or if the roots are complex. Otherwise it should print the two roots.

"""
import math

a = int(input("Enter the 1st number: "))
b = int(input("Enter the 2nd number: "))
c = int(input("Enter the 3rd number: "))

if a == 0:
    print("Error: 'a' cannot be zero.")
else:
    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        print("Error: Roots are complex.")
    else:
        root1 = (-b + math.sqrt(discriminant)) / (2*a)
        root2 = (-b - math.sqrt(discriminant)) / (2*a)

        print("Root 1:", root1)
        print("Root 2:", root2)

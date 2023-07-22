"""
# 1.Print the list of numbers which are divisible by 5 and multiple of 8 between 2000 and 2500
list = []
for i in range (2000,2501):
    if(i % 5 == 0 and i % 8 == 0 ):
        list.append(i)
print (list)
            
# 2.Write a Python program to create the table (from 1 to 10) of a number getting input from the user

n = int (input("Enter a number"))
for i in range (1,11):
    j = (i*n)
    print (n,"*",i,"=", j)

"""
"""
# 3.sort the list in ascending order and print first element

n = input("Enter some Numbers Seperated by spaces: ")
no = n.split()
no.sort()
print ("The ascending order is: ",no)
print ("First element is: ",no[0])

"""
"""
# 4.Python program to find 2nd largest number in a list

n = input("Enter some Numbers Seperated by spaces: ")
no = n.split()
no.sort()
print ("2nd largest element is: ",no[-2])

"""
"""
# 5.python pgm to print even and odd seperately in a list of [1,2,......,10]

l = [1,2,3,4,5,6,7,8,9,10]
even = []
odd = []
for i in l:
    if (i % 2 == 0):
        even.append(i)
    else:
        odd.append(i)
print ("Even number: ",even)
print ("Odd number: ",odd)

"""
"""
# 6.Programing for reversing a list

l = []
a = input ("Enter some number with spaces")
l = a.split()
n = l[::-1]
print (n)

"""
"""
# 7. pgm to print all odd no from 1-50

l = []
for i in range(1,51):
    if (i % 2 != 0):
        l.append(i)
    else:
        continue
print (l)

"""

# 8. program to count even and odd number in a list

l = []
even = []
odd = []
a = input ("Enter some number with spaces")
l = a.split()
for i in l:
    if (int(i) % 2 == 0):
        even.append(i)
    else:
        odd.append(i)     
print (len(even))
print (len(odd))
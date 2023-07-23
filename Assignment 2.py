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

"""
"""

# 9. Write a python program to remove zeros from an IP address("216.08.094.196")

ip_address = input("Enter an IP address: ")
inp = ip_address.split('.')

for i in range((len(inp))):
    inp[i] = inp[i].lstrip('0')
ip_address = '.'.join(inp)

print (ip_address)

"""
"""

# 10 Write a Python program that matches a string that has an 'a' followed by anything. 

import re
pattern = re.compile('a*')

string = input('Enter a string: ')

if pattern.match(string):
    print('The string contains an "a" followed by anything')
else:
    print('The string does not contain an "a" followed by anything')

"""
"""

# 11 Replace all occurences of 6 with 'six' and 10 with 'ten' for the given string 'They ate 6 apples and 10 banana'

import re
input = input ("Enter a string with 6 and 10:")
a = re.sub("6","six",input) 
b = re.sub("10","ten",a)
print (b)

"""
"""

# 12.	Write a program to check whether a person is eligible for voting or not. (accept age from user)

def check_age():
    if (age < 18 or age > 60):
        print ("You are not eligible for voting..\U0001F62D ")
    else:
        print ("You are eligible for voting..\U0001f600")

while True:
    print ('''
    ELIGIBILITY CHECK
    ''')
    age = int (input("Enter your Age: "))
    check_age()

"""
"""
#. Write a program to calculate the electricity bill (accept number of unit from user) according to the following criteria : 
# First 100 units                                               no charge
# Next 100 units                                              Rs 5 per unit
# After 200 units                                             Rs 10 per unit
# (For example if input unit is 350 than total bill amount is Rs2000)

units = int(input("Enter the number of units: "))
total_bill = 0
remaining_units = units

if remaining_units >= 100:
    total_bill += 0
    remaining_units -= 100
else:
    total_bill += remaining_units * 0
    remaining_units = 0

if remaining_units >= 100:
    total_bill += 100 * 5
    remaining_units -= 100
else:
    total_bill += remaining_units * 5
    remaining_units = 0

if remaining_units > 0:
    total_bill += remaining_units * 10

print(f"Total bill amount is Rs {total_bill}")

"""





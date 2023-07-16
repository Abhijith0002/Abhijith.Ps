"""
Write a Python program to check the validity of a password input by the user. The password should satisfy the following conditions:

a.	It should have at least 1 letter [a to z]
b.	It should have at least 1 number [0 to 9]
c.	It should have at least 1 capital letter [A to Z]
d.	It should have at least one special character [$, #, @]
e.	Minimum length = 6 characters
f.	Maximum length = 12 characters

"""
password = input ("Enter a password: ")

valid = True                       # Initialize validity flag

required_conditions = [any(char.isalpha() for char in password),
                        any(char.isdigit() for char in password),
                          any(char.isupper() for char in password),
                            any(char in ['$', '#', '@'] for char in password)]

if not all(required_conditions):
    valid = False

if valid:
    print("Valid password")
else:
    print("Invalid password")
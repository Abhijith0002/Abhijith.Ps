"""
Write a Python program that takes a string of words separated by spaces, together with a ”target” word,
and shows the position of the target word in the string of words.
For example, if the string is: we dont need no education we dont need no thought control no we dont and the target word is ”dont”,
then your program should return the list 1, 6, 13 because ”dont” appears at the 1st, 6th, and 13th position in the string.
Your program should return False if the target word doesn’t appear in the string.

"""

string_of_words = input("Enter a string of words: ")
target_word = input("Enter the target word: ")

words = string_of_words.split()
positions = []

for i in range(len(words)):
    if words[i] == target_word:
        positions.append(i + 1)  # pushing value into list

if positions:
    print(positions)
else:
    print(False)

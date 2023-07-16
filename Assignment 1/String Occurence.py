"""
Write a program that reads in a string on the command line and returns a table of the frequency of occurrence of each word.
Ignore the case. A sample run of the program would look this
Enter some letters >>> The cat in the hat The - 2
Cat - 1
In - 1
Hat - 1

"""

sentence = input("Enter a sentence: ")
words = sentence.split()

frequency = sentence(int)            
for word in words:
    frequency[word] += 1
  
for word, count in frequency.items():    
    print(f"{word} - {count}")

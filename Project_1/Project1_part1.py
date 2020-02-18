#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:19:51 2019

@author: Marcel
"""
a = 1
b = 2
print(b+1)
c = a/(b+1)
print(c)
text = "Result:"
print(text, c)


for i in [1, 2, 3, 4, 5, 6]:
    print(i)

print("Done")


for i in [1, 2, 3, 4, 5, 6]:
    if i % 2 == 0:
        print("Even:", i)
    else:
        print("Odd:", i)
print("Done")


iliad = 'Sing, O goddess, the anger of Achilles son of \
 Peleus, that brought countless ills upon the Achaeans.'
   
print(iliad)


alphabet = "abcdefghijklmnopqrstuvwxyz"
print(alphabet[0]) # ’a’
print(alphabet[1]) # ’b’
print(alphabet[25]) # ’z’

k =', '.join(['abc', 'def', 'ghi'])
print(k)
print(k.upper())


text_vowels = ''
for i in iliad:
    if i in 'aeiou':
        text_vowels += i
print(text_vowels)   # ’ioeeaeoieooeeuaououeiuoeaea’


print(alphabet[-3:])


#begin = 'my''{} string {}'.format(begin, 'is empty') # ’my string is empty’

var1 = 3.14
var2 = 'my string'
list3 = [1, var1, 'Prolog', var2]
print(list3)

list3[1:3]        # [3.14, ’Prolog’]
list3[1:3] = [2.72, 'Perl', 'Python']
list3             # [1, 2.72, ’Perl’, ’Python’, ’my string’]
   
#list4 = [list2, list3]
       # [[1, 8, 3], [1, 2.72, ’Perl’, ’Python’, ’my string’]]
       
wordcount = {}
wordcount['A']= 10
wordcount['the'] = 'hello'
wordcount['test'] = 21.1
print(wordcount['A'])

print(wordcount.get('A'))
print(wordcount.keys())
print(wordcount.values())
print(wordcount.items())

""" 1.6.8 """
letter_count = {}
for letter in iliad.lower():
    if letter in alphabet:
        if letter in letter_count:
            letter_count[letter] +=1
        else:
            letter_count[letter] = 1


print(letter_count.items())

for letter in sorted(letter_count.keys()):
    print(letter, letter_count[letter])

for letter in sorted(letter_count.keys(), 
                     key=letter_count.get, reverse=True):
    print(letter, letter_count[letter])


""" 1.7.1 """
digits = '0123456789'
punctuation = '.,;:?!'
char = '.'
if char in alphabet:
    print('Letter')
elif char in digits:
    print('Number')
elif char in punctuation:
    print('Punctuation')
else:
    print('Other')
    

sum = 0
for i in range(100):
    sum += i
print(sum)


for idx, letter in enumerate(alphabet):
    print(idx, letter)
    
""" 1.7.3 """

sum, i = 0, 0
while True:
    sum += i
    i += 1
    if i >= 100:
        break
print(sum)

""" 1.7.4 """

try:
    int(alphabet)
    int('12.0')
except ValueError:
    print('Caught a value error!')
except TypeError:
    print('Caught a Type error')


def count_letters(text, lc=True):
       letter_count = {}
       if lc:
           text = text.lower()
       for letter in text:
           if letter.lower() in alphabet:
               if letter in letter_count:
                   letter_count[letter] += 1
               else:
                   letter_count[letter] = 1
       return letter_count    


print(count_letters(iliad, True))

""" 1.9.1 Comprehensions """

word = 'acress'
splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
deletes = [a + b[1:] for a, b in splits if b]

splits_generator = ((word[:i], word[i:])
    for i in range(len(word) + 1))


""" 1.9.2 Generators """

for i in splits_generator: print(i)

def splits_generator_function():
       for i in range(len(word) + 1):
           yield (word[:i], word[i:])


""" 1.9.3 Iterators """

latin_alphabet = 'abcdefghijklmnopqrstuvwxyz' 
print(len(latin_alphabet)) # 26
greek_alphabet = 'αβγδεζηθικλμνξοπρστυφχψω' 
print(len(greek_alphabet)) # 24
cyrillic_alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' 
print(len(cyrillic_alphabet)) # 33


la_gr = zip(latin_alphabet[:3], greek_alphabet[:3])
la_gr_cy = zip(latin_alphabet[:3], greek_alphabet[:3],cyrillic_alphabet[0:3])

la_gr.__next__()
la_gr.__next__()
la_gr.__next__()

la_gr_cy_list = list(la_gr_cy)

list2iter = iter(la_gr_cy_list)

""" 1.10 Modules """
import math

print(math.sqrt(2))
print(math.sin(math.pi/2))
print(math.log(8, 2))

import statistics as stats

stats.mean([1, 2, 3, 4, 5])   # 3.0
stats.stdev([1, 2, 3, 4, 5])  # 1.5811388300841898

if __name__ == ’__main__’:
       print("Running the program")
       # Other statements
   else:
       print("Importing the program")
       # Other statements
       
""" 1.12 Basic File Input/Output """

f_iliad = open('iliad.txt', 'r')
iliad_txt = f_iliad.read()
f_iliad.close()
iliad_stats = count_letters(iliad_txt)  # count the letters
with open('iliad_stats.txt', 'w') as f:
    f.write(str(iliad_stats)) # we automatically close the file
    
""" 1.13.1 Memo Functions """

def fibonacci(n):
    if n == 1: return 1
    elif n == 2: return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

f_numbers = {}
def fibonacci2(n):
    if n == 1: return 1
    elif n == 2: return 1
    elif n in f_numbers:
        return f_numbers[n]
    else:
        f_numbers[n] = fibonacci2(n - 1) + fibonacci2(n - 2)
        return f_numbers[n]

""" 1.13.2 Decorators """

def memo_function(f):
       cache = {}
       def memo(x):
           if x in cache:
               return cache[x]
           else:
               cache[x] = f(x)
               return cache[x]
return memo

fibonacci = memo_function(fibonacci)
@memo_function
#def fibonacci(n):
#...


""" Classes """

class Text:
    """Text class to hold and process text """
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
       
    def __init__(self, text=None):
        """The constructor called when an object
        is created"""
        self.content = text
        self.length = len(text)
        self.letter_counts = {}
        
    def count_letters(self, lc=True):
        """Function to count the letters of a text"""
    
        letter_counts = {}
        if lc:
            text = self.content.lower()
        else:
            text = self.content
        for letter in text:
            if letter.lower() in self.alphabet:
                if letter in letter_counts:
                    letter_counts[letter] += 1
                else:
                    letter_counts[letter] = 1
        self.letter_counts = letter_counts
        return letter_counts

txt = Text("""Tell me, O Muse, of that many-sided hero who
traveled far and wide after he had sacked the famous town
of Troy.""")

print(type(txt))
print(txt.length)
print(txt.count_letters())

Text.__doc__    # ’Text class to hold and process text’
#Text.count_letters.__doc__# ’Function to count the letters of a text’
help(Text)

class Word(Text):
       def __init__(self, word=None):
           super().__init__(word)
           self.part_of_speech = None
       def annotate(self, part_of_speech):
           self.part_of_speech = part_of_speech

word = Word('Muse')

import tensorflow as tf
hello = tf.constant("hello TensorFlow!")
sess=tf.Session()
print(sess.run(hello))




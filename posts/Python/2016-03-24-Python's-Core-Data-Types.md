|Object type|Example literals/creation
|-----|:----:|
|Numbers  | 1234, 3.1415, 9999L, Decimal, 3+4j|
|Strings  | 'apple', "Python's Data types"|
|Lists |[1, [2, 'three'], 4]|
|dict|{'food':'spam', 'taste':'yum'}|
|tuple  | (1, 'spam', 4, "U")|
|Files  | file = open('config.txt', 'r')|
|other types | sets, types, None, Booleans|

There are no type declarations in Python.

## Strings

### Sequence Operations

{%highlight Python%}
S = 'Spam'
print len(S)   #4
print S[0]     #S
print S[-1]    #m
print S[1:3]   #pa
print S[1:]    #pam
print S[:3]    #Spa
print S[:-1]   #Spa
print S[:]     #Spam
print S*8      #SpamSpamSpamSpamSpamSpamSpamSpam
{%endhighlight%}

Note: S[left:right] will print S[left]~S[right-1]

### Immutability

We cannot change a string by assigning to one of its position, but we can build a new one and assign it to the same name.Becase Python cleans up old objects as we go, this isn't as inefficient as it may sound.

{%highlight Python%}
S[0] = 'z'     #error
S = 'z'+S[1:]
{%endhighlight%}

### Type-Specific Methods

{%highlight Python%}
print S.find('pa')  #1
print S.replace('pa', 'XYZ')  #SXYZm
print S             #Spam
{%endhighlight%}

other methods

{%highlight Python%}
split()
upper()
isalpha()
rstrip()  #java trim()
{%endhighlight%}

### Pattern Matching

{%highlight Python%}
import re
match = re.match('/(.*)/(.*)/(.*)', /usr/share/application)
print match.groups()
{%endhighlight%}

## Lists

### Sequence Operations
Lists support all sequence operations we discussed for strings.

{%highlight Python%}
L = [123, 'spam', 1.23]
print len(L)
print L[0]
print L[:-1]
print L+[4,5,6]
print L        #  [123, 'spam', 1.23], immutability
{%endhighlight%}

We should notice that lists are not immutable unlike strings. Which means

{%highlight Python%}
L[1] = 2    # this assignment is right
{%endhighlight%}

### Type-Specific Operations
Python's list are related to arrays in other languages. But lists have no fixed type constraint.

{%highlight Python%}
L.append('NI')
print L
L.pop(2)    #this method will return the item and then remove it from the list
print L
L.sort()
print L    #[1.23,123,'spam']
L.reverse()
print L
{%endhighlight%}

### Bounds Checking
Although Python's list are powerful than other languages' array, we should be careful about the bound of list.

### Nesting

{%highlight Python%}
M = [[1,2,3],
     [4,5,6],
     [7,8,9]]
print M[1]        #[4,5,6]
print M[1][2]     #6
{%endhighlight%}

### List Comprehensions

{%highlight Python%}
col2 = [row[1] for row in M]
print col2
print M
{%endhighlight%}

## Dictionaries
Python's dictionaries are similar with the mappings in other languages.

### Mapping Operations

{%highlight Python%}
D = {'food':'Spam', 'quantity':4, 'color':'pink'}
print D['food']
D['quantity'] += 1
print D
{%endhighlight%}

### Nesting Revisited

{%highlight Python%}
rec = {'name':{'first':'Bob', 'last':'Smith'},
        'job':['dev','mgr'],
        'age':40}
print rec['name']     #'name' is a nested dictionaty
print rec['name']['first']
print rec['job']      # 'job'is a nested list
rec = 0               # the object's space is reclaimed
{%endhighlight%}

Technically speaking, Python has a feature known as *garbage collection* that cleans up unused memory as our program runs and frees us from having to manage such details in our code.

### Sorting Keys: for Loops
Dictionaries are not sequences, so they don't maintain any dependable left-to-right order. If we want to sort, we can

{%highlight Python%}
D = {'a':1, 'b':3, 'c':2}
Ks = D.keys()      #return a list
Ks.sort()
for key in Ks:
    print key, '=>', D[key]

for key in sorted(D):
    print key, '=>', D[key]
{%endhighlight%}

### Iteration and Optimization

{%highlight Python%}
squares = [x**2 for x in [1,2,3,4,5]]
print squares
squares = []
for x in [1,2,3,4,5]:
    squares.append(x**2)
print squares
{%endhighlight%}

The list comprehension will run faster.

### Missing Keys: if Tests

{%highlight Python%}
D = {'a':1, 'b':2, 'c':3}
D['e'] = 99         #right
print D['f']        #error
D.has_key('f')      #boolean fucntion
{%endhighlight%}

we can assign to a new key to expand a dict, but fetching a nonexistent key is still a mistake.

## Tuples
Tuple is roughly like a list that cannot be changed——tuples are immutable like strings.

{%highlight Python%}
T = (1,2,3,4,5)
print T
print T+(5,6)
T[0] = 2   #error
{%endhighlight%}

Tuple are not used as often as lists, but in some cases tuple may be useful.

## Files

{%highlight Python%}
f = open('data.txt', 'w')
f.write('Hello world\n')
f.close()

f = open('data.txt')
bytes=f.read()
print bytes
{%endhighlight%}

## Other Core Types

#### Set

{%highlight Python%}
X = set('spam')
print X
Y = set(['h','a','m'])
print Y
print X&Y           #set(['a','m']), intersection
print X|Y           #Union
print X-Y           #Difference, set(['s', 'p'])
{%endhighlight%}

### User-Defined Classed

{%highlight Python%}
class Worker:
    def __init__(self, name, pay):
        self.name = name
        self.pay = pay

    def lastName(self):
        return self.name.split()[-1]

    def giveRaise(self, percent):
        self.pay *= (1.0+percent)

bob = Worker('Bob Smith', 50)
print bob.lastName()
bob.giveRaise(.10)
print bob.pay
{%endhighlight%}

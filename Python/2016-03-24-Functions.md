---
layout:  post
title:  "Functions"
date:	2016-03-24 20:54:43 +0800
categories: [Python]
---
## An Example

{%highlight Python%}
def <name>(arg1, arg2, ..., argN):
    <statement>

def times(x, y):
    try:
        y = int(y)
    except:
        print y, "was no digit"
    else:
        return x*y

print times(2,3)
print times('fad','fds')
print times(2.3, 4.11)
print times('dfs', '4')
{%endhighlight%}
This function works on both numbers and sequences. This function depends on what we pass into it.

## Scope Rules
>1.Names defined inside a def can only be seen by the code within that def. You cannot even refer to such names from outside the function.

>2.Names defined inside a def do not clash with variables outside the def , even if the same names are used elsewhere. A name X assigned outside a given def (i.e.,in a different def or at the top level of a module file) is a completely different variable from a name X assigned inside that def .

## Special Argument-Matching Modes
By default, arguments are matched by position, form left to right, we can also specify matching by name, default value, and collectors for extra arguments.

1.Positionals: Matched from left to right
> The normal case, which we’ve been using so far, is to match arguments by position.

2.Keywords: matched by argument name
>Callers can specify which argument in the function is to receive a value by using the argument’s name in the call, with the name=value syntax.

3.Defaults: specify values for arguments that aren’t passed
>Functions can specify default values for arguments to receive if the call passes too few values, again using the name=value syntax.

4.Varargs: collect arbitrarily many positional or keyword arguments
>Functions can use special arguments preceded with * characters to collect an arbitrary number of extra arguments (this feature is often referred to as varargs, after the varargs feature in the C language, which also supports variable-length argument lists).

5.Varargs: pass arbitrarily many positional or keyword arguments
>Callers can also use the * syntax to unpack argument collections into discrete, separate arguments. This is the inverse of a * in a function header—in the header it means collect arbitrarily many arguments, while in the call it means pass arbitrarily many arguments.

### Keywords and Defaults

{%highlight Python%}
# position
def f(a,b,c): print a,b,c
f(1,2,3)

# keywords
f(c=3, b=2, a=1)
f(1, c=3, b=2)

# defaults
def f(a, b=2, c=3):print a,b,c
f(1)
f(1,4)
f(1,4,5)
f(1,c=6)
{%endhighlight%}

### Arbitrary Arguments Examples

#### Collecting arguments
The last two matching extensions, * and ** , are designed to support functions that take any number of arguments.
The * feature collects unmatched positional arguments into a **tuple**

{%highlight Python%}
def f(*args): print args
f(1)
f(1,2,3,4)
{%endhighlight%}

The ** feature is similar, but it only works for keyword arguments, it collects them into a new dictionary.
{%highlight Python%}
def f(**args): print args
f(a=1,b=2)
{%endhighlight%}

#### Unpacking arguments
{%highlight Python%}
def func(a, b, c, d): print a, b, c, d

args = (1, 2)
args += (3, 4)
func(*args)
{%endhighlight%}

## Argument Matching: The Gritty Details
1. Assign nonkeyword arguments by position.

2. Assign keyword arguments by matching names.

3. Assign extra nonkeyword arguments to *name tuple.

4. Assign extra keyword arguments to **name dictionary.

5. Assign default values to unassigned arguments in header.

## Lambda Expression
{%highlight Python%}
lambda arg1, arg2,..,argN: a single expression using arguments
{%endhighlight%}
Notice that the body of lambda is a single expression, not a block of statements
{%highlight Python%}
def func(x,y,z):return x+y+z
func(2,3,4)

f = lambda x,y,z:x+y+z
f(2,3,4)
{%endhighlight%}

Lambda expression tends to be simpler when you just want to embed small bits of executable code. And lambdas are also commonly use to code jump tables, which are list or dictionaries of actions to be performed on demand.
{%highlight Python%}
# switch....case
print {'a':(lambda x:x**2),
 'b':(lambda x:x**3)}['a'](5)
{%endhighlight%}

## Applying Functions to Arguments
Some programs need to call arbitrary functions in a generic fashion, without knowing their names or arguments ahead of time (we’ll see examples of where this can be useful later). Both the apply built-in function, and some special call syntax available in Python, can do the job.

### The apply Built-in
{%highlight Python%}
f = lambda x,y,z:x+y+z
apply(f, (2,3,4))
{%endhighlight%}
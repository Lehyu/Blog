## The Case of the Missing Declaration Statements
As we can see, we have been using variables without declaring their types. This feature is different from other languages which we have to declare variables' type before using them. In Python, types are determined automatically at runtime, not in response to declarations in our code.

## Variables, Objects, and References
A variable is created when our code first assigns it a value. We should notice that a variable never has any type information, it's just a reference of object. Which means, types live with objects, not variables. When a variable appears in an expression, it is immediately replaced with the object that it **currently** refers to.

## Objects Are Garbage-Collected
In Python, whenever a variable is assigned to a new object, the space held by the prior object is reclaimed(if it's not referenced by any other variable). This mechanism is known as garbage collection.

{%highlight Python%}
x = 42
x = 'spam'  # reclaim 43 now
{%endhighlight%}

Python accomplishes this feat by keeping a counter in every object that keeps track of the number of references currently pointing to that object. As soon as the counter drops to zero, the object's memory space is automatically reclaimed. This mechanism free us from cleaning up unused space in our program.

## Share References

{%highlight Python%}
a = 3
b = a
a = 'spam'
{%endhighlight%}

This is simple, we can easily figure out that b=3, the object 3 wouldn't be reclaimed after variable a change its reference. But what will happens if the object is list which can change in-place item

{%highlight Python%}
# case 1
L1 = [1,2,3]
L2 = L1
L1 = 24
print L2        #[1,2,3]

# case 2
L1 = L2
L1[0] = 3
print L2         #[3,2,3]
{%endhighlight%}

In case 1, we changed variable L1's reference so it wouldn't influence L2, but in case 2, L1 still refered to list[1,2,3], so when it changed the object's item value, the value of L2 changed too(literary, because the reference of L2 didn't change, what has changed is the list object).
But some time, we want to assign a object that one variable refers to to another variable. And the two variable refer to different object which has the same value. We make a copy.

{%highlight Python%}
L1 = [1,2,3]
L2 = L1[:]
L1[0] = 24
print L1
print L2
{%endhighlight%}

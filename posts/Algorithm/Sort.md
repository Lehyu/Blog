## Sort
Select sort is the simplest sorting alogrithms.

### IDEA

1.find the smallest element in the rest of array
2.exchange the element with with the *i* th entry.
3.repeat step1 and step2 util all item is sorted.

### Pseudo Code

{%highlight C++%}
for i = [0,len)
   min = i
   for j = (i,len)
      if less(src[min], src[j]) != 1
	 min = j
   exch(src,i, min)
{%endhighlight%}

### Analysis

1.Selection sort uses pow(n,2)/2 compares and n exchanges.
2.Running time is insensitive to input.No matter the array is in order or not, the running time is direct proportion to pow(n,2)/2.

## Insertion Sort

### IDEA

This sorting alogrithm is similar to insert cards.Each time, we treat the front array as a in order items.And when a item is coming, we compare it with the front items, if it is smaller than the item compared,the the item compared move back. Until we find the item that is smaller than the coming item, we inset the coming item after that item.
(The algorithm that people often use to sort bridge hands is to con-
sider the cards one at a time, inserting each into its proper place among those already
considered (keeping them sorted). In a computer implementation, we need to make
space to insert the current item by moving larger items one position to the right, before
inserting the current item into the vacated position)

### Pseudo Code

{%highlight C++%}
for i = [1,n)
  key = src[i]
  for j = [i-1,0]
      if key < src[j]
	  src[j+1] = src[j];
      else
	  brek
  src[k] = key.
{%endhighlight%}

### Analysis

Unlike the selection sort, the running time of insertion sort depends on the inital order of the item in the input.
The worst case of insertion sort is pow(n,2)/2 compares and pow(n,2)/2 exchange,but the best case is n-1 compares and no exchange.The worst case and best case can be easily proved.On the average,we can get the formula as fellow:

$$(0+1)/2 + (0+1+2)/3 + ...+(0+1+...+(n-1))/n = n^2/4$$

Insertion sort works well for certain types of nonrandom arrays that often arise in practice, even if they are huge.
Insertion sort is an excellent method for partially sorted arrays and is also a fine method for tiny arrays.

## Shellsort

Shellsort is a sorting alogrithm based on insertion sort.

### IDEA

In my opinion,shellsort divide the array into **h**th part,first step is to sort each part by using insertion sort,after that,the origin array will become a partially sorted array,then it is suitable for the origin insertion sort to sort it.

### Pseudo Code
{%highlight C++%}
while h >= 1
  for i = [h,n)
    for j = i to h reduce h each time
      if less(src[j], src[j-h])
	 exch(src,j,j-h)
  h /= k;
{%endhighlight%}

### TODO Analysis

## MERGESORT

### IDEA

The most important idea: combining two ordered arrays to make one larger ordered arrays.
Thus,we have a new question,how can we divide a random-order array into two ordered arrays? We divide each new arrays into two parts util the arrays can't be divide, which only has one elements,and obviously that two arrays which have only one item are an ordered arrays.So, we can slove the problem.

### Pseudo Code

{%highlight C++%}
merge(src, lower, mid, upper)
  for i = [lower, upper]
    tmp[i] = src[i]
  i = lower
  j = mid+1
  for k = [lower, upper]
    if i > mid:  src[k] = tmp[j++]
    else if j > upper: src[k] = tmp[i++]
    else if less(src[i], src[j]): src[k] = tmp[i++]
    else :src[k] = tmp{j++}

mergesort(src, lower, upper)
  if lower < upper:
    mid = (lower+upper)/2
    mergesort(src, lower, mid)
    mergesort(src, mid+1, upper)
    merge(src, lower, mid, upper)
{%endhighlight%}

### Analysis

Mergesort is one of the best-known example of *divide-and-conquer* paradigm for efficient alogrithm design.
Each time combine two arrays to a larger ordered array, the number of compares is larger than or equal to N/2 and smaller than or equal to N(assume that the number of items that two arrays contain is N).
And T(N) is the running time of mergesort.Then we can have

$$\begin{equation}
\begin{array}{rcl}
2T(N/2)+N/2 <= T(N) <= 2T(N/2) + N
\end{array}
\end{equation}$$

In particular, when N = pow(2,n),then we have

$$\begin{equation}
\begin{array}{rcl}
T(2^n) &=& 2T(2^{n-1}) + 2^n \\
\Rightarrow T(2^n)/2^n &=& T(2^0)/2^0 + n \\
\Rightarrow T(N) &=& T(2^n) = n2^n = Nlog_2N
\end{array}
\end{equation}$$

### Improvement

1.Use insertion sort for small subarrays.Insertion sort is faster than mergesort for tinny arrays.
2.Test whether the array is already in order.Each time call *merge()* test whether *src\[mid\]* is smaller than or equal to *src\[mid+1\]* or not.
3.Eliminate the copy to the auxiliary array.(I haven't figured out yet)
4.To make *aux\[\]* array local to *mergesort()*, and pass it as an argument to *merge()*.Because no matter how tiny it is, it will still dominate the running time of mergesort.

### Summary

Mergesort is an asymptotically optimal compare-based sorting algorithm.

## QUICKSORT

### IDEA

1.Select an item to partition an array into two subarrays named *sub_left* and *sub_right*, every item in *sub_left* isn't greater than the partitioning item while every item in *sub_right* isn't less than the partitioning item.
2.sort *sub_left* and *sub_right*
Obviously, for *sub_left* and *sub_right* we can handle them just like their parent array.

### Pseudo Code
{%highlight C++%}
function partition(src, lower, upper)
   item = select(src, lower, upper)
   partition item to a suitable index, src[lower..index-1] <= src[index](=item) <= src[index+1...upper]
   return index

function quicksort(src, lower, upper)
   index = partition(src, lower, upper)
   quicksort(src, lower, index-1)
   quicksort(src, index+1, upper)
{%endhighlight %}

### Analysis

We can see the best case of quicksort would be each partitioning stage divides an array in exactly in half. But it's difficult to or hardly receive. On the average, let's assume C_N be the average compares of sorting an array which have N items. The we can have,

$$\begin{equation}
\begin{array}{rcl}
C_N &=& N+1+(C_0+C_1+...+C_{N-1})/N + (C_{N-1}+...+C_1+C_0)/N \\
\end{array}
\end{equation}$$

Simplification,

$$\begin{equation}
\begin{array}{rcl}
NC_N = (N+1)N + 2(C_0+C_1+...+C_{N-1})
\end{array}
\end{equation}$$

Also,

$$\begin{equation}
\begin{array}{rcl}
(N-1)C_{N-1} &=& N(N-1) + 2(C_0+C_1+...+C_{N-2}) \\
\Rightarrow NC_N - (N-1)C_{N-1} &=& 2N + 2C_{N-1} \\
\Rightarrow C_N &\simeq& 2(N+1)(\frac{1}{3}+\frac{1}{4}+...+\frac{1}{4}) \\
\Rightarrow lnN &=& 1+\frac{1}{2}+\frac{1}{3}+...+\frac{1}{N} \\
\Rightarrow C_N &\simeq& 2NlnN
\end{array}
\end{equation}$$

What's the worst case? Obviously, when each partitioning stage select the smaller item as the partitioning item and the item's index is lower index. Then the worst case comes.
$$C_N = N + (N-1)+...+2+1 = (N+1)N/2$$
Through three case, we can learn the most influence factor is the partitioning item/index.

### Improvement

#### Cutoff to Insertion Sort

As with most recursive alogrithms, it's not a good idea to use quicksort to sort tiny arrays. But obviously, there are always tiny subarrays to sort when we use recursive alogrithms. So, when the subarrays are tiny, we call insertion sort to sort them.

{%highlight C++%}
if(upper <= lower + M)
{
   insertionsort(src, lower, upper);
   return;
}
{%endhighlight%}

The optimum value of the cutoff M is system-dependent, but any value between 5 and 15 is likely to work well in most situations.

#### Median-of-three Partitioning

Median-of-three partitioning means find the median of *src\[lower\],src\[mid\],src\[upper\]* (of course, you can choose other three item). It can eliminate the non randomness in case of avoiding the worst case.

### TODO Entropy-optimal sorting

### Summary

In the MERGESORT section, we say "Mergesort is an asymptotically optimal compare-based sorting algorithm". But quicksort is substantially faster than any other sorting method in typical applications. Because Mergesort does not guarantee optimal performance for any given distribution of duplicates.

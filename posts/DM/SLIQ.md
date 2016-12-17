Before SLIQ, most classification alogrithms have the problem that they do not scale. Because these alogrithms have the limit that the traning data should fit in memory. That's why SLIQ was raised.

# Generic Decision-Tree Classification

Most decision-tree classifiers perform classification in two phases: **Tree Building** and **Tree Pruning**.

## Tree Building

{%highlight C++%}
MakeTree(Training Data T)
   Partition(T);

Partition(Data S)
   if(all points in S are in the same class) then return;
   Evaluate splits for each attribute A
   Use the best split found to partition S into S1 and S2
   Partition(S1);
   Partition(S2);
{%endhighlight%}


## Tree Pruning
As we have known, no matter how your preprocess works, there  always exist "noise" data or other bad data. So, when we use the traning data to build the decision-tree classification, it also create branches for thos bad data. These branches can lead to errors when classifying test data. Tree pruning is aimed at removing these braches from decision tree by selecting the subtree with the least estimated error rate.

# Scalability Issues

## Tree Building

As I mentioned, ID3/C4.5/Gini[1] is used to evaluate the "goodness" of the alternative splits for an attribute.

### Splits for Numeric Attribute

The cost of evaluating splits for a numeric attribute is dominated by the cost of sorting the values. Therefore, an important scalability issue is the reduction of sorting costs for numeric attributes.

### Splits for Categorical Attribute

## Tree Pruning

# SLIQ Classifier

To achieve this pre-sorting, we use the following data structures. We create a separate list for each attribute of the training data. Additionally, a separate list,called class list , is created for the class labels attached to the examples. An entry in an attribute list has two fields: one contains an attribute value, the other anindex into the class list. An entry of the class list also has two fields: one contains a class label, the other a reference to a leaf node of the decision tree. The *i* th entry of the class list corresponds to the i th example in the training data. Each leaf node of the decision tree represents a partition of the training data, the partition being defined by the conjunction of the predicates on the path from the node to the root. Thus, the class list can at any time identify the partition to which an example belongs. We assume that there is enough memory to keep the class list memory-resident. Attribute lists are written to disk if necessary.


[1]:http://www.cnblogs.com/mlhy/p/4856062.html

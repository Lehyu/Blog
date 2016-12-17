## Hidden Units

Issue: how to choose the type of hidden unit to use in the hidden layers of the model.

**Rectified linear units** are an excellent default choice of hidden unit. But many other types of hidden units are available. It's usually impossible to predict in advance which will work best.

### Rectified Linear Units and Their Generalizations

#### Rectified Linear Units

**ReLu** use the activation function g(z)=max{0,z}.

////

One drawback of ReLu is that they cannot learn via gradient-based method on examples for which their activation is zero...

#### Generalizations

Three generalizations: h=g(z,a)=max(0,z)+a*min(0,z)
1. Absolute value rectification fixes a = -1
2. leaky ReLu fixes a to a small value like 0.01
3. parametric ReLu treats a as a learnable parameter.

#### Maxout Units

Maxout Units generalize ReLu further. Maxout Units divide **z** into groups of k values. Each maxout unit then outputs then maximum element of one of these groups:

### Logistic Sigmoid and Hyperbolic Tangent

Their use as hidden units in **feedforward network** is now discouraged. But their use as output units is compatible with the use of gradient-based learning when an appropriate cost function can undo the saturation of the sigmoid in the output layer.

The reason why sigmoidal activation function is discouraged using as hidden units in feedforward network is that sigmoidal activation function saturate when input is very positive or very negative(their gradient are 0). But sigmoidal units more appealing in **RNN** than linear activation functions.

### Other Hidden Units

1. linear hidden uni
2. Softmax units(used as output but may sometimes used as hidden units)

## Architecture Design

## Back-Propagation and Other Differentiation Algorithms
//todo

### Some Questions
1. In gradient-based learning, how to initilize so that we can have a good solution?
2. ReLU can not learn via gradient-based methods on examples for which their activation is 0. What's the effect? how to address this problem?
3. In linear hidden units, how can it reduces the number of parameters in a network?

## Convolutions

### Why We Use Convolutions

Learning features with large features that span the entire image(fully connected networks) is very computationally expensive. So we want another model, locally connected networks, that restrict the connections between the hidden units and the input units. Neurons in the visual cortex have localized receptive fields, so we propose convolutions.

### How Convolutions Work

Given some large $r \times c$ images, learning $k$ $a \times b$ features map.Then we will have a $k \times (r-a+1) \times (c-b+1)$ array of convolved features.

![Convolutions](https://raw.githubusercontent.com/lehyu/lehyu.cn/master/image/DL/CNNs/Convolution_schematic.gif)

## Pooling

### Why We Use Pooling

The same reason why we use locally connected network not fully connected network. After we apply convolutions, we would have $k \times (r-a+1) \times (c-b+1)$ features. If $k, r,c$ are large enough, this can be computationally challenging.

The another reason is if the inputs is too large, it can be prone to over-fitting.

### Why Pooling Works

Images have the stationarity property which implies that features that are useful in one region are also likely to be useful for other regions. Thus, to describe a large image, one natural approach is to aggregate statistics of these features at various location. This aggregation operation is called pooling. There are many way to aggregate such mean or max operation.

### How Pooling Works

There are several kinds of ways to do this thing. In fact, we can treat it as a non-overlap convolution.

![Pooling](https://raw.githubusercontent.com/lehyu/lehyu.cn/master/image/DL/CNNs/Pooling_schematic.gif)

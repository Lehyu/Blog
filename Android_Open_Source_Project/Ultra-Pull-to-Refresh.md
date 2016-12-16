---
layout:  post
title:  "Ultra Pull to Refresh"
date:	2016-06-16 10:54:43 +0800
categories: [AOSP, View]
---

## PtrHandler

It's used to handler the behavior of content view when refreshing or whether can refresh or not.

## PtrUIHandler

It's used to handle the behavior of header view when users are pulling or users release after pulling.

## PtrUIHandlerHolder

This is a single linked list to wrap **PtrUIHandler**. It's a manager to manager all PtrUIHandlers.

## PtrFrameLayout
This class extends **ViewGroup**, and this container must only contain two views, one is header view, another is content view.
### onMeasure




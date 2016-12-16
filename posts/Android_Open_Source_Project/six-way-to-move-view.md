---
layout:  post
title:  "Six ways to settle views when they are moved"
date:	2016-06-16 10:54:43 +0800
categories: [AOSP, View]
---

## layout(int l, int t, int r, int b)
(l, t, r, b) is the target position after views are moved.

{%highlight Java%}

public boolean onTouchEvent(MotionEvent event) {
    //current (x,y)
    int x = (int) event.getX();
    int y = (int) event.getY();
    switch(event.getAction()){
      case MotionEvent.ACTION_DOWN:
        lastX = x;
        lastY = y;
      break;
      case MotionEvent.ACTION_MOVE:
        //offset
        int offX = x - lastX;
        int offY = y - lastY;
        //use layout() to settle view
        layout(getLeft()+offX, getTop()+offY,
          getRight()+offX  , getBottom()+offY);
      break;
    }
    return true;
  }

{%endhighlight%}

## offsetLeftAndRight(int offset) and offsetTopAndBottom(int offset)

offset can be negative or positive. So layout can be represented to

{%highlight Java%}

offsetLeftAndRight(offset);
offsetTopAndBottom(offset);

{%endhighlight%}

These two method would be useful when we just want to horizontal or vertical move view.

## scrollBy(int offsetX, int offsetY) and scrollTo(int targetX, targetY)

Obviously, we can tell the difference of these two method by watching their api.

## LayoutParams

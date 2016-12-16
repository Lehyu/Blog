---
layout:  post
title:  "Android Studio Failed to start emulator"
date:	2015-09-16 20:54:43 +0800
categories: [Android Studio, Error]
---

I'm running a 64-bit system, but the emulator is 32-bit, you can solve this problem

## Solution 1

{%highlight shell%}
sudo apt-get install ia32-libs lib32ncurses5 lib32stdc++6
{%endhighlight%}

## Solution 2

{%highlight Shell%}
mv emulator emulator-backup
ln -s emulator64-arm emulator
{%endhighlight%}
---
layout:  post
title:  "A First Glance at Django(三)"
date:	2016-03-28 10:10:00 +0800
categories: [Python, Django]
---

## Overview
A view is a "type" of Web page in our Django application just like GUI(I think). In Django, web pages and other content are delivered by views. Each view is represented by a simple Python function(or method, class-based view). Django will choose a view by examining the UTL that's requested.

A URL pattern is simply the general form of a URL, such as **/newsarchive/year/month/**.

## Change HTML Page
We create a html page under **\<appname\>/templates**, such as **index.html**, how could we use this page in our views? we can

{% highlight Python%}
#method 1
from django.http import HttpResponse
from django.template import loader
def index(request):
   template = loader.get_template('index.html')
   context = {} #your context
   return HttpResponse(template.render(context, request))
{% endhighlight%}

You may find that it's a little bit complicated, but don't worry Django provides ad shortcut named **render()**

{% highlight Python%}
#method 1
from django.shortcuts import render

def index(request):
   return render(request, 'index.html', context)
{% endhighlight%}

##Raising a 404 error
When there's no object, then we can use `raise Http404("<error>")`, or you can use `get_object_or_404()` method.

If you want to learn more, you can find out [here](https://docs.djangoproject.com/en/1.9/intro/tutorial03/).

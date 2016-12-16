---
layout:  post
title:  "A First Glance at Django(一)"
date:	2016-03-27 13:05:00 +0800
categories: [Python, Django]
---

# Getting Started

## Introduction to Django
Django is written in Python, a general purpose language that is well suited for developing web applications. And Django loosely follows a model-view-controller design pattern, which greatly helps in building clean and maintainable web application.

## Installing Python 
If you use Linux or Unix, you have already have Python installed.

## Installing Django on Linux

{%highlight Python%}
sudo apt-get install python-django
django-admin --version
{%endhighlight%}

or

{%highlight Python%}
sudo apt-get install python3-pip
pip install Django==1.9.4
{%endhighlight%}



## Creating an Empty Project
Open a terminal and type the following command

{%highlight Python%}
django-admin startproject <our project's name>
{%endhighlight%}

We will see there are several files in the project

|File Name  | File Description|
|-----------|-----------------|
|\_\_init\_\_.py|Django projects are Python packages, and this file is required to tell Python that the folder is to be treated as a package. A package in Python's terminology is a collection of modules, and they are used to group similar files together and prevent naming conflicts.|
|manage.py|This is the main configuration file for your Django project. In this file you can specify a variety of options, including the database settings, site languages, which Django features are to be enabled, and so on.|
|settings.py|This is the main configuration file for your Django project. In this file you can specify a variety of options, including the database settings, site languages, which Django features are to be enabled, and so on.|
|urls.py|This is another configuration file. You can think of it as a mapping between URLs and Python functions that handle them. |
|wsgi.py|An entry-point for WSGI-compatible web servers to server yout project|

## The development server
Let;s verify our Django projects works

{%highlight Python%}
python manage.py runserver
{%endhighlight%}

> **note**
>  we may see the warning about unapplied migrations, that's because we hadn't deal with the database yet.

By default, the *runserver* commands starts the server on the internal IP at port 8000, we can change the port by

{%highlight Python%}
python manage.py runserver 8080 [IP:][port]
{%endhighlight%}

## Creating An App
Django comes with a utility that automatically generates basic directory structure of an app, so we can focuse on writing code rather than creating directories.

{%highlight Python%}
python manage.py startapp <our app's name>
{%endhighlight%}

And it will create serval files in

{%highlight Python%}
polls/
    __init__.py
    admin.py
    migrations/
	     __init__.py
    models.py
    tests.py
    view.py
{%endhighlight%}

### Write our First View
First, we open the file named *views.py* under our app's dir and code

{%highlight Python%}
from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def index(request):
    return HttpResponse("Hello world. You're at the timecol index")
{%endhighlight%}

Second, we map it to a URL, so we create a file named *urls.py* under our app's dir

{%highlight Python%}
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
]
{%endhighlight%}

The final step is to point the root URLconf  at the \<appname\>.urls module which edited in *\<projectname\>/urls.py*.

{%highlight Python%}
from django.conf.urls import url, include
from django.contrib import admin

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^timecol/', include('timecol.urls')),
]
{%endhighlight%}




---
layout:  post
title:  "A First Glance at Django(二)"
date:	2016-03-27 15:10:00 +0800
categories: [Python, Django]
---

We have learn how to create our app before. But there are still some configuration we should care about.

## Database Setup
By default, the configuration uses SQLite which is included in Python. If we want to use other database, we must install our own database such as PostgreSQL and configure it in **<our project'name>/settings.py**. Changing the fikkiwubg jets ub tge **DATABASE 'default'** item to match our database connection settings:

|name|config|
|----|-----|
|ENGING|either'django.db.backends.sqlite3', 'django.db.backends.postgresql', 'django.db.backends.mysql' or 'django.db.backends.oracle'|
|NAME|The name of our database. If we're using SQLite, the database will be a file on our computer(**os.path.join(BASE_DIR, 'db.sqlite3')**); other case, **NAME** should be the full absolute path|

For more details, see reference documentation for [DATABASES](https://docs.djangoproject.com/en/1.9/ref/settings/#std:setting-DATABASES).

There's an example:

{%highlight Python%}
DATABASES = {
    'default': {
        #'ENGINE': 'django.db.backends.sqlite3',
        #'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
        'ENGINE': 'django.db.backends.mysql',
        'NAME':'timecol',
        'USER':'root',
        'PASSWORD': 'root',
    }
}
{%endhighlight%}

The **INSTALLED_APPS** setting holds the names of Django applications that are activated in this Django instance. And those apps may make use of at least one database table, but if we need to use the tables. we should

{%highlight Python%}
python manage.py migrate
{%endhighlight%}

This command would create any necessary database tables according to the settings.

We may see there are several tables in our database **timecol** by using `show tables`.

## Creating Models
Models contain the essential fields and behaviors of the data we're storing. In our current app, we'll create two model: User and Medias.

{%highlight Python%}
from django.db import models

# Create your models here.

class User(models.Model):
    user_name = models.CharField(max_length=100)
    user_pswd = models.CharField(max_length=20)


class Medias(models.Model):
    user = models.ForeignKey(User)
    # the json file path
    json = models.CharField(max_length=100)
{%endhighlight%}

## Activating Models
The model code gives Django a lot of information. With it Django is able to

>create a database schema for this app

>create a Python database-access API for accessing User and Medias

But before that, we should tell Django by edit **settings.py**

{%highlight Python%}
INSTALLED_APPS = [
    'timecol.apps.TimecolConfig',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]
{%endhighlight%}

Now Django knows to include the app. But we still have to run several commands to create tables in our database

{%highlight Python%}
python manage.py makemigrations timecol  #out a id
python manage.py sqlmigrate timecol id
python manage.py migrate
{%endhighlight%}

If we change our models, we should tell

{%highlight Python%}
python manage.py makemigrations timecol
python manage.py migrate
{%endhighlight%}

## Playing with the API
In this section, we only tell how to go into the shell that we can test the API Django created. If you want to lean more, you can explore [database API](https://docs.djangoproject.com/en/1.9/topics/db/queries/)

{%highlight Python%}
python manage.py shell
{%endhighlight%}

## Creating an admin user
First we'll need to create a admin user.

{%highlight Python%}
python mange.py createsuperuser
{%endhighlight%}

Second, run server, and got to [http://127.0.0.1:8000/admin/](http://127.0.0.1:8000/admin/).

Third, make the app modifiable in the admin. To gain this goal, we should edit **admin.py**

{%highlight Python%}
from django.contrib import admin

# Register your models here.
from .models import User, Medias


admin.site.register(User)
admin.site.register(Medias)
{%endhighlight%}

Now we can modify these two models in admin site.


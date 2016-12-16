---
layout:  post
title:  "UnicodeEncodeError: 'latin-1' codec can't encode character"
date:	2016-04-14 17:00:00 +0800
categories: [Python, Mysql, Error, Settings]
---

## UnicodeEncodeError: 'latin-1' codec can't encode character

{%highlight shell%}
db = MySQLdb.connect()
dbc = conn.cursor()
db.set_character_set('utf8')
dbc.execute('SET NAMES utf8;')
dbc.execute('SET CHARACTER SET utf8;')
dbc.execute('SET character_set_connection=utf8;')
{%endhighlight %}


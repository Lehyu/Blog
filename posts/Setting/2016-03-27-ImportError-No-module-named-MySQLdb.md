## Installing MySQL for ubuntu

{%highlight shell%}
sudo apt-get install mysql-server
sudo apt-get install mysql-client
sudo apt-get install libmysqlclient-dev
{%endhighlight%}

After we finish the commands above, we can test whether *MySQL* installed successfully or not

{%highlight shell%}
sudo netstat -tap | grep mysql
{%endhighlight%}

## Using MySQL in Python
When it throw an error named `ImportError: No module named 'MySQLdb'`

{%highlight shell%}
pip3 install mysqlclient
{%endhighlight%}

## E: Could not get lock /var/lib/dpkg/lock

通过终端安装程序sudo apt-get cmd 出错：
E: Could not get lock /var/lib/dpkg/lock - open (11: Resource temporarily unavailable)
E: Unable to lock the administration directory (/var/lib/dpkg/), is another process using it?

解决的办法其实很简单：
在终端中敲入以下两句
sudo kill 刚才ps -aux | grep apt-get install 那个进程的pid
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock


## apt update更新慢
添加国内[镜像源](http://www.cnblogs.com/york-hust/p/5438031.html)

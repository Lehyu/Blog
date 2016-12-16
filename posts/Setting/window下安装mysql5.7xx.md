1. 解压mysql-5.7.xx压缩包
2. 配置path
3. 修改my-default.ini(datadir = ...\mysql\data)
4. 进入bin安装 mysqld -install
5. 初始化 mysqld --initialize-insecure(非常重要)
6. 启动 net start mysql

参考
[mysql install](http://dev.mysql.com/doc/refman/5.7/en/installing.html)

## Cuda7.5 安装
Cuda7.5不支持默认的 **g++** 版本，所以要降低版本以适应。
```
sudo apt install g++4.9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 20
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 10

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 20
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 10

sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30
sudo update-alternatives --set cc /usr/bin/gcc

sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
sudo update-alternatives --set c++ /usr/bin/g++
sh cuda.run --override
```

## Caffe安装
[Caffe installation](http://caffe.berkeleyvision.org/installation.html)

## tensorflow安装
[TensorFlow Installation](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html)

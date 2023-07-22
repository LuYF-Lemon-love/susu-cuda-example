# susu-cuda-example

PyTorch 与 CUDA 交互的例子。

## 安装

1. 创建虚拟环境:

```shell
python -m venv env
source env/bin/activate
which python
pip install --upgrade pip
```

2. 使用 pip 安装依赖:

```shell
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ninja -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 运行

方法一，使用即时编译（ninja）:

```shell
python main.py
```

方法二，使用 Setuptools 编译:

```shell
python setup.py install
python setuptools.py
```


## 参考

[1] [PyTorch自定义CUDA算子教程与运行时间分析](https://godweiyang.com/2021/03/18/torch-cpp-cuda/)
[2] [详解PyTorch编译并调用自定义CUDA算子的三种方式](https://godweiyang.com/2021/03/21/torch-cpp-cuda-2/)
[3] []()
[4] [NN-CUDA-Example](https://github.com/godweiyang/NN-CUDA-Example)
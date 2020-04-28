# Pytorch-MSCRED

这是使用PyTorch实现MSCRED

论文原文：
[http://in.arxiv.org/abs/1811.08055](http://in.arxiv.org/abs/1811.08055)

TensorFlow实现地址：
[https://github.com/7fantasysz/MSCRED](https://github.com/7fantasysz/MSCRED)

此项目就是通过上面tensorFlow转为Pytorch，具体流程如下：
- 先将时间序列数据转换为 image matrices

  > python ./utils/matrix_generator.py

- 然后训练模型并对测试集生成相应的reconstructed matrices

  > python main.py

- 最后评估模型，结果存在`outputs`文件夹中

  > python ./utils/evaluate.py
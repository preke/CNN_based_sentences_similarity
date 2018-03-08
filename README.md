# CNN based sentences similarity

这份代码基本是 [Detecting Semantically Equivalent Questions
in Online User Forums][1] 的实现，架构也是该paper的架构
原文说用 Keras 实现，我用 Pytorch 实现了一遍，大部分参考了 [CNN for sentence classification][2] 的`Pytorch`版本的代码。

删去了一些功能和操作，只留下cnn计算相似性的部分

整体架构如图：
![Snip20180308_4.png-76.7kB][3]


CNN 的部分，只用了一层卷积层，池化层
我没有drop out（也是因为和普通问题架构不同不敢贸然用）
最后一层全连接层计算相似度

还不是很会调参...

心得就是...

`Pytorch` 网络结构的搭建过程中，无论怎样自定义中间层的计算，一定要每个中间变量都要用 `Variable` 包起来，也就是要保证中间的计算过程都是`Variable` 类型的变量在参与，无论是函数的输入、返回值，还是运算符操作
这样才能够保证梯度，才能反向传播


  [1]: http://www.aclweb.org/anthology/K15-1013
  [2]: https://github.com/Shawn1993/cnn-text-classification-pytorch
  [3]: http://static.zybuluo.com/Preke/vkc49vk00s7pzpmr1u2yutha/Snip20180308_4.png

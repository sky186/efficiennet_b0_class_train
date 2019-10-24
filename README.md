# efficiennet_b0_class_train

修改代码，EfficientNet-PyTorch-master/examples/imagenet/main.py
1、 代码修改数据处理部分，读取图片，加入数据增强
2、climate.py，天气识别，分类，加入，BCE损失，多分类的二分类，79.5%，更换 adam 损失
3、yun_train.py ，正常的交叉熵损失训练，添加了数据增强部分，其他与作者相同
Reference:  https://github.com/lukemelas/EfficientNet-PyTorch

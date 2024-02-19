# U^2Net源码个人注解

[项目源码地址]: https://github.com/xuebinqin/U-2-Net

这是一篇发布于2020年，发布在CVPR上的一篇有关显著性目标检测（ Salient Object Detetion）的文章。作用根据官网提供的案例主要是提取图像中突出的区域。代码中还包含了有关于素描风格肖像画生成的有关代码。

**写在前面**：因为是本人第一个CV学习的项目，可能在某些地方的解释有些问题。另外注意本项目微调了部分代码，一般会注释解释是否替换（有可能忘了），可能替换成最新版本pytorch使用的写法，注意和源码区别。



## 文章目录

1 论文部分解析

2 源码注释

> 1.[u2net_train.py](u2net_train.py)
>
> 2.model/u2net.py
>
> 3.u2net_test.py
>
> 4.data_loader.py





## u2net_train.py

1、定义损失函数

```python
# ------- 1. define loss function --------
bce_loss = nn.BCELoss(reduction='mean')  # SOD任务中，只有前景和背景[0,1]，使用二值交叉熵损失
										 # reduction = 'mean'返回所有损失的平均值
def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)  # 融合side1~6的结果
    loss1 = bce_loss(d1, labels_v)  # side1
    loss2 = bce_loss(d2, labels_v)  # side2
    loss3 = bce_loss(d3, labels_v)  # side3
    loss4 = bce_loss(d4, labels_v)  # side4
    loss5 = bce_loss(d5, labels_v)  # side5
    loss6 = bce_loss(d6, labels_v)  # side6
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
  	# 打印每一个side预测图损失
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(),
        loss4.data.item(), loss5.data.item(), loss6.data.item()))
    return loss0, loss
```

这一部分是对于训练集中图片和标签集验证所使用的损失计算方法，有关于二值交叉熵的部分。在`u2net_train.py`的主训练部分会用到这一部分，打印出训练中的效果，用于判断模型训练是否收敛。一般会调用`loss0`也就是论文中提到的`side0`来作为结果输出。

> 这里提供一个新的损失部分的写法：
>
> ```python
> loss0 = F.binary_cross_entropy_with_logits(d0, labels_v)
> loss1 = F.binary_cross_entropy_with_logits(d1, labels_v)
> loss2 = F.binary_cross_entropy_with_logits(d2, labels_v)
> loss3 = F.binary_cross_entropy_with_logits(d3, labels_v)
> loss4 = F.binary_cross_entropy_with_logits(d4, labels_v)
> loss5 = F.binary_cross_entropy_with_logits(d5, labels_v)
> loss6 = F.binary_cross_entropy_with_logits(d6, labels_v)
> losses = [loss0, loss1, loss2, loss3, loss4, loss5, loss6]
> loss = sum(losses)
> ```



2、准备训练数据集

```python
# ------- 2. set the directory of training dataset 设置训练数据集的目录 --------
model_name = 'u2net'  # 'u2netp'    # 使用哪种模型进行训练？

data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)  # 获取项目数据目录
tra_image_dir = os.path.join('DUTS', 'DUTS-TR', 'im_aug' + os.sep)  # 图片集
tra_label_dir = os.path.join('DUTS', 'DUTS-TR', 'gt_aug' + os.sep)  # 验证集

image_ext = '.jpg'  # 训练数据集为jpg格式
label_ext = '.png'  # 训练标签集为png格式

# 获取网络模型的位置
model_dir = os.path.join(os.getcwd(), 'saved_models',model_name + os.sep)  

epoch_num = 400  # 训练epoch数，论文中设置的为10000
batch_size_train = 12  # 单次训练样本数，源代码提供的为12(1080ti)
batch_size_val = 1  # 这个参数并没有用上？
train_num = 0  # 统计训练集数量
val_num = 0 # 统计验证集数量

# 加入训练集，以jpg作为文件后缀
tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)  
tra_lbl_name_list = []  # 初始化标签集
for img_path in tra_img_name_list:
    img_name = img_path.split(os.sep)[-1]
    aaa = img_name.split(".")  # 通过"."分割文件名得到数组
    bbb = aaa[0:-1]  # 得到数组第一个和倒数第一个内容
    imidx = bbb[0]  # 取b[0]作为文件名
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]  # 拼接b[]组成文件名
	# 加入标签(验证)集，以jpg作为文件后缀
    tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)  

# 开始训练前，检测当前获取到的训练集和标签集数量

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)  # 获得训练集图片的数量

# Dataset
salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,  # img_name_list 对应上方的train_images_list
    lbl_name_list=tra_lbl_name_list,  # lbl_name_list 对应上方的train_labels_list
    transform=transforms.Compose([RescaleT(320), RandomCrop(288), ToTensorLab(flag=0)]))

# 参数分别为：数据集，单次训练数量，多进程参数，打乱顺序
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train,num_workers=0,shuffle=True)  
```



3-4、网络与优化器

```python
# ------- 3. define model / define the net --------
# 选择网络模型
if (model_name == 'u2net'):  # 完整U2net
    net = U2NET(3, 1)
elif (model_name == 'u2netp'):  # 轻量级U2net，速度快，体积小
    net = U2NETP(3, 1)

if torch.cuda.is_available():  # 检查是否有可用GPU设备
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
# Adam优化器
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
```



5、主进程

```python
# ------- 5. training process --------
ite_num = 0  # 每个epoch循环计数
running_loss = 0.0  # 初始化损失和
running_tar_loss = 0.0  # 初始化平均损失
ite_num4val = 0  # 单次迭代中循环计数
save_frq = 2000  # 每2000次迭代保存一次模型

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1  # epoch + 1
        ite_num4val = ite_num4val + 1  # 迭代次数
        inputs, labels = data['image'], data['label']
        inputs = inputs.type(torch.FloatTensor)  # 转换为torch.FloatTensor
        labels = labels.type(torch.FloatTensor)  # Float类型的张量(默认的torch.Tensor类型)

        # 将读取出来的数据包装到变量中
        if torch.cuda.is_available():
            inputs_v = Variable(inputs.cuda(), requires_grad=False)
            labels_v = Variable(labels.cuda(), requires_grad=False)
        else:
            inputs_v = Variable(inputs, requires_grad=False)
			labels_v = Variable(labels, requires_grad=False)
        # y zero the parameter gradients
        optimizer.zero_grad()  # *清空过往的梯度

        # forward + backward + optimize 获取图像和标签，通过计算得到预测值，得到损失函数
        # 得到通过U2net后经过了sigmoid激活函数的预测概率图
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)  
        # 返回'side0损失'与'side0-6损失相加的和'
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)  

        loss.backward()  # *反向传播，计算当前梯度
        optimizer.step()  # *根据当前梯度，更新网络的参数

        running_loss += loss.data.item()  # side0-6总损失
        running_tar_loss += loss2.data.item()  # side0的损失

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss  # 删除临时的局部变量

        # 打印每轮循环的结果
        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] cost: [%.2fs]/[%.2fm] train loss: %.3f, tar: %.3f " % (epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, (e_time - s_time), (e_time - sta_time) / 60.0, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if ite_num % save_frq == 0:  # 检查是否达到设定的迭代保存次数
            # 保存权重文件
            torch.save(net.state_dict(), model_dir + model_name + "_bce_itr_%d_train_%.3f_tar_%.3f.pth" % (
                ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0  # 重置参数
            running_tar_loss = 0.0  # 重置参数
            net.train()  # resume train 恢复训练
            ite_num4val = 0  # 重置单次迭代

```





## model/u2net.py

这里主要是有关U^2Net模型部分，这里仅举例RSU4和RSU4F模型。

1、通用卷积函数（CONV-BN-RELU）与上采样

```python
class REBNCONV(nn.Module):
    """
       1、Conv2d(in, out, 3, p=1, dilation=1) 这里默认卷积核3x3
       输入通道数=3，输出通道数=3，卷积核大小3x3，padding补0=1，默认dirate=1 代表普通卷积
       2、BN层(Batch Normalization)放置在Conv层和Relu层之间
       作用：使得一批特征矩阵分布满足均值=0，方差=1的分布规律
       3、Relu激活函数 Relu(x) = max(0,x)
       inplace=True 原地操作：计算得到的值不会覆盖之前的值，
       设置为True时，则会把计算得到的值直接覆盖到输入中，这样可以节省内存/显存。
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, dirate: int = 1):
        super(REBNCONV, self).__init__()  # 初始化调用此方法
		# 卷积,输入in_ch，输出out_ch，padding=1/dirate
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)  
        self.bn_s1 = nn.BatchNorm2d(out_ch)  # 根据输出通道数out_ch来创建Bn标准化处理层
        self.relu_s1 = nn.ReLU(inplace=True)  # Relu层，inplace代表是否直接覆盖输入的值

    def forward(self, x: torch.Tensor):
        # 定义正向传播过程，依次通过卷积-Bn-Relu
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))  
    
def _upsample_like(src, tar):  # src代表上一层传入的tensor,tar代表同一级encoder传入的tensor
    # src = F.upsample(src, size=tar.shape[2:], mode='bilinear')
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')  # 进行上采样，使用双线性插值法将src缩放到tar大小，取代原来的转置卷积
    return src
```

2、RSU4

```python
### RSU-4 ###
class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):  # 输入特征图通道数,中间层操作通道数,输出通道数
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)  # 将输入ch改为输出ch个数 和上采样结果拼接
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)  # out_ch改为mid_ch encoder阶段
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 下采样改变HW kernel=2x2 步距为2 缩放系数2
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)  # 其余阶段皆以mid_ch的通道数进行卷积BnRelu
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 一共下采样2次，缩放系数为4x
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)  # 膨胀卷积系数为2，padding=2

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)  # decoder阶段，拼接特征图
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)  # 所以特征图通道数*2
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)  # 上采样恢复到输出大小 和输入拼接

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)  # in_ch -> out_ch 输入图像通道个数 -> 输出的通道个数
        hx1 = self.rebnconv1(hxin)  # out_ch -> mid_ch 改为RSU内所使用的通道个数
        hx = self.pool1(hx1)  # 下采样1次 缩放系数=2x
        hx2 = self.rebnconv2(hx)  # mid_ch -> mid_ch
        hx = self.pool2(hx2)  # 下采样2次 缩放系数=4x
        hx3 = self.rebnconv3(hx)  # mid_ch -> mid_ch
        hx4 = self.rebnconv4(hx3)  # mid_ch -> mid_ch 膨胀卷积系数=2 padding=2
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))  # 拼接[En4,En3] 通过卷积BnRelu得到De3
        hx3dup = _upsample_like(hx3d, hx2)  # 通过双线性插值将De3的特征图高宽调整为En2的尺寸
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))  # 拼接[上采样后的De3,En2] 通过卷积BnRelu
        hx2dup = _upsample_like(hx2d, hx1)  # 调整为En1尺寸
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))  # 拼接[上采样后的De2,En1] 还原回输出尺寸
        return hx1d + hxin  # 返回输入的特征图和Decoder后结果的和
```

3、RSU4F

```python
### RSU-4F ###
class RSU4F(nn.Module):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)  # RSU4F中in_ch = out_ch 不再进行上下采样
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)  # 将通道个数改为out_ch -> mid_ch
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)  # 使用RSU内的mid_ch 膨胀系数为2 padding=2
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)  # 膨胀系数为4 padding=4
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)  # 膨胀系数为8 padding=8
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)  # 拼接[en4,en3] 通过卷积BnRelu
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)  # 拼接[de3,en2] 通过卷积BnRelu
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)  # 拼接[de2,en1] 还原回输出通道个数

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)  # in_ch -> out_ch
        hx1 = self.rebnconv1(hxin)  # out_ch -> mid_ch
        hx2 = self.rebnconv2(hx1)  # dilation=2 padding=2
        hx3 = self.rebnconv3(hx2)  # dilation=4 padding=4
        hx4 = self.rebnconv4(hx3)  # dilation=8 padding=8
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))  # cat[en4,en3] ConvBnRelu
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))  # cat[de3,en2] ConvBnRelu
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))  # cat[de2,en1] 还原输出
        return hx1d + hxin  # 返回Decoder和输入相加的结果
```

4、U^2Net完整模型

```python
##### U^2-Net ####
class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):  # 输入图像通道数=3，输出=1，完整的U2Net模型(150MB左右)
        super(U2NET, self).__init__()

        # encoder
        self.stage1 = RSU7(in_ch, 32, 64)  # channel=[3]->[64] mid_ch=[32]
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 下采样1次，H&W=[1]->[1/2]
        self.stage2 = RSU6(64, 32, 128)  # channel=[64]->[128] mid_ch=[32]
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 下采样2次，H&W=[1/2]->[1/4]
        self.stage3 = RSU5(128, 64, 256)  # channel=[128]->[256] mid_ch=[64]
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 下采样3次，H&W=[1/4]->[1/8]
        self.stage4 = RSU4(256, 128, 512)  # channel=[256]->[512] mid_ch=[128]
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 下采样4次，H&W=[1/8]->[1/16]
        self.stage5 = RSU4F(512, 256, 512)  # 膨胀Conv，channel=[512]->[512] mid_ch=[256]
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 下采样5次，H&W=[1/16]->[1/32]
        self.stage6 = RSU4F(512, 256, 512)  # 膨胀Conv，channel=[512]->[512] mid_ch=[256]

        # decoder
        # 上采样1次，拼接上一层En_6和同层En_5的结果，H&W=[1/32]->[1/16]
        self.stage5d = RSU4F(1024, 256, 512)  # 进行了拼接，输入通道数翻倍，channel=[512*2]->[512] mid_ch=[256]
        # 上采样2次，H&W=[1/16]->[1/8]，拼接De_5和En_4的结果
        self.stage4d = RSU4(1024, 128, 256)  # channel=[512*2]->[256] mid_ch=[128]
        # 上采样3次，H&W=[1/8]->[1/4]，拼接De_4和En_3的结果
        self.stage3d = RSU5(512, 64, 128)  # channel=[256*2]->[128] mid_ch=[64]
        # 上采样4次，H&W=[1/4]->[1/2]，拼接De_3和En_2的结果
        self.stage2d = RSU6(256, 32, 64)  # channel=[128*2]->[64] mid_ch=[32]
        # 上采样5次，H&W=[1/2]->[1]，拼接De_2和En_1的结果
        self.stage1d = RSU7(128, 16, 64)  # channel=[64*2]->[64] mid_ch=[16]

        # 将RSU_Decoder阶段每一层的结果通过卷积处理
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)  # hx1d，in=64,out=1,kernel=3x3
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)  # hx2d
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)  # hx3d
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)  # hx4d
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)  # hx5d
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)  # hx6d
        # output 将所有得到的结果再次通过卷积层
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x
        # stage 1
        hx1 = self.stage1(hx)  # RSU7_En1 channel=[3]->[64] mid_ch=[32]
        hx = self.pool12(hx1)  # 下采样1次，H&W=[1]->[1/2]
        # stage 2
        hx2 = self.stage2(hx)  # RSU6_En2 channel=[64]->[128] mid_ch=[32]
        hx = self.pool23(hx2)  # 下采样2次，H&W=[1/2]->[1/4]
        # stage 3
        hx3 = self.stage3(hx)  # channel=[128]->[256] mid_ch=[64]
        hx = self.pool34(hx3)  # 下采样3次，H&W=[1/4]->[1/8]
        # stage 4
        hx4 = self.stage4(hx)  # channel=[256]->[512] mid_ch=[128]
        hx = self.pool45(hx4)  # 下采样4次，H&W=[1/8]->[1/16]
        # stage 5
        hx5 = self.stage5(hx)  # 膨胀卷积，channel=[512]->[512] mid_ch=[256]
        hx = self.pool56(hx5)  # 下采样5次，H&W=[1/16]->[1/32]
        # stage 6
        hx6 = self.stage6(hx)  # 膨胀Conv，channel=[512]->[512] mid_ch=[256]
        hx6up = _upsample_like(hx6, hx5)  # 上采样1次，H&W=[1/32]->[1/16]

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))  # 拼接上一层En_6和同层En_5，channel=[512*2]->[512] mid_ch=[256]
        hx5dup = _upsample_like(hx5d, hx4)  # 上采样2次，H&W=[1/16]->[1/8]
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))  # 拼接De_5和En_4，channel=[512*2]->[256] mid_ch=[128]
        hx4dup = _upsample_like(hx4d, hx3)  # 上采样3次，H&W=[1/8]->[1/4]，拼接De_4和En_3的结果
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))  # 拼接De_4和En_3，channel=[256*2]->[128] mid_ch=[64]
        hx3dup = _upsample_like(hx3d, hx2)  # 上采样4次，H&W=[1/4]->[1/2]
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))  # 拼接De_3和En_2，channel=[128*2]->[64]
        hx2dup = _upsample_like(hx2d, hx1)  # 上采样5次，H&W=[1/2]->[1]
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))  # 拼接De_2和En_1，channel=[64*2]->[64] mid_ch=[16]

        # side output 用于存储每一层输出的预测图结果（sup1~6）
        d1 = self.side1(hx1d)  # RSU7_De1_out_ch=[64*2->64]=>[out_ch=1]，通过Conv(64,1,3,p=1)
        d2 = self.side2(hx2d)  # RSU6_De2_out_ch=[128*2->64]=>[1]
        d2 = _upsample_like(d2, d1)  # 上采样放大至RSU7_de输出的图像高宽
        d3 = self.side3(hx3d)  # RSU5_De3_out_ch=[512->128]=>[1]
        d3 = _upsample_like(d3, d1)  # 上采样
        d4 = self.side4(hx4d)  # RSU4_De4_out_ch=[1024->256]=>[1]
        d4 = _upsample_like(d4, d1)  # 上采样
        d5 = self.side5(hx5d)  # RSU4F_De5_out_ch=[1024->512]=>[1]
        d5 = _upsample_like(d5, d1)  # 上采样
        d6 = self.side6(hx6)  # RSU4F_En6_out_ch=[512->512]=>[1]
        d6 = _upsample_like(d6, d1)  # 上采样
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))  # 拼接后再次通过卷积层(in=6*1,out=1,ker=1)，和↓Sigmoid激活函数
        # 返回融合后的结果（概率图，每个像素E[0,1]），以及每一层的结果，损失计算
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
```



## u2net_test.py

```python
def normPRED(d):  # 传入一个tensor
    ma = torch.max(d)  # 获取最大值
    mi = torch.min(d)  # 获取最小值
    dn = (d - mi) / (ma - mi)  # 所有值减去最小值/最大减去最小值
    return dn

def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()  # squeeze(): 维度压缩，删除所有为1的维度
    predict_np = predict.cpu().data.numpy()  # .numpy()将Tensor转化为ndarray .cpu()将数据转移到cpu上
    im = Image.fromarray(predict_np * 255).convert('RGB')  # 将image转为array，提供给下面的imo
    img_name = image_name.split(os.sep)[-1]  # 赋值为image_name的路径(os.sep)最后一项[-1]
    image = io.imread(image_name)  # 读取图片文件
    # resize用于修改图片的尺寸  参数1：(高,宽)  参数2：改变图像尺寸的插值方法：双线性插值
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.Resampling.BILINEAR)
    pb_np = np.array(imo)  # 以image转成array再创建一个np的数组

    # result结果更改后缀jpg为png
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    # 保存输出文件，除了文件后缀更改以外跟原文件名一致
    imo.save(d_dir + imidx + '.png')


def main():
    # --------- 1. get image path and name ---------
    model_name = 'u2net'  # u2netp
    # 原·获取图像路径和名称
    # image_dir = os.path.join(os.getcwd(), 'test_data', 'test_photos')  # 要测试的图片路径
    # prediction_dir = os.path.join(os.getcwd(), 'test_data', 'photos_results_' + model_name + os.sep)  # 预测结果文件夹

    # 使用DUTS-TE测试
    image_dir = os.path.join(os.getcwd(), 'train_data', 'DUTS', 'DUTS-TE', 'DUTS-TE-Image')  # 要测试的图片路径
    prediction_dir = os.path.join(os.getcwd(), 'train_data', 'DUTS', 'DUTS-TE',
                                  'DUTS_results_' + model_name + os.sep)  # 预测结果文件夹

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')  # 加载权重文件文件夹
    img_name_list = glob.glob(image_dir + os.sep + '*')  # 图片文件夹路径下的所有文件* os.sep='\'
    print(img_name_list)

    # --------- 2. Dataset and Dataloader ---------
    # 自定义显著性目标test_salobj_dataset  sample = {'imidx': imidx, 'image': image, 'label': label}
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,  # 图片文件夹路径下的文件
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    # Dataloader
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,  # 按序处理 
                                        num_workers=0)  # origin=1

    # --------- 3. model define 加载模型 ---------
    if (model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)

    # 判断设备可用性
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))

    net.eval()  # 测试网络模型

    # --------- 4. inference for each image  推断每个图像(的显著目标) ---------
    start_time = datetime.datetime.now()
    print(f'inference start, now: {start_time:3f}')
    for i_test, data_test in enumerate(test_salobj_dataloader):
        s_time = time.time()  # start time

        inputs_test = data_test['image']  # 获取dataloader中的image
        inputs_test = inputs_test.type(torch.FloatTensor)  # 转换为Tensor类型

        # 检查可用设备
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)  # 经过U2net返回得side0~6

        # 返回值为d1~d7，取d1(side0)
        pred = d1[:, 0, :, :]  # pred取d1第一个三维矩阵
        pred = normPRED(pred)  # 将像素归一化到[0-1]

        # save results to test_results folder
        if not os.path.exists(prediction_dir):  # 创建保存用的文件夹
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, prediction_dir)  # 保存预测结果
        print(f"inferencing: {img_name_list[i_test].split(os.sep)[-1]},"
              f" cost: {time.time() - s_time}")  # 打印当前推断的图像名称
        del d1, d2, d3, d4, d5, d6, d7  # 释放空间

    # end
    print(f'end, cost: {(datetime.datetime.now() - start_time):3f}')


if __name__ == "__main__":
    main()
```



## data_loader.py

ToTensor类

```python
# U2net_test中的dataset调用此方法，所有图像都是使用ToTensor进行预处理，flag=0的处理过程
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))  # 创建一个全0数组
        tmpLbl = np.zeros(label.shape)

        image = image / np.max(image)  # **每个颜色像素值除以所有颜色像素的最大值
        if (np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)

        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229  # **在每个颜色像素上应用均值和标准差[?]
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'imidx': torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}
```

SalObjDataset类

```python

```







## 数据集

DUT-OMRON 和 DUTS 都是用于显着物体检测的大规模数据集。 但是，它们之间存在一些差异： 规模：DUT-OMRON 是一个更大的数据集，有超过 10,000 张图像，而 DUTS 有大约 5,000 张图像。 注释：DUT-OMRON 使用二进制掩码进行注释，而 DUTS 使用像素级注释。 多样性：DUTS包含的图像范围更广，包括场景复杂、遮挡、变形的图像，而DUT-OMRON更侧重于自然图像。 这两个数据集都广泛用于显着目标检测领域，用于训练和评估模型。 数据集的选择通常取决于具体的研究问题和所需的复杂程度。





ChatGPT

> U^2-Net is a deep learning-based model for salient object detection. If you want to study U^2-Net, you can follow these steps:  Familiarize yourself with deep learning: Before diving into U^2-Net, it's important to have a solid understanding of deep learning concepts, such as convolutional neural networks (CNNs), activation functions, and backpropagation.  Read the original paper: The U^2-Net paper, "U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection," provides a detailed description of the model architecture, loss function, and evaluation metrics.  Review the code: The U^2-Net code is available on GitHub, and reviewing the code can give you a better understanding of how the model is implemented.  Practice with datasets: You can practice with datasets, such as DUT-OMRON or DUTS, to see how the U^2-Net model performs on different types of images.  Explore extensions and variations: U^2-Net has been extended and modified in several ways, such as U^2-Net   and U^2-Net with attention. You can explore these extensions and variations to see how they improve upon the original U^2-Net model.  Studying U^2-Net requires a solid understanding of deep learning and computer vision, as well as a willingness to dive into the details of the model architecture and implementation. However, by following these steps, you can gain a deeper understanding of U^2-Net and its applications in salient object detection.
>
>
>
> U^2-Net 是一种基于深度学习的显着目标检测模型。 如果你想研究 U^2-Net，你可以遵循以下步骤： 熟悉深度学习：在深入研究 U^2-Net 之前，对深度学习概念有扎实的理解很重要，例如卷积神经网络（ CNN）、激活函数和反向传播。 阅读原文：U^2-Net 论文“U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection”详细描述了模型架构、损失函数和评估指标。 查看代码：GitHub 上提供了 U^2-Net 代码，查看代码可以让您更好地了解模型的实现方式。 使用数据集进行练习：您可以使用 DUT-OMRON 或 DUTS 等数据集进行练习，以了解 U^2-Net 模型在不同类型图像上的表现。 Explore extensions and variations：U^2-Net在几个方面进行了扩展和修改，例如U^2-Net和U^2-Net with attention。 您可以探索这些扩展和变体，了解它们如何改进原始 U^2-Net 模型。 研究 U^2-Net 需要对深度学习和计算机视觉有扎实的理解，并愿意深入研究模型架构和实现的细节。 但是，通过执行这些步骤，您可以更深入地了解 U^2-Net 及其在显着目标检测中的应用。
>
>
>
> U^2-Net代码中的 "data_loader.py "文件负责加载和预处理用于训练和测试的数据。数据加载器的主要功能包括。
>
> 数据加载。数据加载器从数据集中加载图像和注释，数据集通常存储在一个目录结构中。图像通常以JPEG或PNG格式存储，而注释则以二进制掩码或像素化注释的形式存储。
>
> 数据预处理。数据加载器对图像和注释进行预处理，使其适合训练和测试。这可能包括将图像的大小调整到一个固定的尺寸，将像素值标准化，并将注释转换为二进制掩码或热图。
>
> 数据增强。数据加载器还可以应用数据增强技术，如随机翻转、旋转和缩放，以增加训练数据的多样性并提高模型的泛化。
>
> data_loader.py "文件是U^2-Net代码的一个重要组成部分，因为它为训练和测试模型提供了输入数据。通过了解数据加载器的功能和实现，你可以对U^2-Net模型的训练和评估有更深的了解。





# 综述

近年来，显著性目标检测领域取得了很多重要的进展，主要集中在以下几个方面：

1. 多目标检测：随着视频和图像内容的不断复杂，显著性目标检测需要同时识别多个目标，因此最近的研究重点是提高多目标检测的效率和精度。
2. 小目标检测：随着摄像机像素的不断提高，越来越多的小目标被检测到，因此对于小目标的识别和检测成为了当前的研究热点。
3. 超高分辨率图像检测：随着图像分辨率的不断提高，检测高分辨率图像中的目标变得越来越困难，因此对于超高分辨率图像的检测也是当前研究的重点。
4. 计算效率优化：随着模型的不断复杂，计算效率的提高成为了当前的研究热点。
5. 全景图像检测：随着无人机技术的不断发展，对于全景图像的检测成为了当前研究的重点。
6. 跨模态检测：随着多模态数据的广泛使用，跨模态的显著性目标检测成为了当前研究的热点。例如，跨视频和图像的目标检测，或者跨多种传感器的目标检测。
7. 端到端学习：与传统的显著性目标检测方法相比，端到端学习方法可以直接从原始图像中学习显著性特征，因此是当前研究的热点。
8. 数据增强：由于大量数据对于显著性目标检测是非常重要的，因此数据增强技术是当前研究的重点。
9. 增强现实：随着增强现实技术的不断发展，显著性目标检测在增强现实领域的应用也成为了当前研究的重点。



这些方向都在不断发展，并对显著性目标检测技术产生了深远的影响。



**如何在显著性目标检测的网络模型中添加注意力层**

在显著性目标检测的网络模型中添加注意力机制的一种方法是通过构建注意力模型。这个模型用于计算每个输入图像位置的注意力权重，以确定该位置的显著性特征。

这可以通过以下步骤实现：

1. 预处理输入图像，通过一系列卷积和池化层将图像特征提取出来。
2. 在预处理后的特征图上构建一个注意力模型，该模型可以通过全连接层，卷积层，递归层等实现。注意力模型用于计算每个特征图位置的注意力权重。
3. 通过对预处理后的特征图与注意力权重进行乘法，得到最终的特征图，该图捕获了图像的显著性特征。
4. 对最终的特征图进行分类，确定图像中的显著性目标。

在显著性目标检测中添加注意力机制的目的是通过学习输入图像的重要性，提高模型的准确性和效率。然而，由于注意力模型的构建和训练过程复杂，它们需要大量的训练数据和计算资源。

注意力机制可以通过在模型的某些层中引入注意力机制来实现。一种常见的方法是在卷积层后面加入注意力层，以便模型可以关注特定的特征。在每一个注意力层中，可以计算出每个特征的重要程度，并将其乘以对应的特征图。这样，最终生成的特征图将会更加突出显著的目标。在多层注意力网络中，每一层都可以计算出一组权重，从而使模型可以逐渐关注不同的特征。

在实际实现中，注意力机制可以通过以下几种方法来实现：

- Soft Attention: 使用一个全连接层和一个 Softmax 函数计算特征的权重，并将权重乘以特征图。
- Hard Attention: 使用一个硬件层（如 Gumbel-Softmax）来生成固定的注意力权重，从而关注特定的特征。
- Multi-Head Attention: 在模型中同时使用多个注意力头，以便从不同的角度关注不同的特征。

在使用注意力机制时，还需要注意数据标准化和训练方法等问题，以便更好地获得更好的注意力效果。



**注意力模型的构建和训练过程复杂，它们需要大量的训练数据和计算资源。如何解决？**

通常来说，可以通过以下几种方法来解决注意力模型训练所需的数据和计算资源问题：

1. 数据增强：使用数据增强技术，例如图像旋转、翻转和裁剪等，以扩展现有的数据集，从而减少对更多数据的依赖。
2. 小型模型：使用较小的模型结构，并选择具有较低复杂度的注意力机制，以减少计算资源的需求。
3. 轻量级注意力机制：使用轻量级注意力机制，例如SE-Net和CBAM等，以减少计算资源的需求。
4. 分布式训练：使用分布式训练技术，例如分布式数据并行或分布式模型并行，以利用多个计算节点的计算能力来加速训练过程。
5. 预训练：使用预训练技术，即从已有的预训练模型中提取特征，再使用这些特征来训练注意力模型，从而避免从头开始训练的复杂过程。



**基于U^2Net，举个例子引入注意力机制**

U^2-Net是一种在显著性目标检测领域中广受欢迎的模型。要在U^2-Net中引入注意力机制，需要在网络模型中添加注意力层。

注意力层可以使用多种不同的方法实现，其中一种是使用Self-Attention机制。 Self-Attention机制在图像处理中常常用来实现动态空间重点关注，并且它适用于卷积神经网络。在U^2-Net中，Self-Attention机制可以通过计算输入特征图上每个像素与其他像素的相似性，来构建注意力层。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, size=1):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_atten = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        attention = self.conv_atten(x)
        attention = self.sigmoid(attention)

        x = x * attention

        return x

class U2Net(nn.Module):
    def __init__(self, num_classes=1):
        super(U2Net, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn

```










































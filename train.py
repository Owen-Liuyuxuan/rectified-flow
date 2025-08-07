import torch
import os
import yaml
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
from model import MiniUnet
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from rectified_flow import RectifiedFlow

"""
We assume a uniform linear path and express 

x_t = (1 - t) * x_1 + t * x_0

then the velocity is 

v = dx_t/dt = x_0 - x_1 --- Which becomes the network expected prediction to be trained. During revert, we simulate using -v = x_1 - x_0

-------

Generally speaking

x_t = a(t) * x_1 + b_t * x_0. With marginal requirement of a_0 = 1, b_0 = 0 & a_1 = 0, b_1 = 1.

v = dx_t/dt = a'(t) * x_1 + b'(t) * x_0 = u

-------

Then in conditional flow matching, we define 

loss = ||  v_\theta(z, t) - u_t(x_t|\epsilon) ||2

In order to convert u_t without boundaries:

1. x_1_expected = (x_t - b(t) * x_0) / a(t)
2. v = a'(t) * x_1_expected + b'(t) * x_0 = a'(t) / a(t) x_t - \
        x_0 * b(t)( a'(t)/a(t) - b'(t) / b(t))
3. signal_to_noise ratio  \lambda(t) = 2 ln (a(t)/b(t))  \lambda'(t) = 2 a'(t)/ a(t) - 2 b'(t) / b(t)
4. v_expect = a'(t)/a(t) * x_t  - b(t)/2 \lambda'(t) * x_0
5. loss = || v\theta(z,t) - v_expect  || = || v\theta(z,t) - a'(t)/a(t) * x_t  + b(t) / 2 \lambda'(t) * x_0  ||2
        = (-bt/2 \lambda'(t))^2 || \epsilon\theta(z, t) - x_0 ||

    \epsilon\theta(x(t), t) = (-2) / (\lambda'(t)b(t)) (v - a'(t)/a(t) * x_t) 
            --- This can be the network taking noisy input (x_t) and time embedding to learned

    In inference, we can restore v = \epsilon\theta(x(t), t) *(\lambda'(t) b(t)) / (-2) + a'(t)/a(t) * x_t

------ 
Back to example of conditional flow matching.

1. we can train a network, taking noisy input x(t), time embedding t to learn the correct x_0
2. During inference:
    v_base = model(x_t, t)
    v = v_base  * \lambda'(t) * b(t) / (-2) + a'(t) / a(t) * x_t
       = v_base * 1 /  (1 - t) - 1 / (1 - t) * x_t
      = (v_base - x_t) / (1 - t)

3. 1 / t is only the size of the direction, which is rather sensitive to the time step, it is rather common to regularize it.
   to v = (x_t - v_base)
    We point out that in training v_base converge to the correct x_0, so v = x_t - v_base ~ x_t - x_0, 
    This CFM aligned with the motion of the base rectified flow.

-------

There are many motion functions

Rectified flow: 
x_t = t * x_1 + (1 - t) * x_0

Cosine:
x_t = cos(t * pi / 2) * x_1 + sin(t * pi / 2) * x_0

LDM-Linear for modified DDPM:
It means the gaussian noise is increasing linearly from x_0 to x_1, while preserving  b(t)^2 + a(t)^2 = 1

a(t) = \sqrt( (1-\beta_0) * (1 - \beta_1) * (1 - \beta_2) * .... * (1 - \beta_t))
b(t) = \sqrt(1 - a(t)^2)

DDPM: \beta_t = \beta_0 + t/(T-1) (\beta_{T-1} - \beta_0)
LDM:  \beta_t = (\sqrt(\beta_0) + t/(T-1) * \sqrt(\beta_{T-1}))

This link back to the diffusion process on the forward processing with diffusion schedules.

-------
a**2 = (1 - beta_0) * (1 - beta_1) * (1 - beta_2) * .... * (1 - beta_t)
2a * a' = (1 - beta_0) * (1 - beta_1) * (1 - beta_2) * .... * (1 - beta_t_1) * ( - beta_t)

a'(t) = (1 - beta_0) * (1 - beta_1) * (1 - beta_2) * .... * (1 - beta_t_1) * (- beta_t) / 2 * a(t)
b(t) ** 2 = 1 - a ** 2
2 * b(t) b'(t) = 1 - 2 * a(t) * a'(t) = 1 - (1 - beta_0) * (1 - beta_1) * (1 - beta_2) * .... * (1 - beta_t_1) * ( - beta_t)
b'(t) = [1 - 2 * a(t) * a'(t) = 1 - (1 - beta_0) * (1 - beta_1) * (1 - beta_2) * .... * (1 - beta_t_1) * ( - beta_t) ]  / (2 * b(t))


"""


def train(config: str):
    """训练flow matching模型

    Args:
        config (str): yaml配置文件路径，包含以下参数：
            base_channels (int, optional): MiniUnet的基础通道数，默认值为16。
            epochs (int, optional): 训练轮数，默认值为10。
            batch_size (int, optional): 批大小，默认值为128。
            lr_adjust_epoch (int, optional): 学习率调整轮数，默认值为50。
            batch_print_interval (int, optional): batch打印信息间隔，默认值为100。
            checkpoint_save_interval (int, optional): checkpopint保存间隔(单位为epoch)，默认值为1。
            save_path (str, optional): 模型保存路径，默认值为'./checkpoints'。
            use_cfg (bool, optional): 是否使用Classifier-free Guidance训练条件生成模型，默认值为False。
            device (str, optional): 训练设备，默认值为'cuda'。

    """
    # 读取yaml配置文件
    config = yaml.load(open(config, 'rb'), Loader=yaml.FullLoader)
    # 解析参数数据，有默认值
    base_channels = config.get('base_channels', 16)
    epochs = config.get('epochs', 10)
    batch_size = config.get('batch_size', 128)
    lr_adjust_epoch = config.get('lr_adjust_epoch', 50)
    batch_print_interval = config.get('batch_print_interval', 100)
    checkpoint_save_interval = config.get('checkpoint_save_interval', 1)
    save_path = config.get('save_path', './checkpoints')
    use_cfg = config.get('use_cfg', False)
    device = config.get('device', 'cuda')
    cfm = config.get("cfm", False)

    # 打印训练参数
    print('Training config:')
    print(f'base_channels: {base_channels}')
    print(f'epochs: {epochs}')
    print(f'batch_size: {batch_size}')
    print(f'lr_adjust_epoch: {lr_adjust_epoch}')
    print(f'batch_print_interval: {batch_print_interval}')
    print(f'checkpoint_save_interval: {checkpoint_save_interval}')
    print(f'save_path: {save_path}')
    print(f'use_cfg: {use_cfg}')
    print(f'device: {device}')

    # 训练flow matching模型

    # 数据集加载
    # 把PIL转为tensor
    transform = Compose([ToTensor()])  # 变换成tensor + 变为[0, 1]

    dataset = MNIST(
        root='./data',
        train=True,  # 6w
        download=True,
        transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型加载
    model = MiniUnet(base_channels)
    model.to(device)

    # 优化器加载 Rectified Flow的论文里面有的用的就是AdamW
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    # 学习率调整
    scheduler = StepLR(optimizer, step_size=lr_adjust_epoch, gamma=0.1)

    # RF加载
    rf = RectifiedFlow()

    # 记录训练时候每一轮的loss
    loss_list = []

    # 一些文件夹提前创建
    os.makedirs(save_path, exist_ok=True)

    # 训练循环
    for epoch in range(epochs):
        for batch, data in enumerate(dataloader):
            x_1, y = data  # x_1原始图像，y是标签，用于CFG
            # 均匀采样[0, 1]的时间t randn 标准正态分布
            t = torch.rand(x_1.size(0))

            # 生成flow（实际上是一个点） # adding gaussian noise to x_1 to x_t, where x_0 is pure gaussian noise
            x_t, x_0 = rf.create_flow(x_1, t)

            # 4090 大概占用显存3G
            x_t = x_t.to(device)
            x_0 = x_0.to(device)
            x_1 = x_1.to(device)
            t = t.to(device)

            optimizer.zero_grad()

            # 这里我们要做一个数据的复制和拼接，复制原始x_1，把一半的y替换成-1表示无条件生成，这里也可以直接有条件、无条件累计两次计算两次loss的梯度
            # 一定的概率，把有条件生成换为无条件的 50%的概率 [x_t, x_t] [t, t] ## attached on batch dimension
            if use_cfg:
                x_t = torch.cat([x_t, x_t.clone()], dim=0)
                t = torch.cat([t, t.clone()], dim=0)
                y = torch.cat([y, -torch.ones_like(y)], dim=0)
                x_1 = torch.cat([x_1, x_1.clone()], dim=0)
                x_0 = torch.cat([x_0, x_0.clone()], dim=0)
                y = y.to(device)
            else:
                y = None


            v_pred = model(x=x_t, t=t, y=y)

            loss = rf.mse_loss(v_pred, x_1, x_0, cfm=cfm)

            loss.backward()
            optimizer.step()

            if batch % batch_print_interval == 0:
                print(f'[Epoch {epoch}] [batch {batch}] loss: {loss.item()}')

            loss_list.append(loss.item())

        scheduler.step()

        if epoch % checkpoint_save_interval == 0 or epoch == epochs - 1 or epoch == 0:
            # 第一轮也保存一下，快速测试用，大家可以删除
            # 保存模型
            print(f'Saving model {epoch} to {save_path}...')
            save_dict = dict(model=model.state_dict(),
                             optimizer=optimizer.state_dict(),
                             epoch=epoch,
                             loss_list=loss_list)
            torch.save(save_dict,
                       os.path.join(save_path, f'miniunet_{epoch}.pth'))


if __name__ == '__main__':
    train(config='./config/train_config.yaml')

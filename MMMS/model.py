import torch
import torch.nn as nn


# 图像GAN生成器（生成1000维图像特征）
class ImageGenerator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1000),
            nn.Tanh()  # 输出归一化到[-1,1]范围
        )

    def forward(self, z):
        return self.net(z)


# 文本GAN生成器（生成768维文本特征）
class TextGenerator(nn.Module):
    def __init__(self, noise_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 768),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


# 通用判别器（MLP结构）
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

#
# # 初始化模型
# img_gen = ImageGenerator()
# txt_gen = TextGenerator()
# img_disc = Discriminator(1000)  # 图像判别器
# txt_disc = Discriminator(768)  # 文本判别器
#
# # 定义优化器与损失函数
# g_optimizer = torch.optim.Adam(
#     list(img_gen.parameters()) + list(txt_gen.parameters()),
#     lr=0.0002, betas=(0.5, 0.999)
# )
# d_optimizer = torch.optim.Adam(
#     list(img_disc.parameters()) + list(txt_disc.parameters()),
#     lr=0.0002, betas=(0.5, 0.999)
# )
# criterion = nn.BCELoss()
#
# # 训练参数
# noise_dim = 100
# batch_size = 64
# epochs = 100
#
# # 数据加载（假设已加载numpy格式特征）
# # real_img_feat: [N, 1000], real_txt_feat: [N, 768]
# # labels: 真实样本标签为1，生成样本标签为0
#
# for epoch in range(epochs):
#     # 生成随机噪声
#     z = torch.randn(batch_size, noise_dim)
#
#     # ======================
#     # 1. 训练生成器
#     # ======================
#     g_optimizer.zero_grad()
#
#     # 生成假特征
#     fake_img = img_gen(z)
#     fake_txt = txt_gen(z)
#
#     # 计算生成器的对抗损失
#     img_validity = img_disc(fake_img)
#     txt_validity = txt_disc(fake_txt)
#     g_loss = (criterion(img_validity, torch.ones_like(img_validity)) +
#               criterion(txt_validity, torch.ones_like(txt_validity))) / 2
#
#     g_loss.backward()
#     g_optimizer.step()
#
#     # ======================
#     # 2. 训练判别器
#     # ======================
#     d_optimizer.zero_grad()
#
#     # 真实数据损失
#     real_img_pred = img_disc(real_img_feat)
#     real_txt_pred = txt_disc(real_txt_feat)
#     real_loss = (criterion(real_img_pred, torch.ones_like(real_img_pred)) +
#                  criterion(real_txt_pred, torch.ones_like(real_txt_pred))) / 2
#
#     # 假数据损失
#     fake_img_pred = img_disc(fake_img.detach())
#     fake_txt_pred = txt_disc(fake_txt.detach())
#     fake_loss = (criterion(fake_img_pred, torch.zeros_like(fake_img_pred)) +
#                  criterion(fake_txt_pred, torch.zeros_like(fake_txt_pred))) / 2
#
#     d_loss = (real_loss + fake_loss) / 2
#     d_loss.backward()
#     d_optimizer.step()
#
#     # 打印训练状态
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
#
# # 保存专家判别器
# torch.save(img_disc.state_dict(), "img_gan_expert.pth")
# torch.save(txt_disc.state_dict(), "txt_gan_expert.pth")
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from model import TextGenerator, ImageGenerator, Discriminator
import torch.nn.functional as F


"""
   此代码为预训练GAN网络代码，作为MMoE的其中一个专家网络
   训练完毕后会冻结其参数，从而保留检测AI伪造的相关知识
"""

# 构建多模态GAN的数据集类
class GANDataset(Dataset):
    def __init__(self, img_feat_path, text_feat_path):
        """
        参数说明：
        img_feat_path: 图像特征numpy文件路径
        text_feat_path: 文本特征numpy文件路径
        """
        # 图像特征与文本特征都由numpy形式保存，这里是加载数据集
        self.raw_img_feats = np.load(img_feat_path)
        self.raw_text_feats = np.load(text_feat_path)

        # 验证数据一致性
        assert len(self.raw_img_feats) == len(self.raw_text_feats), "图像与文本特征数量不匹配"

        # 特征标准化
        self.img_scaler = StandardScaler()
        self.text_scaler = StandardScaler()
        self.img_feats = self.img_scaler.fit_transform(self.raw_img_feats)
        self.text_feats = self.text_scaler.fit_transform(self.raw_text_feats)

    def __len__(self):
        return len(self.img_feats)

    def __getitem__(self, idx):
        # 将numpy形式转换为张量
        return {
            'img_feat': torch.tensor(self.img_feats[idx], dtype=torch.float32),
            'text_feat': torch.tensor(self.text_feats[idx], dtype=torch.float32)
        }


# 对数据集和数据集的加载器进行初始化，
dataset = GANDataset(
    img_feat_path="pic_real.npy",
    text_feat_path="text_real.npy"
)

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False
)

# 设备配置
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 改进的GAN训练循环
def train_gan():
    # 初始化模型
    img_gen = ImageGenerator()
    txt_gen = TextGenerator()
    img_disc = Discriminator(1000)
    txt_disc = Discriminator(768)

    # 优化器
    g_optim = torch.optim.Adam(
        list(img_gen.parameters()) + list(txt_gen.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    d_optim = torch.optim.Adam(
        list(img_disc.parameters()) + list(txt_disc.parameters()),
        lr=0.0001, betas=(0.5, 0.999)
    )

    # 训练参数
    noise_dim = 100
    epochs = 5

    for epoch in range(epochs):
        for batch in dataloader:
            current_batch_size = batch['img_feat'].size(0)

            real_img = batch['img_feat']
            real_txt = batch['text_feat']

            # 生成随机噪声
            z = torch.randn(current_batch_size, noise_dim)

            # 训练判别器
            d_optim.zero_grad()

            # 使用生成器生成假特征
            with torch.no_grad():
                fake_img = img_gen(z)
                fake_txt = txt_gen(z)

            # 图像判别器损失
            real_img_loss = F.binary_cross_entropy(
                img_disc(real_img),
                torch.ones(current_batch_size, 1)
            )
            fake_img_loss = F.binary_cross_entropy(
                img_disc(fake_img.detach()),
                torch.zeros(current_batch_size, 1)
            )
            img_d_loss = (real_img_loss + fake_img_loss) / 2

            # 文本判别器损失
            real_txt_loss = F.binary_cross_entropy(
                txt_disc(real_txt),
                torch.ones(current_batch_size, 1)
            )
            fake_txt_loss = F.binary_cross_entropy(
                txt_disc(fake_txt.detach()),
                torch.zeros(current_batch_size, 1)
            )
            txt_d_loss = (real_txt_loss + fake_txt_loss) / 2

            # 总判别器损失
            d_total_loss = (img_d_loss + txt_d_loss) / 2
            d_total_loss.backward()
            d_optim.step()

            # 2. 训练生成器
            g_optim.zero_grad()

            # 重新生成特征（重要！需要重新前向传播）
            fake_img = img_gen(z)
            fake_txt = txt_gen(z)

            # 对抗损失
            img_g_loss = F.binary_cross_entropy(
                img_disc(fake_img),
                torch.ones(current_batch_size, 1)
            )
            txt_g_loss = F.binary_cross_entropy(
                txt_disc(fake_txt),
                torch.ones(current_batch_size, 1)
            )
            g_total_loss = (img_g_loss + txt_g_loss) / 2
            g_total_loss.backward()
            g_optim.step()

        # 每个epoch打印统计信息
        with torch.no_grad():
            # 计算判别器准确率
            real_img_acc = (img_disc(real_img) > 0.5).float().mean()
            fake_img_acc = (img_disc(fake_img) < 0.5).float().mean()
            real_txt_acc = (txt_disc(real_txt) > 0.5).float().mean()
            fake_txt_acc = (txt_disc(fake_txt) < 0.5).float().mean()

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"D Loss: {d_total_loss.item():.4f} | G Loss: {g_total_loss.item():.4f}")
            print(f"Image Disc Acc: Real {real_img_acc.item():.2f}, Fake {fake_img_acc.item():.2f}")
            print(f"Text Disc Acc: Real {real_txt_acc.item():.2f}, Fake {fake_txt_acc.item():.2f}")
            print("--------------------------")

    # 保存判别器作为专家网络
    # torch.save(img_disc.state_dict(), "pretrained_img_disc.pth")
    # torch.save(txt_disc.state_dict(), "pretrained_text_disc.pth")


if __name__ == "__main__":
    train_gan()
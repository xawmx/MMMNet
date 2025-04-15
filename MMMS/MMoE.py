import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


# 基础模块定义，分别有预训练好的GAN网络，图文不一致检测专家，情绪风险检测专家
class PretrainedDiscriminator(nn.Module):
    """ 预训练GAN判别器（冻结参数）"""

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
        return self.net(x).squeeze(-1)  # 输出维度[B]


class InconsistencyExpert(nn.Module):
    """ 图文不一致检测专家 """

    def __init__(self, img_dim=1000, text_dim=768, proj_dim=256):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)

    def forward(self, img_feat, text_feat):
        img_emb = F.normalize(self.img_proj(img_feat), dim=-1)
        text_emb = F.normalize(self.text_proj(text_feat), dim=-1)
        return F.cosine_similarity(img_emb, text_emb, dim=-1)  # 输出维度[B]


class EmotionExpert(nn.Module):
    """ 情绪风险检测专家 """

    def __init__(self, combined_dim=1768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # 输出维度[B]


class TaskGate(nn.Module):
    """ 任务特定门控网络 """

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # 4个专家
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


# 完整系统的定义
class MultiModalDetector(nn.Module):
    def __init__(self, gan_paths):
        super().__init__()
        # 加载并冻结GAN判别器
        self.img_disc = PretrainedDiscriminator(1000)
        self.txt_disc = PretrainedDiscriminator(768)
        self.img_disc.load_state_dict(torch.load(gan_paths['img']))
        self.txt_disc.load_state_dict(torch.load(gan_paths['text']))
        for model in [self.img_disc, self.txt_disc]:
            for param in model.parameters():
                param.requires_grad = False

        # 初始化组件
        self.incon_expert = InconsistencyExpert()
        self.emotion_expert = EmotionExpert()

        # 门控网络
        self.gates = nn.ModuleDict({
            'fake_img': TaskGate(1000),
            'fake_text': TaskGate(768),
            'inconsistency': TaskGate(1768),
            'emotion': TaskGate(1768)
        })

    def forward(self, img, text):
        # 特征拼接
        combined = torch.cat([img, text], dim=1)

        # 各专家输出
        experts = {
            'img_disc': self.img_disc(img),
            'txt_disc': self.txt_disc(text),
            'incon': self.incon_expert(img, text),
            'emotion': self.emotion_expert(combined)
        }

        # 门控加权
        outputs = {}
        for task in self.gates.keys():
            # 选择门控输入
            gate_input = img if task == 'fake_img' else \
                text if task == 'fake_text' else combined

            # 计算门控权重
            weights = self.gates[task](gate_input)

            # 加权融合（维度对齐关键步骤）
            weighted = sum(
                weights[:, i].unsqueeze(1) * experts[name].unsqueeze(1)
                for i, name in enumerate(experts)
            ).squeeze(1)

            outputs[task] = weighted

        return outputs


# ----------------- 数据集类 -----------------
class MultiModalDataset(Dataset):
    def __init__(self, img_path, text_path, label_paths):
        # 加载特征
        self.img_feats = np.load(img_path).astype(np.float32)
        self.text_feats = np.load(text_path).astype(np.float32)


        # 加载标签（确保一维）
        self.labels = {
            task: np.load(path).reshape(-1).astype(np.float32)
            for task, path in label_paths.items()
        }

    def __len__(self):
        return len(self.img_feats)

    def __getitem__(self, idx):
        return {
            'img': torch.tensor(self.img_feats[idx], dtype=torch.float32),
            'text': torch.tensor(self.text_feats[idx], dtype=torch.float32),
            'labels': {
                task: torch.tensor(value[idx], dtype=torch.float32)
                for task, value in self.labels.items()
            }
        }


# ----------------- 训练流程 -----------------
class MultiTaskTrainer:
    def __init__(self, config):
        self.device = torch.device("cpu")

        # 初始化模型
        self.model = MultiModalDetector(config['gan_paths']).to(self.device)

        # 数据加载
        dataset = MultiModalDataset(
            img_path=config['data']['img'],
            text_path=config['data']['text'],
            label_paths=config['data']['labels']
        )
        self.loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # 优化器设置
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config['lr'],
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=3
        )

        # 损失函数
        self.criterion = nn.BCELoss()
        self.ranking_loss = nn.MarginRankingLoss(margin=0.3)

    # def train_epoch(self):
    #     self.model.train()
    #     total_loss = 0.0
    #
    #     for batch in self.loader:
    #         img = batch['img'].to(self.device, non_blocking=True)
    #         text = batch['text'].to(self.device, non_blocking=True)
    #         labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
    #
    #         # 前向传播
    #         self.optimizer.zero_grad()
    #         outputs = self.model(img, text)
    #
    #         # 计算损失
    #         loss_dict = {}
    #         # GAN相关任务
    #         loss_dict['fake_img'] = self.criterion(outputs['fake_img'], labels['fake_img'])
    #         loss_dict['fake_text'] = self.criterion(outputs['fake_text'], labels['fake_text'])
    #         loss_dict['inconsistency'] = self.ranking_loss(
    #             outputs['inconsistency'],
    #             torch.ones_like(labels['inconsistency']),
    #             labels['inconsistency']
    #         )
    #         # 情绪检测
    #         loss_dict['emotion'] = self.criterion(outputs['emotion'], labels['emotion'])
    #
    #         # 总损失
    #         total_loss = sum(loss_dict.values())
    #
    #         # 反向传播
    #         total_loss.backward()
    #         nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
    #         self.optimizer.step()
    #
    #         total_loss += total_loss.item()
    #
    #     return total_loss / len(self.loader)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in self.loader:
            img = batch['img'].to(self.device, non_blocking=True)
            text = batch['text'].to(self.device, non_blocking=True)
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(img, text)

            # **防止输出超出范围**
            outputs['fake_img'] = torch.clamp(outputs['fake_img'], 1e-7, 1 - 1e-7)
            outputs['fake_text'] = torch.clamp(outputs['fake_text'], 1e-7, 1 - 1e-7)
            outputs['inconsistency'] = torch.clamp(outputs['inconsistency'], 1e-7, 1 - 1e-7)
            outputs['emotion'] = torch.clamp(outputs['emotion'], 1e-7, 1 - 1e-7)

            # 计算损失
            loss_dict = {}
            loss_dict['fake_img'] = self.criterion(outputs['fake_img'], labels['fake_img'])
            loss_dict['fake_text'] = self.criterion(outputs['fake_text'], labels['fake_text'])
            loss_dict['inconsistency'] = self.ranking_loss(
                outputs['inconsistency'],
                torch.ones_like(labels['inconsistency']),
                labels['inconsistency']
            )
            loss_dict['emotion'] = self.criterion(outputs['emotion'], labels['emotion'])

            # 计算总损失
            total_loss = sum(loss_dict.values())

            # 反向传播
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss += total_loss.item()

        return total_loss / len(self.loader)

    def run(self, epochs):
        best_loss = float('inf')
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            self.scheduler.step(train_loss)

            # 保存最佳模型
            # if train_loss < best_loss:
            #     best_loss = train_loss
            #     torch.save(self.model.state_dict(), "best_model.pth")

            print(
                f"Epoch {epoch + 1}/{epochs} | Loss: {train_loss*2000/(epoch + 1):.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}")


# 配置文件
config = {
    #来自预训练好的GAN网络参数
    'gan_paths': {
        'img': 'pretrained_img_disc.pth',
        'text': 'pretrained_text_disc.pth'
    },
    #数据集与标签集 加载相应的数据集与标签集
    'data': {
        'img': 'pic.npy',
        'text': 'text.npy',
        'labels': {
            'fake_img': 'pic_label.npy',
            'fake_text': 'text_label.npy',
            'inconsistency': 'cross_label.npy',
            'emotion': 'emo_label.npy'
        }
    },
    #训练设置
    'batch_size': 25,
    'lr': 3e-4,
    'epochs': 10
}

# 执行训练
if __name__ == "__main__":
    trainer = MultiTaskTrainer(config)
    trainer.run(config['epochs'])

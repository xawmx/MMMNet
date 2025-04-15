import torch
import numpy as np
import torch.nn.functional as F


# 定义与训练时相同的模型结构
class RiskDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义与训练完全一致的子模块
        self.img_disc = torch.nn.Sequential(
            torch.nn.Linear(1000, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

        self.txt_disc = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

        self.incon_expert = torch.nn.ModuleDict({
            'img_proj': torch.nn.Linear(1000, 256),
            'text_proj': torch.nn.Linear(768, 256)
        })

        self.emotion_expert = torch.nn.Sequential(
            torch.nn.Linear(1768, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )

        # 门控网络
        self.gates = torch.nn.ModuleDict({
            task: torch.nn.Sequential(
                torch.nn.Linear(1768 if "incon" in task else dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 4),
                torch.nn.Softmax(dim=-1)
            ) for task, dim in [('fake_img', 1000), ('fake_text', 768),
                                ('inconsistency', 1768), ('emotion', 1768)]
        })

    def forward(self, combined_feat):
        # 分割特征
        img_feat = combined_feat[:, :1000]
        text_feat = combined_feat[:, 1000:]

        # 各检测项计算
        outputs = {
            'fake_img': self.img_disc(img_feat).squeeze(),
            'fake_text': self.txt_disc(text_feat).squeeze(),
            'incon': torch.cosine_similarity(
                F.normalize(self.incon_expert['img_proj'](img_feat), dim=-1),
                F.normalize(self.incon_expert['text_proj'](text_feat), dim=-1),
                dim=-1
            ),
            'emotion': self.emotion_expert(combined_feat).squeeze()
        }

        # 门控融合（简化解码逻辑）
        final_output = {
            'AI伪造风险': 0.6 * outputs['fake_img'] + 0.4 * outputs['fake_text'],
            '图文不一致风险': (1 - outputs['incon']).clamp(0, 1),
            '情绪风险': outputs['emotion']
        }
        return final_output


# 风险检测器
class NewsRiskTester:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.thresholds = {
            'AI伪造风险': 0.65,
            '图文不一致风险': 0.55,
            '情绪风险': 0.7
        }

    def _load_model(self, path):
        model = RiskDetector()
        state_dict = torch.load(path, map_location=self.device)

        # 兼容性加载（自动跳过不匹配的层）
        filtered_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model.load_state_dict(filtered_dict, strict=False)
        return model.to(self.device).eval()

    def analyze_risk(self, feature):
        """ 输入1768维特征(numpy数组)，输出风险分析 """
        with torch.no_grad():
            tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
            outputs = self.model(tensor)

        risk_report = []
        for risk_type in ['AI伪造风险', '图文不一致风险', '情绪风险']:
            prob = outputs[risk_type].item()
            if prob > self.thresholds[risk_type]:
                risk_report.append(f"{risk_type}（置信度：{prob * 100:.1f}%）")

        return "检测到的风险：\n" + "\n".join(risk_report) if risk_report else "未检测到显著风险"


if __name__ == "__main__":
    # 示例使用
    tester = NewsRiskTester("best_model.pth")

    # 模拟输入（实际应替换为真实特征）
    demo_feature = np.random.randn(1768).astype(np.float32)

    # 获得检测结果
    print(tester.analyze_risk(demo_feature))
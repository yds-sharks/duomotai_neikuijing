import torch
import torch.nn as nn

class QwenRerankerModel(nn.Module):
    def __init__(self, backbone):  # ✅ 接收模型对象
        super().__init__()
        self.backbone = backbone    # ✅ 直接赋值，不再 from_pretrained
        hidden_size = self.backbone.config.hidden_size

        # 回归得分 head：输出连续值 [0,1]
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # 分类 head：输出 logits，后面接 BCEWithLogitsLoss
        self.cls_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, T, D]
        mask = attention_mask.unsqueeze(-1).float()
        sentence_rep = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # [B, D]
        sentence_rep = sentence_rep.to(self.score_head[0].weight.dtype)

        # 两个输出：score 用于回归，logits 用于分类
        score = self.score_head(sentence_rep).squeeze(-1)           # [B]
        label_logits = self.cls_head(sentence_rep).squeeze(-1)      # [B]

        return score, label_logits

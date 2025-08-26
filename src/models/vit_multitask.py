import timm
import torch.nn as nn
import torch.nn.functional as F

class ViTMultiTask(nn.Module):
    def __init__(self, backbone_name: str, n_model: int, n_year: int, n_view: int, n_type: int, img_size: int = 224, drop_path_rate: float = 0.0):
        super().__init__()
        # Use the cfg img_size so DINOv2 won't assume 518
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            img_size=img_size,
            drop_path_rate=drop_path_rate,  
        )
        # Enable gradient checkpointing to fit larger batches if available
        if hasattr(self.backbone, "set_grad_checkpointing"):
            try:
                self.backbone.set_grad_checkpointing(True)
            except Exception:
                pass

        feat_dim = self.backbone.num_features
        self.head_model = nn.Linear(feat_dim, n_model)
        self.head_year  = nn.Linear(feat_dim, n_year)
        self.head_view  = nn.Linear(feat_dim, n_view)
        self.head_type  = nn.Linear(feat_dim, n_type)

        for m in [self.head_model, self.head_year, self.head_view, self.head_type]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "model": self.head_model(feat),
            "year":  self.head_year(feat),
            "view":  self.head_view(feat),
            "type":  self.head_type(feat),
        }

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18


class SpatialResNet18(nn.Module):
    """

    Checkpoints live in:
        /oak/.../projects/spacenet/spacetorch/vissl_checkpoints/<model_name>/<blah.torch>

    To load trunk params:
        ckpt = torch.load(weight_path)
        model_params = ckpt["classy_state_dict"]["base_model"]["model"]["trunk"]
        model.load_state_dict(model_params)

    Layers have names like:
        "base_model.layer4.1"

    Example models:
        model_name: simclr_spatial_resnet18_fuzzy_swappedon_SineGrating2019_lwx5_checkpoints
        checkpoint: model_phase155.torch
        position_dir: /oak/stanford/groups/kalanit/biac2/kgs/projects/spacenet/spacetorch/positions/supervised_resnet18/resnet18_retinotopic_init_fuzzy_swappedon_SineGrating2019
    """

    def __init__(self):
        super(SpatialResNet18, self).__init__()
        self.base_model = resnet18(pretrained=False)

        self._feature_blocks = nn.ModuleDict(
            [  # type: ignore
                ("conv1", self.base_model.conv1),
                ("maxpool", self.base_model.maxpool),
                ("layer1_0", self.base_model.layer1[0]),
                ("layer1_1", self.base_model.layer1[1]),
                ("layer2_0", self.base_model.layer2[0]),
                ("layer2_1", self.base_model.layer2[1]),
                ("layer3_0", self.base_model.layer3[0]),
                ("layer3_1", self.base_model.layer3[1]),
                ("layer4_0", self.base_model.layer4[0]),
                ("layer4_1", self.base_model.layer4[1]),
                ("avgpool", self.base_model.avgpool),
            ]
        )

    def forward(self, x: torch.Tensor):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        maxpool = self.base_model.maxpool(x)

        x_1_0 = self.base_model.layer1[0](maxpool)
        x_1_1 = self.base_model.layer1[1](x_1_0)
        x_2_0 = self.base_model.layer2[0](x_1_1)
        x_2_1 = self.base_model.layer2[1](x_2_0)
        x_3_0 = self.base_model.layer3[0](x_2_1)
        x_3_1 = self.base_model.layer3[1](x_3_0)
        x_4_0 = self.base_model.layer4[0](x_3_1)
        x_4_1 = self.base_model.layer4[1](x_4_0)

        x = self.base_model.avgpool(x_4_1)
        flat_outputs = torch.flatten(x, 1)

        return flat_outputs

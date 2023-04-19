"""
Implements image encoders
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import copy
from collections import defaultdict

GROUP_NORM_LOOKUP = {
    16: 2,  # -> channels per group: 8
    32: 4,  # -> channels per group: 8
    64: 8,  # -> channels per group: 8
    128: 8,  # -> channels per group: 16
    256: 16,  # -> channels per group: 16
    512: 32,  # -> channels per group: 16
    1024: 32,  # -> channels per group: 32
    2048: 32,  # -> channels per group: 64
}


def convert_batch_norm(layer, new_norm="instance"):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    if new_norm == "group":
                        layer._modules[name] = torch.nn.GroupNorm(GROUP_NORM_LOOKUP[num_channels], num_channels)
                    elif new_norm == "instance":
                        layer._modules[name] = torch.nn.InstanceNorm2d(num_channels)

            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split(".")[0]
                sub_layer = getattr(layer, name)
                sub_layer = convert_batch_norm(sub_layer, new_norm)
                layer.__setattr__(name=name, value=sub_layer)
    return layer


def replace_bn(model, new_norm="instance"):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # Get current bn layer
            bn = getattr(model, name)
            # Create new layer
            if new_norm == "instance":
                new_n = nn.InstanceNorm2d(bn.num_features)

            elif new_norm == "group":
                new_n = nn.GroupNorm(1, bn.num_features)
            # Assign gn
            setattr(model, name, new_n)



class MultiHeadImgEncoder(nn.Module):
    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        shared_layers=3,
        global_size=0,
        color_size=128,
        density_size=128,
        spatials=None,
        norm_type="instance",
        input_dim=3,
        add_out_lvl=None,
        agg_fct="mean",
        **kwargs
    ):
        super().__init__()
        backbone_model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        if norm_type != "batch":
            backbone_model = convert_batch_norm(backbone_model, norm_type)
        if input_dim != 3:
            backbone_model.conv1 = torch.nn.Conv2d(input_dim, 64, 7, 2, 3, bias=False)

        feature_dims = [64, 64, 128, 256, 512, 1024]

        self.latent_size = feature_dims[num_layers]
        self.add_out_lvl = add_out_lvl

        if spatials is None:
            spatials = []
        self.spatials = spatials
        self.shared_layers = shared_layers

        self.store_latents = len(self.spatials) > 0

        self.shared_model = [
            ["conv1", backbone_model.conv1],
            ["bn1", backbone_model.bn1],
            ["relu", backbone_model.relu],
            ["maxpool", backbone_model.maxpool],  # /4
            ["layer1", backbone_model.layer1],  # /4
        ]
        if shared_layers >= 2:
            self.shared_model.append(["layer2", backbone_model.layer2])  # /8
        if shared_layers >= 3:
            self.shared_model.append(["layer3", backbone_model.layer3])  # /16
        if shared_layers >= 4:
            self.shared_model.append(["layer4", backbone_model.layer4])  # /32
        self.shared_model = nn.ModuleDict(self.shared_model)

        if self.store_latents:
            self.shared_latents = []  # self.register_buffer("shared_latents", torch.empty(1), persistent=False)

        def create_head(
            lin_out_size=None, spatial_in_size=None, spatial_out_size=None,
        ):
            head_model = []
            if shared_layers < 2:
                head_model.append(copy.deepcopy(backbone_model.layer2))  # /8
            if shared_layers < 3:
                head_model.append(copy.deepcopy(backbone_model.layer3))  # /8
            if shared_layers < 4:
                head_model.append(copy.deepcopy(backbone_model.layer4))  # /8
            if lin_out_size is not None:
                head_model.append(copy.deepcopy(backbone_model.avgpool))
                head_model.append(nn.Linear(self.latent_size, lin_out_size))
            else:
                head_model.append(nn.Conv2d(spatial_in_size, spatial_out_size, 1))

            return nn.Sequential(*head_model)
            
        if global_size > 0:
            if "global" in spatials:
                self.global_head = create_head(spatial_in_size=sum(feature_dims[: num_layers + 1]), spatial_out_size=global_size)
                # self.register_buffer("global_latents", torch.empty(1), persistent=False)
            else:
                self.global_head = create_head(global_size)

        if color_size > 0:
            if "color" in spatials:
                self.color_head = create_head(spatial_in_size=sum(feature_dims[: num_layers + 1]), spatial_out_size=color_size)
            else:
                self.color_head = create_head(color_size)
        if density_size > 0:
            if "density" in spatials:
                self.density_head = create_head(
                    spatial_in_size=sum(feature_dims[: num_layers + 1]), spatial_out_size=density_size
                )
            else:
                self.density_head = create_head(density_size)

        self.agg_fct = agg_fct

    def head_forward(
        self, head_model, x, store_latents, latent_sz=None,
    ):
        latents = []
        for head_idx, head_layer in enumerate(head_model):
            if head_idx == len(head_model) - 1:
                # last layer
                if store_latents:
                    # stack feature pyramid and pass through 1x1 conv
                    for i in range(len(latents)):
                        latents[i] = F.interpolate(latents[i], latent_sz, mode="bilinear", align_corners=True,)
                    # latents = torch.cat(latents, dim=1)
                    latents = head_layer(torch.cat(self.shared_latents + latents, 1))
                else:
                    x = torch.flatten(x, 1)
                    x = head_layer(x)
            else:
                x = head_layer(x)
                if store_latents:
                    latents.append(x)
        if store_latents:
            return x, latents
        return x

    def forward(self, cond):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        add_output = None
        x = cond
        # handle list of lists
        if isinstance(x, list) and isinstance(x[0], list) or (isinstance(x, torch.Tensor) and x.dim()==5):
            batch_outs = []
            for batch_idx in range(len(x)):
                view_outs = defaultdict(list)
                for view_x in x[batch_idx]:
                    view_out = self.forward(view_x.unsqueeze(0))
                    for k in view_out:
                        view_outs[k].append(view_out[k][0])
                # aggregate all views
                if self.agg_fct is not None:
                    for k, vals in view_outs.items():
                        if self.agg_fct == "mean":
                            view_outs[k] = torch.mean(torch.stack(vals,0),0)
                        elif self.agg_fct == "max":
                            view_outs[k] = torch.max(torch.stack(vals,0),0)
                        else:
                            raise NotImplementedError(f"Agg function {self.agg_fct} not supported")
                
                batch_outs.append(view_outs)
            return batch_outs
        out = {}
        if self.store_latents:
            shared_latents = [x]
        x = self.shared_model.conv1(x)
        x = self.shared_model.bn1(x)
        x = self.shared_model.relu(x)
        if self.store_latents:
            shared_latents.append(x)
        x = self.shared_model.maxpool(x)
        x = self.shared_model.layer1(x)
        if self.store_latents:
            shared_latents.append(x)

        if self.shared_layers >= 2:
            if self.add_out_lvl == 2:
                add_output = torch.clone(x)
            x = self.shared_model.layer2(x)
            if self.store_latents:
                shared_latents.append(x)

        if self.shared_layers >= 3:
            x = self.shared_model.layer3(x)
            if self.store_latents:
                shared_latents.append(x)

        if self.shared_layers >= 4:
            x = self.shared_model.layer4(x)
            if self.store_latents:
                shared_latents.append(x)
        if self.store_latents:
            latent_sz = shared_latents[1].shape[-2:]

            for i in range(len(shared_latents)):
                shared_latents[i] = F.interpolate(shared_latents[i], latent_sz, mode="bilinear", align_corners=True,)
            self.shared_latents = shared_latents

        if hasattr(self, "global_head"):
            if "global" in self.spatials:
                global_x, self.global_latents = self.head_forward(self.global_head, x, True, latent_sz)
            else:
                global_x = self.head_forward(self.global_head, x, False)
            out["global"] = global_x
        if hasattr(self, "color_head"):
            if "color" in self.spatials:
                color_x, self.color_latents = self.head_forward(self.color_head, x, True, latent_sz)
            else:
                color_x = self.head_forward(self.color_head, x, False)
            out["color"] = color_x
        if hasattr(self, "density_head"):
            if "density" in self.spatials:
                density_x, self.density_latents = self.head_forward(self.density_head, x, True, latent_sz)
            else:
                density_x = self.head_forward(self.density_head, x, False)

            out["density"] = density_x

        return out

   
    @classmethod
    def from_conf(cls, conf):
        # PyHocon construction
        return cls(
            backbone=conf.get("backbone"),
            pretrained=conf.get("pretrained"),
            num_layers=conf.get("num_layers"),
            shared_layers=conf.get("shared_layers"),
            global_size=conf.get("global_size"),
            color_size=conf.get("color_size"),
            density_size=conf.get("density_size"),
            spatials=conf.get("spatials"),
            norm_type=conf.get("norm_type"),
            input_dim=conf.get("input_dim", 3),
            add_out_lvl=conf.get("add_out_lvl", None),
            density_vae=conf.get("density_vae", False),
        )
    

class ImgEncoder_MultiHead_Art(nn.Module):
    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        shared_layers=3,
        global_size=0,
        color_size=128,
        density_size=128,
        art_size = 32,
        spatials=None,
        norm_type="instance",
        input_dim=3,
        add_out_lvl=None,
        agg_fct="mean",
        **kwargs
    ):
        super().__init__()
        backbone_model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        if norm_type != "batch":
            backbone_model = convert_batch_norm(backbone_model, norm_type)
        if input_dim != 3:
            backbone_model.conv1 = torch.nn.Conv2d(input_dim, 64, 7, 2, 3, bias=False)

        feature_dims = [64, 64, 128, 256, 512, 1024]

        self.latent_size = feature_dims[num_layers]
        self.add_out_lvl = add_out_lvl

        if spatials is None:
            spatials = []
        self.spatials = spatials
        self.shared_layers = shared_layers

        self.store_latents = len(self.spatials) > 0

        self.shared_model = [
            ["conv1", backbone_model.conv1],
            ["bn1", backbone_model.bn1],
            ["relu", backbone_model.relu],
            ["maxpool", backbone_model.maxpool],  # /4
            ["layer1", backbone_model.layer1],  # /4
        ]
        if shared_layers >= 2:
            self.shared_model.append(["layer2", backbone_model.layer2])  # /8
        if shared_layers >= 3:
            self.shared_model.append(["layer3", backbone_model.layer3])  # /16
        if shared_layers >= 4:
            self.shared_model.append(["layer4", backbone_model.layer4])  # /32
        self.shared_model = nn.ModuleDict(self.shared_model)

        if self.store_latents:
            self.shared_latents = []  # self.register_buffer("shared_latents", torch.empty(1), persistent=False)

        def create_head(
            lin_out_size=None, spatial_in_size=None, spatial_out_size=None,
        ):
            head_model = []
            if shared_layers < 2:
                head_model.append(copy.deepcopy(backbone_model.layer2))  # /8
            if shared_layers < 3:
                head_model.append(copy.deepcopy(backbone_model.layer3))  # /8
            if shared_layers < 4:
                head_model.append(copy.deepcopy(backbone_model.layer4))  # /8
            if lin_out_size is not None:
                head_model.append(copy.deepcopy(backbone_model.avgpool))
                head_model.append(nn.Linear(self.latent_size, lin_out_size))
            else:
                head_model.append(nn.Conv2d(spatial_in_size, spatial_out_size, 1))

            return nn.Sequential(*head_model)
            
        if global_size > 0:
            if "global" in spatials:
                self.global_head = create_head(spatial_in_size=sum(feature_dims[: num_layers + 1]), spatial_out_size=global_size)
                # self.register_buffer("global_latents", torch.empty(1), persistent=False)
            else:
                self.global_head = create_head(global_size)

        if color_size > 0:
            if "color" in spatials:
                self.color_head = create_head(spatial_in_size=sum(feature_dims[: num_layers + 1]), spatial_out_size=color_size)
            else:
                self.color_head = create_head(color_size)
        if density_size > 0:
            if "density" in spatials:
                self.density_head = create_head(
                    spatial_in_size=sum(feature_dims[: num_layers + 1]), spatial_out_size=density_size
                )
            else:
                self.density_head = create_head(density_size)

        if art_size > 0:
            if "articulation" in spatials:
                self.density_head = create_head(
                    spatial_in_size=sum(feature_dims[: num_layers + 1]), spatial_out_size=art_size
                )
            else:
                self.articulation_head = create_head(art_size)

        self.agg_fct = agg_fct

    def head_forward(
        self, head_model, x, store_latents, latent_sz=None,
    ):
        latents = []
        for head_idx, head_layer in enumerate(head_model):
            if head_idx == len(head_model) - 1:
                # last layer
                if store_latents:
                    # stack feature pyramid and pass through 1x1 conv
                    for i in range(len(latents)):
                        latents[i] = F.interpolate(latents[i], latent_sz, mode="bilinear", align_corners=True,)
                    # latents = torch.cat(latents, dim=1)
                    latents = head_layer(torch.cat(self.shared_latents + latents, 1))
                else:
                    x = torch.flatten(x, 1)
                    x = head_layer(x)
            else:
                x = head_layer(x)
                if store_latents:
                    latents.append(x)
        if store_latents:
            return x, latents
        return x

    def forward(self, cond):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        add_output = None
        x = cond
        # handle list of lists
        if isinstance(x, list) and isinstance(x[0], list) or (isinstance(x, torch.Tensor) and x.dim()==5):
            batch_outs = []
            for batch_idx in range(len(x)):
                view_outs = defaultdict(list)
                for view_x in x[batch_idx]:
                    view_out = self.forward(view_x.unsqueeze(0))
                    for k in view_out:
                        view_outs[k].append(view_out[k][0])
                # aggregate all views
                if self.agg_fct is not None:
                    for k, vals in view_outs.items():
                        if self.agg_fct == "mean":
                            view_outs[k] = torch.mean(torch.stack(vals,0),0)
                        elif self.agg_fct == "max":
                            view_outs[k] = torch.max(torch.stack(vals,0),0)
                        else:
                            raise NotImplementedError(f"Agg function {self.agg_fct} not supported")
                
                batch_outs.append(view_outs)
            return batch_outs
        out = {}
        if self.store_latents:
            shared_latents = [x]
        x = self.shared_model.conv1(x)
        x = self.shared_model.bn1(x)
        x = self.shared_model.relu(x)
        if self.store_latents:
            shared_latents.append(x)
        x = self.shared_model.maxpool(x)
        x = self.shared_model.layer1(x)
        if self.store_latents:
            shared_latents.append(x)

        if self.shared_layers >= 2:
            if self.add_out_lvl == 2:
                add_output = torch.clone(x)
            x = self.shared_model.layer2(x)
            if self.store_latents:
                shared_latents.append(x)

        if self.shared_layers >= 3:
            x = self.shared_model.layer3(x)
            if self.store_latents:
                shared_latents.append(x)

        if self.shared_layers >= 4:
            x = self.shared_model.layer4(x)
            if self.store_latents:
                shared_latents.append(x)
        if self.store_latents:
            latent_sz = shared_latents[1].shape[-2:]

            for i in range(len(shared_latents)):
                shared_latents[i] = F.interpolate(shared_latents[i], latent_sz, mode="bilinear", align_corners=True,)
            self.shared_latents = shared_latents

        if hasattr(self, "global_head"):
            if "global" in self.spatials:
                global_x, self.global_latents = self.head_forward(self.global_head, x, True, latent_sz)
            else:
                global_x = self.head_forward(self.global_head, x, False)
            out["global"] = global_x
        if hasattr(self, "color_head"):
            if "color" in self.spatials:
                color_x, self.color_latents = self.head_forward(self.color_head, x, True, latent_sz)
            else:
                color_x = self.head_forward(self.color_head, x, False)
            out["color"] = color_x
        if hasattr(self, "density_head"):
            if "density" in self.spatials:
                density_x, self.density_latents = self.head_forward(self.density_head, x, True, latent_sz)
            else:
                density_x = self.head_forward(self.density_head, x, False)

            out["density"] = density_x

        if hasattr(self, "articulation_head"):
            if "articulation" in self.spatials:
                articulation_x, self.articulation_latents = self.head_forward(self.articulation_head, x, True, latent_sz)
            else:
                articulation_x = self.head_forward(self.articulation_head, x, False)

            out["articulation"] = articulation_x

        return out

   
    @classmethod
    def from_conf(cls, conf):
        # PyHocon construction
        return cls(
            backbone=conf.get("backbone"),
            pretrained=conf.get("pretrained"),
            num_layers=conf.get("num_layers"),
            shared_layers=conf.get("shared_layers"),
            global_size=conf.get("global_size"),
            color_size=conf.get("color_size"),
            density_size=conf.get("density_size"),
            spatials=conf.get("spatials"),
            norm_type=conf.get("norm_type"),
            input_dim=conf.get("input_dim", 3),
            add_out_lvl=conf.get("add_out_lvl", None),
            density_vae=conf.get("density_vae", False),
        )

if __name__ == "__main__":
    image_encoder = ImgEncoder_MultiHead_Art()
    img = torch.randn((3,3,128,128))
    out = image_encoder(img)
    print("out texture", out["color"].shape)
    print("out density", out["density"].shape)
    print("out articulation", out["articulation"].shape)
    # print("out", out)
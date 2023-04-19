import torch
from torch import nn
import sys

sys.path.append("./")
from opt import get_opts
import torch.nn.init as init
import math
from einops import rearrange, reduce


class CodeLibraryVoxel(nn.Module):
    """
    Store various codes.
    """

    def __init__(self, hparams):
        super(CodeLibraryVoxel, self).__init__()

        self.embedding_instance = torch.nn.Embedding(
            hparams.N_max_objs, hparams.N_obj_code_length
        )

    def forward(self, instance_ids):
        ret_dict = dict()
        ret_dict["embedding_instance"] = self.embedding_instance(
            instance_ids
        ).unsqueeze(0)

        return ret_dict


class CodeLibraryRefNeRF(nn.Module):
    """
    Store various codes.
    """

    def __init__(self, hparams):
        super(CodeLibraryRefNeRF, self).__init__()

        self.embedding_instance_shape = torch.nn.Embedding(
            hparams.N_max_objs, hparams.N_obj_code_length
        )
        self.embedding_instance_appearance = torch.nn.Embedding(
            hparams.N_max_objs, hparams.N_obj_code_length
        )
        init.xavier_uniform_(self.embedding_instance_shape.weight)
        init.xavier_uniform_(self.embedding_instance_appearance.weight)

    def forward(self, instance_ids):
        ret_dict = dict()
        ret_dict["embedding_instance_shape"] = self.embedding_instance_shape(
            instance_ids
        )
        ret_dict["embedding_instance_appearance"] = self.embedding_instance_appearance(
            instance_ids
        )

        return ret_dict


class CodeLibraryVanillaDisentagled(nn.Module):
    """
    Store various codes.
    """

    def __init__(self, hparams):
        super(CodeLibraryVanillaDisentagled, self).__init__()

        self.embedding_instance_shape = torch.nn.Embedding(
            hparams.N_max_objs, hparams.N_obj_code_length
        )
        self.embedding_instance_appearance = torch.nn.Embedding(
            hparams.N_max_objs, hparams.N_obj_code_length
        )
        init.xavier_uniform_(self.embedding_instance_shape.weight)
        init.xavier_uniform_(self.embedding_instance_appearance.weight)

    def forward(self, instance_ids):
        ret_dict = dict()
        ret_dict["embedding_instance_shape"] = self.embedding_instance_shape(
            instance_ids
        )
        ret_dict["embedding_instance_appearance"] = self.embedding_instance_appearance(
            instance_ids
        )

        return ret_dict


class CodeLibraryVanilla(nn.Module):
    """
    Store various codes.
    """

    def __init__(self, hparams):
        super(CodeLibraryVanilla, self).__init__()

        self.embedding_instance = torch.nn.Embedding(
            hparams.N_max_objs, hparams.N_obj_code_length
        )
        # self.embedding_instance_appearance = torch.nn.Embedding(
        #     hparams.N_max_objs,
        #     hparams.N_obj_code_length
        # )
        torch.nn.init.normal_(
            self.embedding_instance.weight.data,
            0.0,
            1.0 / math.sqrt(hparams.N_obj_code_length),
        )
        # init.xavier_uniform_(self.embedding_instance_appearance.weight)

    def forward(self, instance_ids):
        ret_dict = dict()
        ret_dict["embedding_instance"] = self.embedding_instance(instance_ids)
        # ret_dict["embedding_instance_appearance"] = self.embedding_instance_appearance(instance_ids
        # )

        return ret_dict


class CodeLibraryVanilla_VAD(nn.Module):
    """
    Store various codes.
    """

    def __init__(self, hparams):
        super(CodeLibraryVanilla_VAD, self).__init__()

        self.weight_mu = nn.Parameter(
            torch.Tensor(hparams.N_max_objs, hparams.N_obj_code_length)
        )
        self.weight_logvar = nn.Parameter(
            torch.Tensor(hparams.N_max_objs, hparams.N_obj_code_length)
        )

        mu_init_std = 1.0 / math.sqrt(hparams.N_obj_code_length)
        torch.nn.init.normal_(
            self.weight_mu.data,
            0.0,
            mu_init_std,
        )

        logvar_init_std = 1.0 / math.sqrt(hparams.N_obj_code_length)
        torch.nn.init.normal_(
            self.weight_logvar.data,
            0.0,
            logvar_init_std,
        )

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        batch_latent = mu + eps * std
        return batch_latent

    def forward(self, instance_ids, is_train=True):
        # forward as in encode() function in VAE encoder
        mu = self.weight_mu[instance_ids]
        logvar = self.weight_logvar[instance_ids]

        # reparameterization trick
        # std = torch.exp(0.5*logvar)
        # eps = torch.randn_like(std)
        # batch_latent = mu + eps*std

        if is_train:
            batch_latent = self.reparametrize(mu, logvar)
        else:
            batch_latent = mu

        ret_dict = dict()
        ret_dict["embedding_instance"] = batch_latent
        ret_dict["mu"] = mu
        ret_dict["logvar"] = logvar

        return ret_dict


class CodeLibraryVanilla_VAD_Disentagled(nn.Module):
    """
    Store various codes.
    """

    def __init__(self, hparams):
        super(CodeLibraryVanilla_VAD_Disentagled, self).__init__()

        self.weight_mu_shape = nn.Parameter(
            torch.Tensor(hparams.N_max_objs, hparams.N_obj_code_length)
        )
        self.weight_logvar_shape = nn.Parameter(
            torch.Tensor(hparams.N_max_objs, hparams.N_obj_code_length)
        )

        self.weight_mu_app = nn.Parameter(
            torch.Tensor(hparams.N_max_objs, hparams.N_obj_code_length)
        )
        self.weight_logvar_app = nn.Parameter(
            torch.Tensor(hparams.N_max_objs, hparams.N_obj_code_length)
        )

        mu_init_std = 1.0 / math.sqrt(hparams.N_obj_code_length)
        logvar_init_std = 1.0 / math.sqrt(hparams.N_obj_code_length)

        torch.nn.init.normal_(
            self.weight_mu_shape.data,
            0.0,
            mu_init_std,
        )

        torch.nn.init.normal_(
            self.weight_logvar_shape.data,
            0.0,
            logvar_init_std,
        )

        torch.nn.init.normal_(
            self.weight_mu_app.data,
            0.0,
            mu_init_std,
        )

        torch.nn.init.normal_(
            self.weight_logvar_app.data,
            0.0,
            logvar_init_std,
        )

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        batch_latent = mu + eps * std
        return batch_latent

    def forward(self, instance_ids):
        # forward as in encode() function in VAE encoder
        mu_shape = self.weight_mu_shape[instance_ids]
        logvar_shape = self.weight_logvar_shape[instance_ids]

        mu_app = self.weight_mu_app[instance_ids]
        logvar_app = self.weight_logvar_app[instance_ids]

        # reparameterization trick
        batch_latent_shape = self.reparametrize(mu_shape, logvar_shape)
        batch_latent_app = self.reparametrize(mu_app, logvar_app)

        ret_dict = dict()
        ret_dict["embedding_instance_shape"] = batch_latent_shape
        ret_dict["embedding_instance_appearance"] = batch_latent_app

        ret_dict["mu_shape"] = mu_shape
        ret_dict["logvar_shape"] = logvar_shape
        ret_dict["mu_app"] = mu_app
        ret_dict["logvar_app"] = logvar_app

        return ret_dict


class CodeLibrary(nn.Module):
    """
    Store various codes.
    """

    def __init__(self, hparams):
        super(CodeLibrary, self).__init__()

        self.embedding_instance = torch.nn.Embedding(
            hparams.N_max_objs, hparams.N_obj_code_length
        )

    def forward(self, inputs):
        ret_dict = dict()
        if "instance_ids" in inputs:
            # ret_dict["embedding_instance"] = self.embedding_instance(
            #     inputs["instance_ids"].squeeze()
            # )
            # shape (1,128) for voxel grid optimization
            ret_dict["embedding_instance"] = self.embedding_instance(
                inputs["instance_ids"].squeeze()[0]
            ).unsqueeze(0)

        return ret_dict


class CodeLibraryShapeAppearance(nn.Module):
    """
    Store various codes.
    """

    def __init__(self, hparams):
        super(CodeLibraryShapeAppearance, self).__init__()

        self.embedding_instance_shape = torch.nn.Embedding(
            hparams.N_max_objs, hparams.N_obj_code_length
        )
        self.embedding_instance_appearance = torch.nn.Embedding(
            hparams.N_max_objs, hparams.N_obj_code_length
        )

    def forward(self, inputs):
        ret_dict = dict()

        # if 'frame_idx' in inputs:
        #     ret_dict['embedding_a'] = self.embedding_a(inputs['frame_idx'].squeeze())
        # print("inputs[instance_ids].squeeze()", inputs["instance_ids"].squeeze().shape)
        # print("inputs[instance_ids].squeeze()", inputs["instance_ids"].shape)

        # print("self.embedding_instance(inputs[instance_ids].squeeze()", self.embedding_instance(
        #         inputs["instance_ids"].squeeze().shape))
        if "instance_ids" in inputs:
            ret_dict["embedding_instance_shape"] = self.embedding_instance_shape(
                inputs["instance_ids"].squeeze()
            )
            ret_dict[
                "embedding_instance_appearance"
            ] = self.embedding_instance_appearance(inputs["instance_ids"].squeeze())

        return ret_dict


class CodeLibraryBckgObj(nn.Module):
    """
    Store various codes.
    """

    def __init__(self, hparams):
        super(CodeLibraryBckgObj, self).__init__()

        self.embedding_instance = torch.nn.Embedding(
            hparams.N_max_objs, hparams.N_obj_code_length
        )

        self.embedding_backgrounds = torch.nn.Embedding(hparams.N_max_objs, 128)

    def forward(self, inputs):
        ret_dict = dict()

        # if 'frame_idx' in inputs:
        #     ret_dict['embedding_a'] = self.embedding_a(inputs['frame_idx'].squeeze())
        # print("inputs[instance_ids].squeeze()", inputs["instance_ids"].squeeze().shape)
        # print("inputs[instance_ids].squeeze()", inputs["instance_ids"].shape)

        # print("self.embedding_instance(inputs[instance_ids].squeeze()", self.embedding_instance(
        #         inputs["instance_ids"].squeeze().shape))
        if "instance_ids" in inputs:
            ret_dict["embedding_instance"] = self.embedding_instance(
                inputs["instance_ids"].squeeze()
            )
            ret_dict["embedding_backgrounds"] = self.embedding_backgrounds(
                inputs["instance_ids"].squeeze()
            )

        return ret_dict


class CodeLibraryBckgObjShapeApp(nn.Module):
    """
    Store various codes.
    """

    def __init__(self, hparams):
        super(CodeLibraryBckgObjShapeApp, self).__init__()

        # self.embedding_instance = torch.nn.Embedding(
        #     hparams.N_max_objs,
        #     hparams.N_obj_code_length
        # )

        self.embedding_instance_shape = torch.nn.Embedding(
            hparams.N_max_objs, hparams.N_obj_code_length
        )
        self.embedding_instance_appearance = torch.nn.Embedding(
            hparams.N_max_objs, hparams.N_obj_code_length
        )

        self.embedding_backgrounds = torch.nn.Embedding(hparams.N_max_objs, 128)

    def forward(self, inputs):
        ret_dict = dict()

        # if 'frame_idx' in inputs:
        #     ret_dict['embedding_a'] = self.embedding_a(inputs['frame_idx'].squeeze())
        # print("inputs[instance_ids].squeeze()", inputs["instance_ids"].squeeze().shape)
        # print("inputs[instance_ids].squeeze()", inputs["instance_ids"].shape)

        # print("self.embedding_instance(inputs[instance_ids].squeeze()", self.embedding_instance(
        #         inputs["instance_ids"].squeeze().shape))
        if "instance_ids" in inputs:
            ret_dict["embedding_instance_shape"] = self.embedding_instance_shape(
                inputs["instance_ids"].squeeze()
            )
            ret_dict[
                "embedding_instance_appearance"
            ] = self.embedding_instance_appearance(inputs["instance_ids"].squeeze())
            # ret_dict["embedding_backgrounds"] = self.embedding_backgrounds(
            #     inputs["instance_ids"].squeeze()
            # )
            ret_dict["embedding_backgrounds"] = self.embedding_backgrounds(
                inputs["instance_ids"].squeeze()[0]
            ).unsqueeze(0)

        return ret_dict


if __name__ == "__main__":
    # conf_cli = OmegaConf.from_cli()
    # # conf_dataset = OmegaConf.load(conf_cli.dataset_config)
    # conf_default = OmegaConf.load("../config/default_conf.yml")
    # # # merge conf with the priority
    # # conf_merged = OmegaConf.merge(conf_default, conf_dataset, conf_cli)

    hparams = get_opts()
    code_library = CodeLibraryVanilla_VAD(hparams)
    inputs = {}
    H = 240
    W = 320
    instance_id = 1
    instance_mask = torch.ones((H, W))
    instance_mask = instance_mask.view(-1)

    inputs = torch.ones_like(instance_mask).long() * instance_id
    print("inputs", inputs.shape)
    ret_dict = code_library.forward(inputs)

    for k, v in ret_dict.items():
        print(k, v.shape)

    def KLD(mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        KLD = torch.mean(KLD)
        return KLD

    def KLD_loss(mu, log_var, kl_std=0.25):
        std = torch.exp(0.5 * log_var)
        gt_dist = torch.distributions.normal.Normal(
            torch.zeros_like(mu), torch.ones_like(std) * kl_std
        )
        sampled_dist = torch.distributions.normal.Normal(mu, std)
        kl = torch.distributions.kl.kl_divergence(sampled_dist, gt_dist)  # reversed KL
        kl_loss = reduce(kl, "b ... -> b (...)", "mean").mean()
        return kl_loss

    kld = KLD_loss(ret_dict["mu"][0], ret_dict["logvar"][0])

    print("kld", kld)

    # for i in range(0, 76800, 3840):

    #     latents_dict_chunk = dict()
    #     for k, v in ret_dict.items():
    #         latents_dict_chunk[k] = v[i : i + 3840]

    #     for k,v in latents_dict_chunk.items():
    #         print(k,v.shape)
    # print("ret_dict", ret_dict["embedding_instance_shape"].shape)
    # print("ret_dict", ret_dict["embedding_instance_appearance"].shape)
    # print("ret_dict", ret_dict["embedding_backgrounds"].shape)

    # from models.nerf import StyleGenerator2D

    # decoder = StyleGenerator2D()

    # z = ret_dict["embedding_backgrounds"]
    # print("z", z.shape)
    # w = decoder(z=z)
    # print("w", w.shape)

    # from models.nerf import ObjectBckgNeRFGSN
    # nerf_coarse = ObjectBckgNeRFGSN(hparams)

    # xyz = torch.randn(1, 2048, 96, 3)

    # bckg_code, _ = nerf_coarse.sample_local_latents(z, xyz)

    # print("bckg_code", bckg_code.shape)

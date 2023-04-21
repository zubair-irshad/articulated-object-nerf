import torch
from torch import nn
import sys

sys.path.append("./")
from opt import get_opts
import torch.nn.init as init
import math
from einops import rearrange, reduce


class CodeLibraryArticulated(nn.Module):
    """
    Store various codes.
    """

    def __init__(self, hparams):
        super(CodeLibraryArticulated, self).__init__()

        N_max_articulations = 10
        N_art_code_length = 32
        self.embedding_instance_shape = torch.nn.Embedding(
            hparams.N_max_objs, hparams.N_obj_code_length
        )
        self.embedding_instance_appearance = torch.nn.Embedding(
            hparams.N_max_objs, hparams.N_obj_code_length
        )

        self.embedding_instance_articulation = torch.nn.Embedding(
            N_max_articulations, N_art_code_length
        )
        init.xavier_uniform_(self.embedding_instance_shape.weight)
        init.xavier_uniform_(self.embedding_instance_appearance.weight)
        init.xavier_uniform_(self.embedding_instance_articulation.weight)

    def forward(self, batch, is_test=False):
        ret_dict = dict()
        ret_dict["density"] = self.embedding_instance_shape(batch["instance_id"])
        ret_dict["color"] = self.embedding_instance_appearance(batch["instance_id"])

        if is_test:
            interpolated_embeddings = self.get_interpolated_articulations(
                max_interpolations=2
            )
            ret_dict["articulation"] = interpolated_embeddings[batch["articulation_id"]]
        else:
            ret_dict["articulation"] = self.embedding_instance_articulation(
                batch["articulation_id"]
            )

        return ret_dict

    def get_interpolated_articulations(self, max_interpolations=2):
        N_max_articulations = 10
        interpolated_embeddings = torch.zeros(
            N_max_articulations * max_interpolations, 32
        ).to(self.embedding_instance_articulation.device)
        for i in range(N_max_articulations):
            embedding_articulation = self.embedding_instance_articulation(
                torch.tensor(i, dtype=torch.long).to(
                    self.embedding_instance_articulation.device
                )
            )
            interpolated_embeddings[i * 2] = embedding_articulation

        for i in range(N_max_articulations - 1):
            interpolated_embeddings[i * 2 + 1] = (
                interpolated_embeddings[i * 2] + interpolated_embeddings[i * 2 + 2]
            ) / 2

        return interpolated_embeddings


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

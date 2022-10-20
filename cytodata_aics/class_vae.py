from turtle import forward
from serotiny.models.vae.image_vae import ImageVAE
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import os
import pickle

class ImageClassVAE(ImageVAE):
    def __init__(self,         
            latent_dim,
            in_channels,
            hidden_channels,
            max_pool_layers,
            input_dims,
            x_label,
            optimizer= torch.optim.Adam,
            beta = 1.0,
            id_label = None,
            non_linearity = None,
            decoder_non_linearity = None,
            loss_mask_label= None,
            reconstruction_loss=nn.MSELoss(reduction="none"),
            skip_connections = True,
            batch_norm = True,
            mode= "3d",
            prior = None,
            kernel_size = 3,
            cache_outputs = ("test",),
            encoder_clamp = 6,):
        super().__init__(
                in_channels=in_channels,
                hidden_channels=hidden_channels, 
                max_pool_layers=max_pool_layers,
                input_dims=input_dims,
                latent_dim=latent_dim,
                x_label=x_label,
                id_label=id_label,
                beta=beta,
                reconstruction_loss=reconstruction_loss,
                cache_outputs=cache_outputs,
                optimizer=optimizer,
                prior=prior,)
        # a classifier with two layers
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 5),
            nn.Softmax()
        )
        self.class_criterion = nn.CrossEntropyLoss()
    
    def forward(self, batch, decode=False, compute_loss=False, **kwargs):
        out = super().forward(batch, decode=decode, compute_loss=compute_loss, **kwargs)
        if len(out) == 2:
            return out
        
        (
            x_hat,
            z_parts,
            z_parts_params,
            z_composed,
            vae_loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
        ) = out

        pred_class = self.classifier(z_parts["image"])
        class_loss = self.class_criterion(pred_class, batch['class'].ravel())
        pred_class = torch.argmax(pred_class,axis=1)
        if np.random.choice(2):
            print("class", class_loss)
            loss = class_loss
        else:
            print("vae", vae_loss)
            loss = vae_loss
        print(loss, class_loss, vae_loss)
        return (
            x_hat,
            z_parts,
            z_parts_params,
            z_composed,
            loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
            pred_class,
            batch,
        )

    def _step(self, stage, batch, batch_idx, logger):
        (
            x_hat,
            z_parts,
            z_parts_params,
            z_composed,
            loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
            pred_class,
            batch,
        ) = self.forward(batch, decode=True, compute_loss=True)

        results = self.make_results_dict(
            stage,
            batch,
            loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
            z_parts,
            z_parts_params,
            z_composed,
            x_hat,
            pred_class,
        )

        self.log_metrics(stage, results, logger, batch[self.hparams.x_label].shape[0])

        return results
    
    
    
    
    def make_results_dict(
        self,
        stage,
        batch,
        loss,
        reconstruction_loss,
        kld_loss,
        kld_per_part,
        z_parts,
        z_parts_params,
        z_composed,
        x_hat,
        pred_class,
    ):
        results = {
            "loss": loss,
            f"{stage}_loss": loss.detach().cpu(),  # for epoch end logging purposes
            "kld_loss": kld_loss.detach().cpu(),
        }

        for part, z_comp_part in z_composed.items():
            results.update(
                {
                    f"z_composed/{part}": z_comp_part.detach().cpu(),
                }
            )

        for part, recon_part in reconstruction_loss.items():
            results.update(
                {
                    f"reconstruction_loss/{part}": recon_part.detach().cpu(),
                }
            )

        for part, z_part in z_parts.items():
            results.update(
                {
                    f"z_parts/{part}": z_part.detach().cpu(),
                    f"z_parts_params/{part}": z_parts_params[part].detach().cpu(),
                    f"kld/{part}": kld_per_part[part].detach().float().cpu(),
                }
            )

        if self.hparams.id_label is not None:
            if self.hparams.id_label in batch:
                ids = batch[self.hparams.id_label].detach().cpu()
                results.update({self.hparams.id_label: ids, "id": ids})

        return results

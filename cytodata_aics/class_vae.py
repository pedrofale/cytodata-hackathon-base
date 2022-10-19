from turtle import forward
from serotiny.models.vae.image_vae import ImageVAE
import torch.nn as nn

class ImageClassVAE(ImageVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # a classifier with two layers
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 5),
            nn.Softmax()
        )
        self.class_criterion = nn.CrossEntropyLoss()
    
    def forward(self, batch, decode=False, compute_loss=False, **kwargs):
        (
            x_hat,
            z_parts,
            z_parts_params,
            z_composed,
            loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
        ) = super().forward(self, batch, decode=decode, compute_loss=compute_loss, **kwargs)

        pred_class = self.classifier(z_parts["image"])
        class_loss = self.class_criterion(pred_class, batch['class'].ravel())
        loss = class_loss + loss

        return (
            x_hat,
            z_parts,
            z_parts_params,
            z_composed,
            loss,
            reconstruction_loss,
            kld_loss,
            kld_per_part,
        )

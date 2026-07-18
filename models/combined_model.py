import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
from einops import reduce
from torch.utils.tensorboard import SummaryWriter
import os

# add paths in model/__init__.py for new models
from models import * 

class CombinedModel(pl.LightningModule):
    def __init__(self, specs):
        super().__init__()
        self.specs = specs
        self.metric_writers = {}
        self.log_every_n_steps = max(1, int(specs.get("log_every_n_steps", 1)))

        self.task = specs['training_task'] # 'combined' or 'modulation' or 'diffusion'

        if self.task in ('combined', 'modulation'):
            self.sdf_model = SdfModel(specs=specs) 

            feature_dim = specs["SdfModelSpecs"]["latent_dim"] # latent dim of pointnet 
            modulation_dim = feature_dim*3 # latent dim of modulation
            latent_std = specs.get("latent_std", 0.25) # std of target gaussian distribution of latent space
            hidden_dims = [modulation_dim, modulation_dim, modulation_dim, modulation_dim, modulation_dim]
            self.vae_model = BetaVAE(in_channels=feature_dim*3, latent_dim=modulation_dim, hidden_dims=hidden_dims, kl_std=latent_std)

        if self.task in ('combined', 'diffusion'):
            self.diffusion_model = DiffusionModel(model=DiffusionNet(**specs["diffusion_model_specs"]), **specs["diffusion_specs"]) 
 

    def training_step(self, x, idx):

        if self.task == 'combined':
            return self.train_combined(x)
        elif self.task == 'modulation':
            return self.train_modulation(x)
        elif self.task == 'diffusion':
            return self.train_diffusion(x)


    def validation_step(self, x, idx):

        if self.task == 'combined':
            losses = self.combined_losses(x)
        elif self.task == 'modulation':
            losses = self.modulation_losses(x)
        elif self.task == 'diffusion':
            losses = self.diffusion_losses(x)
        else:
            return None

        if losses is None:
            return None

        return {
            key: value.detach()
            for key, value in losses.items()
            if value is not None
        }


    def validation_epoch_end(self, outputs):

        outputs = [output for output in outputs if output is not None]
        if len(outputs) == 0:
            return None

        losses = {}
        for key in outputs[0]:
            values = [output[key].float() for output in outputs if key in output]
            if len(values) > 0:
                losses[key] = torch.stack(values).mean()

        self.write_losses("val", losses, self.global_step)
        return losses["loss"]


    def on_train_end(self):

        for writer in self.metric_writers.values():
            writer.close()
        self.metric_writers = {}
        

    def configure_optimizers(self):

        if self.task == 'combined':
            params_list = [
                    { 'params': list(self.sdf_model.parameters()) + list(self.vae_model.parameters()), 'lr':self.specs['sdf_lr'] },
                    { 'params': self.diffusion_model.parameters(), 'lr':self.specs['diff_lr'] }
                ]
        elif self.task == 'modulation':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['sdf_lr'] }
                ]
        elif self.task == 'diffusion':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['diff_lr'] }
                ]

        optimizer = torch.optim.Adam(params_list)
        return {
                "optimizer": optimizer,
                # "lr_scheduler": {
                # "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50000, threshold=0.0002, min_lr=1e-6, verbose=False),
                # "monitor": "total"
                # }
        }


    #-----------different training steps for sdf modulation, diffusion, combined----------

    def modulation_losses(self, x):

        xyz = x['xyz'] # (B, N, 3)
        gt = x['gt_sdf'] # (B, N)
        pc = x['point_cloud'] # (B, 1024, 3)

        # STEP 1: obtain reconstructed plane feature and latent code 
        plane_features = self.sdf_model.pointnet.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)
        out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]

        # STEP 2: pass recon back to GenSDF pipeline 
        pred_sdf = self.sdf_model.forward_with_plane_features(reconstructed_plane_feature, xyz)
        
        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"] )
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        loss = sdf_loss + vae_loss

        return {"loss": loss, "sdf": sdf_loss, "vae": vae_loss}


    def get_metric_writer(self, split):

        if split not in self.metric_writers:
            logger = getattr(self, "logger", None)
            trainer = getattr(self, "trainer", None)
            log_dir = logger.log_dir if logger is not None else None
            if log_dir is None and trainer is not None:
                log_dir = trainer.default_root_dir
            if log_dir is None:
                return None
            self.metric_writers[split] = SummaryWriter(os.path.join(log_dir, split))
        return self.metric_writers[split]


    def write_losses(self, split, losses, step):

        trainer = getattr(self, "trainer", None)
        if trainer is not None and not trainer.is_global_zero:
            return
        if not self.should_write_losses(split, step):
            return

        writer = self.get_metric_writer(split)
        if writer is None:
            return
        for key, value in losses.items():
            if value is None:
                continue
            writer.add_scalar(key, value.detach().float().mean().cpu().item(), step)
        writer.flush()


    def should_write_losses(self, split, step):

        if self.log_every_n_steps <= 1:
            return True
        if split != "train":
            return step % self.log_every_n_steps == 0
        return (step + 1) % self.log_every_n_steps == 0


    def train_modulation(self, x):

        losses = self.modulation_losses(x)
        if losses is None:
            return None

        self.write_losses("train", losses, self.global_step)

        return losses["loss"]


    def diffusion_losses(self, x):


        latent = x['latent'] # (B, D)

        # unconditional training if cond is None 
        if self.specs['diffusion_model_specs']['cond']:
            cond = x.get('conditioning', x.get('point_cloud'))
        else:
            cond = None

        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
        # visualizing loss curves can help with debugging if training is unstable
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent, perturbed_cond = self.diffusion_model.diffusion_model_from_latent(latent, cond=cond)

        return {
                        "loss": diff_loss,
                        "diff100": diff_100_loss, # note that this can appear as nan when the training batch does not have sampled timesteps < 100
                        "diff1000": diff_1000_loss
                    }


    def train_diffusion(self, x):

        losses = self.diffusion_losses(x)
        self.write_losses("train", losses, self.global_step)

        return losses["loss"]

    # the first half is the same as "train_sdf_modulation"
    # the reconstructed latent is used as input to the diffusion model, rather than loading latents from the dataloader as in "train_diffusion"
    def combined_losses(self, x):
        xyz = x['xyz'] # (B, N, 3)
        gt = x['gt_sdf'] # (B, N)
        pc = x['point_cloud'] # (B, 1024, 3)

        # STEP 1: obtain reconstructed plane feature for SDF and latent code for diffusion
        plane_features = self.sdf_model.pointnet.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)
        #print("plane feat shape: ", feat.shape)
        out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1] # [B, D*3, resolution, resolution], [B, D*3]

        # STEP 2: pass recon back to GenSDF pipeline 
        pred_sdf = self.sdf_model.forward_with_plane_features(reconstructed_plane_feature, xyz)
        
        # STEP 3: losses for VAE and SDF 
        try:
            vae_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"] )
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch
        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        # STEP 4: use latent as input to diffusion model
        if self.specs['diffusion_model_specs']['cond']:
            cond = x.get('conditioning', pc)
        else:
            cond = None
        diff_loss, diff_100_loss, diff_1000_loss, pred_latent, perturbed_cond = self.diffusion_model.diffusion_model_from_latent(latent, cond=cond)
        
        # STEP 5: use predicted / reconstructed latent to run SDF loss 
        generated_plane_feature = self.vae_model.decode(pred_latent)
        generated_sdf_pred = self.sdf_model.forward_with_plane_features(generated_plane_feature, xyz)
        generated_sdf_loss = F.l1_loss(generated_sdf_pred.squeeze(), gt.squeeze())

        # surface weight could prioritize points closer to surface but we did not notice better results when using it 
        #surface_weight = torch.exp(-50 * torch.abs(gt))
        #generated_sdf_loss = torch.mean( F.l1_loss(generated_sdf_pred, gt, reduction='none') * surface_weight )

        # we did not experiment with using constants/weights for each loss (VAE loss is weighted using value in specs file)
        # results could potentially improve with a grid search 
        loss = sdf_loss + vae_loss + diff_loss + generated_sdf_loss

        return {
                        "loss": loss,
                        "sdf": sdf_loss,
                        "vae": vae_loss,
                        "diff": diff_loss,
                        # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
                        # typically diff_100 approaches 0 while diff_1000 can still be relatively high
                        # visualizing loss curves can help with debugging if training is unstable
                        #"diff100": diff_100_loss, # note that this can sometimes appear as nan when the training batch does not have sampled timesteps < 100
                        #"diff1000": diff_1000_loss,
                        "gensdf": generated_sdf_loss,
                    }


    def train_combined(self, x):

        losses = self.combined_losses(x)
        if losses is None:
            return None

        self.write_losses("train", losses, self.global_step)

        return losses["loss"]

from loss import FocalLoss
from collections import OrderedDict
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Projection, PatchMaker

import numpy as np
import pandas as pd
import torch.nn.functional as F

import logging
import os
import math
import torch
import tqdm
import common
import metrics
import cv2
import utils
import glob
import shutil

LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class GLASS(torch.nn.Module):
    def __init__(self, device):
        super(GLASS, self).__init__()
        self.device = device


    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension, # Convert output dimension from backbone model to pretrain_embed_dimension
            target_embed_dimension, # Conver all feature extracted from all layer target_embed_dimension
            patchsize=3, # Patchsize after pachify
            patchstride=1, # Stride
            meta_epochs=640,
            eval_epochs=1,
            dsc_layers=2, # Discriminator layer
            dsc_hidden=1024, # Discriminator hidden layer
            dsc_margin=0.5, # Discriminator threshold
            train_backbone=False,
            pre_proj=1,
            mining=1, # Using LAS or not
            noise=0.015,
            radius=0.75, # Quantile for mainfold or hyperpherse, 75% normal sample are selected
            p=0.5, # Quantile for hard mining 50% hard sameple are selected
            lr=0.0001,
            backbone_lr_ratio=0.1,
            backbone_train_start_epoch=0,
            svd=0,
            step=20, # Update Gaussian ascent 20 steps after 1 batch training
            limit=392,
            **kwargs,
    ):

        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape # [3, 288, 288]
        self.device = device

        self.forward_modules = torch.nn.ModuleDict({}) # Like dict()
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        ) # Read comment from fuction to understand
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape) # 
        self.forward_modules["feature_aggregator"] = feature_aggregator
        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.meta_epochs = meta_epochs
        self.lr = lr
        self.train_backbone = train_backbone
        self.backbone_lr_ratio = backbone_lr_ratio
        self.backbone_train_start_epoch = max(0, backbone_train_start_epoch)
        self.backbone_training_active = False
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(
                self.forward_modules["feature_aggregator"].backbone.parameters(),
                lr * self.backbone_lr_ratio,
            )
        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.Adam(self.pre_projection.parameters(), lr, weight_decay=1e-5)

        self.eval_epochs = eval_epochs
        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=lr * 2)
        self.dsc_margin = dsc_margin

        self.c = torch.tensor(0)
        self.c_ = torch.tensor(0)
        self.p = p
        self.radius = radius
        self.mining = mining
        self.noise = noise
        self.svd = svd
        self.step = step
        self.limit = limit
        self.distribution = 0
        self.focal_loss = FocalLoss()

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.anomaly_segmentor = common.RescaleSegmentor(device=self.device, target_size=input_shape[-2:])
        self.model_dir = ""
        self.dataset_name = ""
        self.logger = None
        
        # Load state dict
        # checkpoint = torch.load('/home/phatnguyen/Documents/repo/base-glass/results/models/backbone_0/mvtec_24_ad_s/', map_location=device)

        # self.discriminator.load_state_dict(checkpoint["discriminator"])

        # self.pre_projection.load_state_dict(checkpoint["pre_projection"])

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)

    def _should_train_backbone(self):
        return self.train_backbone and self.backbone_training_active

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""
        if not evaluation and self._should_train_backbone():
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from] # [n_layer, B, L, C]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2) # [n_layer, B, C, w, h]

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features] # [n_layer, n_patch_w, n_patch_h]
        patch_features = [x[0] for x in features] # [n_layer, B, N_patches, C, patchsize, patchsize]
        ref_num_patches = patch_shapes[0] # [n_patch_w, n_patch_h]

        for i in range(1, len(patch_features)):
            _features = patch_features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            ) # (B, n_patch_w, n_patch_h, C, patchsize, patchsize)
            _features = _features.permute(0, 3, 4, 5, 1, 2) # (B, C, patchsize, patchsize, n_patch_w, n_patch_h)
            perm_base_shape = _features.shape # (B, C, patchsize, patchsize, n_patch_w, n_patch_h)
            _features = _features.reshape(-1, *_features.shape[-2:]) # (BxCxpatchsizexpatchsize, n_patch_w, n_patch_h)
            _features = F.interpolate(
                _features.unsqueeze(1), # (BxCxpatchsizexpatchsize, 1, n_patch_w, n_patch_h)
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            ) # (BxCxpatchsizexpatchsize, 1, ref_w, ref_w)
            _features = _features.squeeze(1) # (BxCxpatchsizexpatchsize, ref_w, n_ref_wpatch_h)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            ) # (B, C, patchsize, patchsize, ref_w, ref_h)
            _features = _features.permute(0, 4, 5, 1, 2, 3) # (B, ref_w, ref_h, C, patchsize, patchsize)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:]) # (B, ref_wxref_h, C, patchsize, patchsize)
            patch_features[i] = _features
        # patch_features: (n_layer, B, ref_wxref_h, C, patchsize, patchsize)
        patch_features = [x.reshape(-1, *x.shape[-3:]) for x in patch_features] # (n_layer, Bxref_wxref_h, C, patchsize, patchsize)
        patch_features = self.forward_modules["preprocessing"](patch_features) # (B × ref_w × ref_h , n_layer , preprocessing_dim)
        patch_features = self.forward_modules["preadapt_aggregator"](patch_features) # (B × ref_w × ref_h , target_dim)

        return patch_features, patch_shapes

    def trainer(self, training_data, val_data, name):
        state_dict = {}
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        ckpt_path_save = os.path.join(self.ckpt_dir, "ckpt.pth")
        if len(ckpt_path) != 0:
            LOGGER.info("Start testing, ckpt file found!")
            return 0., 0., 0., 0., 0., -1.

        def update_state_dict():
            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.pre_projection.state_dict().items()})
            if self.train_backbone:
                state_dict["backbone"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.backbone.state_dict().items()})

        self.distribution = training_data.dataset.distribution
        xlsx_path = './datasets/excel/' + name.split('_')[0] + '_distribution.xlsx'
        try:
            if self.distribution == 1:  # rejudge by image-level spectrogram analysis
                self.distribution = 1
                self.svd = 1
            elif self.distribution == 2:  # manifold
                self.distribution = 0
                self.svd = 0
            elif self.distribution == 3:  # hypersphere
                self.distribution = 0
                self.svd = 1
            elif self.distribution == 4:  # opposite choose by file
                self.distribution = 0
                df = pd.read_excel(xlsx_path)
                self.svd = 1 - df.loc[df['Class'] == name, 'Distribution'].values[0]
            else:  # choose by file
                self.distribution = 0
                df = pd.read_excel(xlsx_path)
                self.svd = df.loc[df['Class'] == name, 'Distribution'].values[0]
        except:
            self.distribution = 1
            self.svd = 1

        # judge by image-level spectrogram analysis
        if self.distribution == 1:
            self.forward_modules.eval()
            with torch.no_grad():
                for i, data in enumerate(training_data):
                    img = data["image"]
                    img = img.to(torch.float).to(self.device)
                    batch_mean = torch.mean(img, dim=0)
                    if i == 0:
                        self.c = batch_mean
                    else:
                        self.c += batch_mean
                self.c /= len(training_data)

            avg_img = utils.torch_format_2_numpy_img(self.c.detach().cpu().numpy())
            self.svd = utils.distribution_judge(avg_img, name)
            os.makedirs(f'./results/judge/avg/{self.svd}', exist_ok=True)
            cv2.imwrite(f'./results/judge/avg/{self.svd}/{name}.png', avg_img)
            return self.svd

        pbar = tqdm.tqdm(range(self.meta_epochs), unit='epoch')
        pbar_str1 = ""
        best_record = None
        for i_epoch in pbar:
            self.backbone_training_active = self.train_backbone and i_epoch >= self.backbone_train_start_epoch
            self.forward_modules.eval()
            with torch.no_grad():  # compute center
                for i, data in enumerate(training_data):
                    img = data["image"] # (B, C, H, W)
                    img = img.to(torch.float).to(self.device)
                    if self.pre_proj > 0:
                        outputs = self.pre_projection(self._embed(img, evaluation=True)[0])
                        outputs = outputs[0] if len(outputs) == 2 else outputs # (B × ref_w × ref_h , target_dim)

                    else:
                        outputs = self._embed(img, evaluation=True)[0]  # (B × ref_w × ref_h , target_dim)
                    outputs = outputs[0] if len(outputs) == 2 else outputs
                    outputs = outputs.reshape(img.shape[0], -1, outputs.shape[-1]) # (B, ref_w × ref_h ,target_dim)

                    batch_mean = torch.mean(outputs, dim=0) # (ref_w × ref_h ,target_dim)
                    if i == 0:
                        self.c = batch_mean # (ref_w × ref_h ,target_dim)
                    else:
                        self.c += batch_mean
                self.c /= len(training_data)

            pbar_str, pt, pf = self._train_discriminator(training_data, i_epoch, pbar, pbar_str1)
            update_state_dict()

            if (i_epoch + 1) % self.eval_epochs == 0:
                images, scores, segmentations, labels_gt, masks_gt, image_paths = self.predict(val_data)
                image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, tpr, tnr = self._evaluate(images, scores, segmentations,
                                                                                         labels_gt, masks_gt, name)

                self.logger.logger.add_scalar("i-auroc", image_auroc, i_epoch)
                self.logger.logger.add_scalar("i-ap", image_ap, i_epoch)
                self.logger.logger.add_scalar("p-auroc", pixel_auroc, i_epoch)
                self.logger.logger.add_scalar("p-ap", pixel_ap, i_epoch)
                self.logger.logger.add_scalar("p-pro", pixel_pro, i_epoch)

                eval_path = './results/eval/' + name + '/'
                train_path = './results/training/' + name + '/'
                if best_record is None:
                    best_record = [image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, i_epoch, tpr, tnr]
                    ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best)
                    shutil.rmtree(eval_path, ignore_errors=True)
                    shutil.copytree(train_path, eval_path)

                # elif image_auroc + pixel_auroc > best_record[0] + best_record[2]:
                # elif image_auroc  > best_record[0]:
                elif tpr + tnr > best_record[-2] + best_record[-1]:
                    best_record = [image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, i_epoch, tpr, tnr]
                    os.remove(ckpt_path_best)
                    ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best)
                    shutil.rmtree(eval_path, ignore_errors=True)
                    shutil.copytree(train_path, eval_path)

                pbar_str1 = f" IAUC:{round(image_auroc * 100, 2)}({round(best_record[0] * 100, 2)})" \
                            f" IAP:{round(image_ap * 100, 2)}({round(best_record[1] * 100, 2)})" \
                            f" PAUC:{round(pixel_auroc * 100, 2)}({round(best_record[2] * 100, 2)})" \
                            f" PAP:{round(pixel_ap * 100, 2)}({round(best_record[3] * 100, 2)})" \
                            f" PRO:{round(pixel_pro * 100, 2)}({round(best_record[4] * 100, 2)})" \
                            f" E:{i_epoch}({best_record[-1]})"
                pbar_str += pbar_str1
                pbar.set_description_str(pbar_str)

            torch.save(state_dict, ckpt_path_save)
        return best_record

    def _train_discriminator(self, input_data, cur_epoch, pbar, pbar_str1):
        self.backbone_training_active = self.train_backbone and cur_epoch >= self.backbone_train_start_epoch
        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()

        all_loss, all_p_true, all_p_fake, all_r_t, all_r_g, all_r_f = [], [], [], [], [], []
        sample_num = 0
        for i_iter, data_item in enumerate(input_data):
            self.dsc_opt.zero_grad()
            if self.pre_proj > 0:
                self.proj_opt.zero_grad()
            if self._should_train_backbone():
                self.backbone_opt.zero_grad()

            aug = data_item["aug"]
            aug = aug.to(torch.float).to(self.device)
            img = data_item["image"]
            img = img.to(torch.float).to(self.device)
            if self.pre_proj > 0:
                fake_feats = self.pre_projection(self._embed(aug, evaluation=False)[0])
                fake_feats = fake_feats[0] if len(fake_feats) == 2 else fake_feats # (B × ref_w × ref_h , target_dim)
                true_feats = self.pre_projection(self._embed(img, evaluation=False)[0])
                true_feats = true_feats[0] if len(true_feats) == 2 else true_feats # (B × ref_w × ref_h , target_dim)
            else:
                fake_feats = self._embed(aug, evaluation=False)[0]
                fake_feats.requires_grad = True
                true_feats = self._embed(img, evaluation=False)[0]
                true_feats.requires_grad = True

            mask_s_gt = data_item["mask_s"].reshape(-1, 1).to(self.device) # [B x feat_size x feat_size, 1]
            noise = torch.normal(0, self.noise, true_feats.shape).to(self.device) # (B × ref_w × ref_h , target_dim)
            gaus_feats = true_feats + noise

            center = self.c.repeat(img.shape[0], 1, 1) # (B, ref_w × ref_h ,target_dim)
            center = center.reshape(-1, center.shape[-1]) # (B x ref_w × ref_h ,target_dim)
            true_points = torch.concat([fake_feats[mask_s_gt[:, 0] == 0], true_feats], dim=0) # (K + B x ref_w × ref_h ,target_dim); K < B x ref_w × ref_h
            c_t_points = torch.concat([center[mask_s_gt[:, 0] == 0], center], dim=0)  # (K + B x ref_w × ref_h ,target_dim); K < B x ref_w × ref_h
            dist_t = torch.norm(true_points - c_t_points, dim=1) # (K + B x ref_w × ref_h)
            r_t = torch.tensor([torch.quantile(dist_t, q=self.radius)]).to(self.device) # shape = 1

            for step in range(self.step + 1):
                scores = self.discriminator(torch.cat([true_feats, gaus_feats])) # (2 x B × ref_w × ref_h, 1)
                true_scores = scores[:len(true_feats)] # (B × ref_w × ref_h, 1)
                gaus_scores = scores[len(true_feats):] # (B × ref_w × ref_h, 1)
                true_loss = torch.nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
                gaus_loss = torch.nn.BCELoss()(gaus_scores, torch.ones_like(gaus_scores))
                bce_loss = true_loss + gaus_loss

                if step == self.step:
                    break
                elif self.mining == 0:
                    dist_g = torch.norm(gaus_feats - center, dim=1) # (B × ref_w × ref_h)
                    r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device) # shape = 1
                    break

                grad = torch.autograd.grad(gaus_loss, [gaus_feats])[0] # (B × ref_w × ref_h ], target_dim)
                grad_norm = torch.norm(grad, dim=1) # (B × ref_w × ref_h)
                grad_norm = grad_norm.view(-1, 1) # (B × ref_w × ref_h, 1)
                grad_normalized = grad / (grad_norm + 1e-10) # (B × ref_w × ref_h , target_dim)

                with torch.no_grad():
                    gaus_feats.add_(0.001 * grad_normalized) # add all element on tensor

                if (step + 1) % 5 == 0:
                    dist_g = torch.norm(gaus_feats - center, dim=1) # (B × ref_w × ref_h)
                    r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device) # shape = 1
                    proj_feats = center if self.svd == 1 else true_feats # (B x ref_w × ref_h ,target_dim)
                    r = r_t if self.svd == 1 else 0.5 # shape = 1

                    h = gaus_feats - proj_feats # (B x ref_w × ref_h, target_dim)
                    h_norm = dist_g if self.svd == 1 else torch.norm(h, dim=1) # (B × ref_w × ref_h)
                    alpha = torch.clamp(h_norm, r, 2 * r) # (B × ref_w × ref_h)
                    proj = (alpha / (h_norm + 1e-10)).view(-1, 1) # (B × ref_w × ref_h, 1)
                    h = proj * h # (B × ref_w × ref_h, target_dim)
                    gaus_feats = proj_feats + h # (B × ref_w × ref_h, target_dim)

            fake_points = fake_feats[mask_s_gt[:, 0] == 1] # (K, target_dim) ; K < B x ref_w × ref_h
            true_points = true_feats[mask_s_gt[:, 0] == 1] # (K, target_dim) ; K < B x ref_w × ref_h
            c_f_points = center[mask_s_gt[:, 0] == 1] # (K, target_dim) ; K < B x ref_w × ref_h 
            dist_f = torch.norm(fake_points - c_f_points, dim=1) # (K)
            r_f = torch.tensor([torch.quantile(dist_f, q=self.radius)]).to(self.device) # 1
            proj_feats = c_f_points if self.svd == 1 else true_points # (K, target_dim)
            r = r_t if self.svd == 1 else 1 # shape = 1

            if self.svd == 1:
                h = fake_points - proj_feats # (K, target_dim)
                h_norm = dist_f if self.svd == 1 else torch.norm(h, dim=1) # (K)
                alpha = torch.clamp(h_norm, 2 * r, 4 * r) # (K)
                proj = (alpha / (h_norm + 1e-10)).view(-1, 1) # (K, 1)
                h = proj * h # (K, target_dim)
                fake_points = proj_feats + h # (K, target_dim)
                fake_feats[mask_s_gt[:, 0] == 1] = fake_points # (B × ref_w × ref_h, target_dim)

            fake_scores = self.discriminator(fake_feats) # (B × ref_w × ref_h, 1)
            if self.p > 0:
                fake_dist = (fake_scores - mask_s_gt) ** 2  # (B × ref_w × ref_h, 1)
                d_hard = torch.quantile(fake_dist, q=self.p) # (1)
                fake_scores_ = fake_scores[fake_dist >= d_hard].unsqueeze(1) # (K , 1)
                mask_ = mask_s_gt[fake_dist >= d_hard].unsqueeze(1) # (K , 1)
            else:
                fake_scores_ = fake_scores
                mask_ = mask_s_gt # (B × ref_w × ref_h, 1)
            output = torch.cat([1 - fake_scores_, fake_scores_], dim=1) # (K , 2)
            focal_loss = self.focal_loss(output, mask_)

            loss = bce_loss + focal_loss
            loss.backward()
            if self.pre_proj > 0:
                self.proj_opt.step()
            if self._should_train_backbone():
                self.backbone_opt.step()
            self.dsc_opt.step()

            pix_true = torch.concat([fake_scores.detach() * (1 - mask_s_gt), true_scores.detach()])
            pix_fake = torch.concat([fake_scores.detach() * mask_s_gt, gaus_scores.detach()])
            p_true = ((pix_true < self.dsc_margin).sum() - (pix_true == 0).sum()) / ((mask_s_gt == 0).sum() + true_scores.shape[0])
            p_fake = (pix_fake >= self.dsc_margin).sum() / ((mask_s_gt == 1).sum() + gaus_scores.shape[0])

            self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
            self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_t", r_t, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_g", r_g, self.logger.g_iter)
            self.logger.logger.add_scalar(f"r_f", r_f, self.logger.g_iter)
            self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
            self.logger.step()

            all_loss.append(loss.detach().cpu().item())
            all_p_true.append(p_true.cpu().item())
            all_p_fake.append(p_fake.cpu().item())
            all_r_t.append(r_t.cpu().item())
            all_r_g.append(r_g.cpu().item())
            all_r_f.append(r_f.cpu().item())

            all_loss_ = np.mean(all_loss)
            all_p_true_ = np.mean(all_p_true)
            all_p_fake_ = np.mean(all_p_fake)
            all_r_t_ = np.mean(all_r_t)
            all_r_g_ = np.mean(all_r_g)
            all_r_f_ = np.mean(all_r_f)
            sample_num = sample_num + img.shape[0]

            pbar_str = f"epoch:{cur_epoch} loss:{all_loss_:.2e}"
            pbar_str += f" pt:{all_p_true_ * 100:.2f}"
            pbar_str += f" pf:{all_p_fake_ * 100:.2f}"
            pbar_str += f" rt:{all_r_t_:.2f}"
            pbar_str += f" rg:{all_r_g_:.2f}"
            pbar_str += f" rf:{all_r_f_:.2f}"
            pbar_str += f" svd:{self.svd}"
            pbar_str += f" sample:{sample_num}"
            pbar_str2 = pbar_str
            pbar_str += pbar_str1
            pbar.set_description_str(pbar_str)

            if sample_num > self.limit:
                break

        return pbar_str2, all_p_true_, all_p_fake_

    def compute_metrics(self, scores, labels_gt, threshold=0.45):
        scores = np.array(scores)
        labels_gt = np.array(labels_gt)

        # Convert scores to binary predictions using fixed threshold = 0.5
        predictions = (scores > threshold).astype(int)

        # Confusion matrix components
        tp = np.sum((predictions == 1) & (labels_gt == 1))
        fp = np.sum((predictions == 1) & (labels_gt == 0))
        tn = np.sum((predictions == 0) & (labels_gt == 0))
        fn = np.sum((predictions == 0) & (labels_gt == 1))

        # Metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0      # Recall / Sensitivity
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0      # Specificity
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        accuracy = (tp + tn) / len(labels_gt) if len(labels_gt) > 0 else 0.0

        return {
            "threshold": threshold,
            "tpr": tpr,
            "tnr": tnr,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        }
        
    def tester(self, test_data, name):
        test_dict = {}
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        if len(ckpt_path) != 0:
            state_dict = torch.load(ckpt_path[0], map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
                if "backbone" in state_dict:
                    self.backbone.load_state_dict(state_dict["backbone"])
            else:
                self.load_state_dict(state_dict, strict=False)

            images, scores, segmentations, labels_gt, masks_gt, image_paths = self.predict(test_data)
            for i in range(len(images)):
                test_dict[i] = [str(i + 1).zfill(3),scores[i].tolist(), labels_gt[i], image_paths[i]]
            import json
            with open(os.path.join(self.ckpt_dir, "test_scores.json"), 'w') as f:
                json.dump(test_dict, f)
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, tpr, tnr = self._evaluate(images, scores, segmentations,
                                                                                     labels_gt, masks_gt, name, path='eval')
            epoch = int(ckpt_path[0].split('_')[-1].split('.')[0])
        else:
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch = 0., 0., 0., 0., 0., -1.
            LOGGER.info("No ckpt file found!")

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch

    def _evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, path='training'):
        scores = np.squeeze(np.array(scores))
        image_scores = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt, path)
        image_auroc = image_scores["auroc"]
        image_ap = image_scores["ap"]

        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(segmentations, masks_gt, path)
            pixel_auroc = pixel_scores["auroc"]
            pixel_ap = pixel_scores["ap"]
            if path == 'eval':
                try:
                    pixel_pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), segmentations)
                except:
                    pixel_pro = 0.
            else:
                pixel_pro = 0.
        else:
            pixel_auroc = -1.
            pixel_ap = -1.
            pixel_pro = -1.
            return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

        results = self.compute_metrics(scores=scores, labels_gt=labels_gt)
        LOGGER.info(
            f"TPR: {results['tpr']*100:.2f}%, "
            f"TNR: {results['tnr']*100:.2f}%, "
            f"Accuracy: {results['accuracy']*100:.2f}%, "
        )
        
        defects = np.array(images)
        targets = np.array(masks_gt)
        for i in range(len(defects)):
            defect = utils.torch_format_2_numpy_img(defects[i])
            target = utils.torch_format_2_numpy_img(targets[i])

            mask = cv2.cvtColor(cv2.resize(segmentations[i], (defect.shape[1], defect.shape[0])),
                                cv2.COLOR_GRAY2BGR)
            mask = (mask * 255).astype('uint8')
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            img_up = np.hstack([defect, target, mask])
            img_up = cv2.resize(img_up, (256 * 3, 256))
            full_path = './results/' + path + '/' + name + '/'
            utils.del_remake_dir(full_path, del_flag=False)
            cv2.imwrite(full_path + str(i + 1).zfill(3) + '.png', img_up)

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, results['tpr'], results['tnr']

    def predict(self, test_dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        self.forward_modules.eval()

        img_paths = []
        images = []
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []

        with tqdm.tqdm(test_dataloader, desc="Inferring...", leave=False, unit='batch') as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask_gt", None) is not None:
                        masks_gt.extend(data["mask_gt"].numpy().tolist())
                    image = data["image"]
                    images.extend(image.numpy().tolist())
                    img_paths.extend(data["image_path"])
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)

        return images, scores, masks, labels_gt, masks_gt, img_paths

    def _predict(self, img):
        """Infer score and mask for a batch of images."""
        img = img.to(torch.float).to(self.device)
        self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()

        with torch.no_grad():

            patch_features, patch_shapes = self._embed(img, provide_patch_shapes=True, evaluation=True)
            if self.pre_proj > 0:
                patch_features = self.pre_projection(patch_features)
                patch_features = patch_features[0] if len(patch_features) == 2 else patch_features

            patch_scores = image_scores = self.discriminator(patch_features)
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=img.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(img.shape[0], scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=img.shape[0])
            image_scores = self.patch_maker.score(image_scores)
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

        return list(image_scores), list(masks)

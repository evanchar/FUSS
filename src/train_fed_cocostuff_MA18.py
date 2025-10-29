from utils import *
from modules import *
from data import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
import torch.multiprocessing
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import copy

from sklearn.cluster import KMeans
import sys
import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# todo: what is this? could this break results? Commented it out for now -->
#  I am finding that this changes results a bit in the non-crop scenario that I was working on
torch.set_float32_matmul_precision('medium')
torch.multiprocessing.set_sharing_strategy('file_system')


def get_class_labels(dataset_name):
    if dataset_name.startswith("cityscapes"):
        return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'caravan', 'trailer', 'train',
            'motorcycle', 'bicycle']
    elif dataset_name == "cocostuff27":
        return [
            "electronic", "appliance", "food", "furniture", "indoor",
            "kitchen", "accessory", "animal", "outdoor", "person",
            "sports", "vehicle", "ceiling", "floor", "food",
            "furniture", "rawmaterial", "textile", "wall", "window",
            "building", "ground", "plant", "sky", "solid",
            "structural", "water"]
    elif dataset_name == "voc":
        return [
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == "potsdam":
        return [
            'roads and cars',
            'buildings and clutter',
            'trees and vegetation']
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))


class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        if not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        data_dir = join(cfg.output_root, "data")
        if cfg.arch == "feature-pyramid":
            cut_model = load_model(cfg.model_type, data_dir).cuda()
            self.net = FeaturePyramidNet(cfg.granularity, cut_model, dim, cfg.continuous)
        elif cfg.arch == "dino":
            self.net = DinoFeaturizer(dim, cfg)
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))

        self.train_cluster_probe = ClusterLookup(dim, n_classes)

        self.cluster_probe = ClusterLookup(dim, n_classes + cfg.extra_clusters)
        self.cluster_list = []
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))

        self.decoder = nn.Conv2d(dim, self.net.n_feats, (1, 1))

        self.cluster_metrics = UnsupervisedMetrics(
            "test/cluster/", n_classes, cfg.extra_clusters, True)
        self.linear_metrics = UnsupervisedMetrics(
            "test/linear/", n_classes, 0, False)

        self.test_cluster_metrics = UnsupervisedMetrics(
            "final/cluster/", n_classes, cfg.extra_clusters, True)
        self.test_linear_metrics = UnsupervisedMetrics(
            "final/linear/", n_classes, 0, False)

        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss()
        self.crf_loss_fn = ContrastiveCRFLoss(
            cfg.crf_samples, cfg.alpha, cfg.beta, cfg.gamma, cfg.w1, cfg.w2, cfg.shift)

        self.contrastive_corr_loss_fn = ContrastiveCorrelationLoss(cfg)
        for p in self.contrastive_corr_loss_fn.parameters():
            p.requires_grad = False

        # todo: this is to choose students that will benefit from model aggregation.
        #  Use it in val step
        # self.contrastive_corr_loss_fn_val = ContrastiveCorrelationLossVal(cfg)
        # for p in self.contrastive_corr_loss_fn_val.parameters():
        #     p.requires_grad = False
        self.val_dists = []
        self.val_dists_pos_intra = []
        self.val_dists_pos_inter = []

        self.automatic_optimization = False

        if self.cfg.dataset_name1.startswith("cityscapes"):
            self.label_cmap = create_cityscapes_colormap()
        else:
            self.label_cmap = create_pascal_label_colormap()

        self.outputs = []
        self.val_steps = 0
        self.save_hyperparameters()

        # todo: this is the global model cluster2 module. It will be updated after each aggregation and
        #  stay frozen to be used with the FedProx and FedMOON loses

        self.global_model = copy.deepcopy(self.net.cluster2)
        self.prev_local_model = copy.deepcopy(self.net.cluster2)  # this is only for initialization for moon. It will be changed after first local training.


    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)[1]

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        assert self.training, "***Model is not in training mode!!!"
        net_optim, linear_probe_optim, cluster_probe_optim = self.optimizers()

        net_optim.zero_grad()
        linear_probe_optim.zero_grad()
        cluster_probe_optim.zero_grad()

        with torch.no_grad():
            ind = batch["ind"]
            img = batch["img"]
            img_aug = batch["img_aug"]
            coord_aug = batch["coord_aug"]
            img_pos = batch["img_pos"]
            label = batch["label"]
            label_pos = batch["label_pos"]

        feats, code = self.net(img)
        if self.cfg.correspondence_weight > 0:
            feats_pos, code_pos = self.net(img_pos)
        log_args = dict(sync_dist=False, rank_zero_only=True)

        if self.cfg.use_true_labels:
            signal = one_hot_feats(label + 1, self.n_classes + 1)
            signal_pos = one_hot_feats(label_pos + 1, self.n_classes + 1)
        else:
            signal = feats
            signal_pos = feats_pos

        loss = 0

        should_log_hist = (self.cfg.hist_freq is not None) and \
                          (self.global_step % self.cfg.hist_freq == 0) and \
                          (self.global_step > 0)
        if self.cfg.use_salience:
            salience = batch["mask"].to(torch.float32).squeeze(1)
            salience_pos = batch["mask_pos"].to(torch.float32).squeeze(1)
        else:
            salience = None
            salience_pos = None

        if self.cfg.correspondence_weight > 0:
            (
                pos_intra_loss, pos_intra_cd,
                pos_inter_loss, pos_inter_cd,
                neg_inter_loss, neg_inter_cd,
            ) = self.contrastive_corr_loss_fn(
                signal, signal_pos,
                salience, salience_pos,
                code, code_pos,
            )

            if should_log_hist:
                self.logger.experiment.add_histogram("intra_cd", pos_intra_cd, self.global_step)
                self.logger.experiment.add_histogram("inter_cd", pos_inter_cd, self.global_step)
                self.logger.experiment.add_histogram("neg_cd", neg_inter_cd, self.global_step)
            neg_inter_loss = neg_inter_loss.mean()
            pos_intra_loss = pos_intra_loss.mean()
            pos_inter_loss = pos_inter_loss.mean()
            self.log('loss/pos_intra', pos_intra_loss, **log_args)
            self.log('loss/pos_inter', pos_inter_loss, **log_args)
            self.log('loss/neg_inter', neg_inter_loss, **log_args)
            self.log('cd/pos_intra', pos_intra_cd.mean(), **log_args)
            self.log('cd/pos_inter', pos_inter_cd.mean(), **log_args)
            self.log('cd/neg_inter', neg_inter_cd.mean(), **log_args)

            loss += (self.cfg.pos_inter_weight * pos_inter_loss +
                     self.cfg.pos_intra_weight * pos_intra_loss +
                     self.cfg.neg_inter_weight * neg_inter_loss) * self.cfg.correspondence_weight
        # print(f"Loss for backward of STEGO batch {batch_idx}: {loss}")
        if self.cfg.rec_weight > 0:
            rec_feats = self.decoder(code)
            rec_loss = -(norm(rec_feats) * norm(feats)).sum(1).mean()
            self.log('loss/rec', rec_loss, **log_args)
            loss += self.cfg.rec_weight * rec_loss

        if self.cfg.aug_alignment_weight > 0:
            orig_feats_aug, orig_code_aug = self.net(img_aug)
            downsampled_coord_aug = resize(
                coord_aug.permute(0, 3, 1, 2),
                orig_code_aug.shape[2]).permute(0, 2, 3, 1)
            aug_alignment = -torch.einsum(
                "bkhw,bkhw->bhw",
                norm(sample(code, downsampled_coord_aug)),
                norm(orig_code_aug)
            ).mean()
            self.log('loss/aug_alignment', aug_alignment, **log_args)
            loss += self.cfg.aug_alignment_weight * aug_alignment

        if self.cfg.crf_weight > 0:
            crf = self.crf_loss_fn(
                resize(img, 56),
                norm(resize(code, 56))
            ).mean()
            self.log('loss/crf', crf, **log_args)
            loss += self.cfg.crf_weight * crf

        flat_label = label.reshape(-1)
        mask = (flat_label >= 0) & (flat_label < self.n_classes)

        detached_code = torch.clone(code.detach())

        linear_logits = self.linear_probe(detached_code)
        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes)
        linear_loss = self.linear_probe_loss_fn(linear_logits[mask], flat_label[mask]).mean()
        loss += linear_loss
        self.log('loss/linear', linear_loss, **log_args)

        cluster_loss, cluster_probs = self.cluster_probe(detached_code, None)
        loss += cluster_loss

        self.log('loss/cluster', cluster_loss, **log_args)
        self.log('loss/total', loss, **log_args)

        # todo: new for FedProx proximal term only (partial updates of struggler do not apply to us
        #  since we assume same system resources for each client)
        if self.cfg.fedprox:
            proximal_term = 0.0
            mu = self.cfg.proximal_mu  # set in your config, e.g., 0.001

            for local_param, global_param in zip(self.net.cluster2.parameters(), self.global_model.parameters()):
                proximal_term += ((local_param - global_param.detach()) ** 2).sum()
            proximal_term *= (mu / 2)
            loss += proximal_term

        # todo: new for MOON local contrastive loss
        if self.cfg.fedmoon:
            temperature = self.cfg.moon_temperature
            moon_mu = self.cfg.moon_weight

            # Forward pass to get representations
            # Assume: self.representation_fn(model, x) gives you z = R_w(x)
            z = code
            z_glob = self.global_model(feats)
            z_prev = self.prev_local_model(feats)

            # Normalize if needed
            z = F.normalize(z, dim=1)
            z_glob = F.normalize(z_glob, dim=1)
            z_prev = F.normalize(z_prev, dim=1)

            # Compute cosine similarities
            sim_z_zglob = F.cosine_similarity(z, z_glob, dim=1) / temperature
            sim_z_zprev = F.cosine_similarity(z, z_prev, dim=1) / temperature

            # Contrastive loss (batch-wise)
            moon_loss = -torch.log(
                torch.exp(sim_z_zglob) /
                (torch.exp(sim_z_zglob) + torch.exp(sim_z_zprev) + 1e-8)
            )
            moon_loss = moon_loss.mean()

            loss += moon_mu * moon_loss

        self.manual_backward(loss)

        net_optim.step()
        cluster_probe_optim.step()
        linear_probe_optim.step()
        new_clusters = cluster_probe_optim.optimizer.param_groups[0]['params'][0].clone().detach()
        self.cluster_list.append(new_clusters)

        if self.cfg.reset_probe_steps is not None and self.global_step == self.cfg.reset_probe_steps:
            print("RESETTING PROBES")
            self.linear_probe.reset_parameters()
            self.cluster_probe.reset_parameters()
            self.trainer.optimizers[1] = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
            self.trainer.optimizers[2] = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)

        if self.global_step % 2000 == 0 and self.global_step > 0:
            print("RESETTING TFEVENT FILE")
            # Make a new tfevent file
            self.logger.experiment.close()
            self.logger.experiment._get_file_writer()

        return loss

    def on_train_start(self):
        tb_metrics = {
            **self.linear_metrics.compute(),
            **self.cluster_metrics.compute()
        }
        self.logger.log_hyperparams(self.cfg, tb_metrics)

    def validation_step(self, batch, batch_idx):

        img = batch["img"]
        label = batch["label"]
        self.net.eval()

        with torch.no_grad():
            feats, code = self.net(img)

            with torch.no_grad():
                img = batch["img"]
                img_pos = batch["img_pos"]
                label = batch["label"]

            if self.cfg.correspondence_weight > 0:
                feats_pos, code_pos = self.net(img_pos)

            signal = feats
            signal_pos = feats_pos
            salience = None
            salience_pos = None

            loss = 0

            (
                pos_intra_loss, pos_intra_cd,
                pos_inter_loss, pos_inter_cd,
                neg_inter_loss, neg_inter_cd
                # ) = self.contrastive_corr_loss_fn_val(
            ) = self.contrastive_corr_loss_fn(
                signal, signal_pos,
                salience, salience_pos,
                code, code_pos,
            )
            neg_inter_loss = neg_inter_loss.mean()
            pos_intra_loss = pos_intra_loss.mean()
            pos_inter_loss = pos_inter_loss.mean()

            loss += (self.cfg.pos_inter_weight * pos_inter_loss +
                     self.cfg.pos_intra_weight * pos_intra_loss +
                     self.cfg.neg_inter_weight * neg_inter_loss) * self.cfg.correspondence_weight
            # print(pos_inter_loss, pos_intra_loss, neg_inter_loss.size())
            # todo - three lines bellow do the following :

            self.val_dists.append(loss)  # only dist_inter (similar image) for debugging

            code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

            linear_preds = self.linear_probe(code)
            linear_preds = linear_preds.argmax(1)
            self.linear_metrics.update(linear_preds, label)

            cluster_loss, cluster_preds = self.cluster_probe(code, None)
            cluster_preds = cluster_preds.argmax(1)
            self.cluster_metrics.update(cluster_preds, label)

            self.outputs.append({
                'img': img[:self.cfg.n_images].detach().cpu(),
                'linear_preds': linear_preds[:self.cfg.n_images].detach().cpu(),
                "cluster_preds": cluster_preds[:self.cfg.n_images].detach().cpu(),
                "label": label[:self.cfg.n_images].detach().cpu(),
                # "dist_val": torch.mean(self.val_dists.detach().cpu())
            })  # mean() ?

    def on_validation_epoch_end(self) -> None:
        with torch.no_grad():
            tb_metrics = {
                **self.linear_metrics.compute(),
                **self.cluster_metrics.compute(),
            }
            if self.global_step > 2:
                self.log_dict(tb_metrics)
                if self.trainer.is_global_zero and self.cfg.azureml_logging:
                    from azureml.core.run import Run
                    run_logger = Run.get_context()
                    for metric, value in tb_metrics.items():
                        run_logger.log(metric, value)

            self.linear_metrics.reset()
            self.cluster_metrics.reset()

        self.outputs.clear()

    def get_validation_loss(self):
        distances = self.val_dists
        mean_distance = sum(distances) / len(distances)
        self.val_dists = []

        return mean_distance, distances

    def configure_optimizers(self):
        main_params = list(self.net.parameters())

        if self.cfg.rec_weight > 0:
            main_params.extend(self.decoder.parameters())

        net_optim = torch.optim.Adam(main_params, lr=self.cfg.lr)
        linear_probe_optim = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)

        return net_optim, linear_probe_optim, cluster_probe_optim


@hydra.main(config_path="configs", config_name="train_config_cocostuff_MA18.yaml")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)

    # todo: addition to include competitor names in paths
    if cfg.fedprox:
        cfg.experiment_name = 'exp_base_fedprox'
    elif cfg.fedmoon:
        cfg.experiment_name = 'exp_base_fedmoon'

    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir
    # os.makedirs(cfg.save_dir, exist_ok=True) # not needed
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    checkpoint_dir = join(cfg.output_root, "checkpoints")

    prefix = "{}/{}_{}".format(cfg.log_dir, cfg.dataset_name1, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.full_name = prefix

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed_everything(seed=0)

    print(data_dir)
    print(cfg.output_root)

    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(0.8, 1.0))
    ])
    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        T.RandomGrayscale(.2),
        T.RandomApply([T.GaussianBlur((5, 5))])
    ])

    sys.stdout.flush()

    train_datasets = []
    train_loaders = []

    agg_type_path = f"../checkpoints/cocostuff/{cfg.aggregation_type}_{name}_/"
    if not os.path.exists(agg_type_path):
        os.makedirs(agg_type_path)  # Creates the directory (including parent dirs if needed)
        print(f"Directory '{agg_type_path}' created.")

    if cfg.client_num == 3:
        my_pre = "MA3"
    elif cfg.client_num == 6:
        my_pre = "MA6"
    elif cfg.client_num == 18:
        my_pre = "MA18"

    for i in range(cfg.client_num):
        train_datasets.append(ContrastiveSegDataset(
            pytorch_data_dir=pytorch_data_dir,
            dataset_name=f"{my_pre}_{cfg.dataset_name}{i + 1}",
            crop_type=cfg.crop_type,
            image_set="train",
            transform=get_transform(cfg.res, False, cfg.loader_crop_type),
            target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
            cfg=cfg,
            aug_geometric_transform=geometric_transforms,
            aug_photometric_transform=photometric_transforms,
            num_neighbors=cfg.num_neighbors,
            mask=True,
            pos_images=True,
            pos_labels=True
            ))
        train_loaders.append(
            DataLoader(train_datasets[i], cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True))
        print(f"client {i} set! Dataset length: {len(train_datasets[i])}")
    val_loader_crop = "center"
    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name_val,
        crop_type=None,
        image_set="val",
        transform=get_transform(320, False, val_loader_crop),
        target_transform=get_transform(320, True, val_loader_crop),
        mask=True,
        cfg=cfg,
        pos_images=True,  # todo: my additions
        pos_labels=True  # todo: my additions
    )

    if cfg.submitting_to_aml:
        val_batch_size = 16
    else:
        val_batch_size = cfg.batch_size

    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    default_model = LitUnsupervisedSegmenter(train_datasets[0].n_classes, cfg)
    models = []
    clusters = []
    for i in range(cfg.client_num):
        models.append(copy.deepcopy(default_model))

    tb_logger = TensorBoardLogger(
        join(log_dir, name),
        default_hp_metric=False
    )

    gpu_args = {
        "accelerator": "gpu",
        "devices": 1
        # "val_check_interval": 500  # removed it to save time in fed learning
    }

    for j in range(cfg.aggregation_num):
        # reinitialize trainers for re-fit after each aggregation and fit in the same loop
        trainers = []
        model_dist = []
        data_quantity = []
        agg_clusters = []
        for i in range(cfg.client_num):
            trainers.append(Trainer(
                log_every_n_steps=cfg.scalar_log_freq,
                logger=tb_logger,
                max_steps=cfg.max_steps,  # default - problem with global step!

                callbacks=[
                    ModelCheckpoint(
                        dirpath=join(checkpoint_dir, name),
                        save_top_k=1,
                        monitor="test/cluster/mIoU",
                        mode="max",
                    )
                ],
                strategy=DDPStrategy(find_unused_parameters=True),
                **gpu_args))

            models[i].global_model = copy.deepcopy(models[i].net.cluster2)  # global frozen model for fedprox and fedmoon
            trainers[i].fit(models[i], train_loaders[i])  # no val during training
            models[i].prev_local_model = copy.deepcopy(models[i].net.cluster2) # local frozen model for fedmoon


            torch.save(models[i].cluster_list, f'{agg_type_path}clusters_client{i}_agg{j}')
            agg_clusters.append(models[i].cluster_list)

            # todo: stopped val to check displacement replace
            trainers[i].validate(model=models[i], dataloaders=val_loader)  # manual val to get distances
            losses = trainers[i].callback_metrics['test/cluster/mIoU']
            model_dist.append(losses)  # losses[0] is get_validation_loss is used
            data_quantity.append(len(train_loaders[i]))

            # Reset epoch and global_step manually
            trainers[i].fit_loop.epoch_progress.current.completed = 0  # Reset epoch count
            trainers[i].fit_loop.epoch_loop._batches_that_stepped = 0  # Reset global_step

            # Optional: Reset other internal states if needed
            trainers[i].fit_loop.epoch_progress.current.processed = 0  # Reset processed epoch count

            print(f"\n --- Client {i} done with fitting! --- \n Trainers list len: {len(trainers)}")

        cluster_displacements = cluster_analysis(agg_clusters, j)
        with open(f"{agg_type_path}outputs.txt", "a") as f:
            for k in range(cfg.client_num):
                output_train = f"model{k} cluster displacement: {cluster_displacements[k]} \n model{k} val res: {model_dist[k]} \n"
                print(output_train)
                f.write(output_train)


            max_index_val = model_dist.index(
                max(model_dist))  # this is for GLOBAL alignments. min index means most negative (hence smaller) loss value, but miou shows the opposite. Best miou is max.. why?

            mean_val = sum(model_dist) / len(model_dist)
            max_index = np.argmin(cluster_displacements)
            output_displ = f"Best model displacement:  {max_index}\n"
            output_best = f"Best model val:  {max_index_val}\n"
            output_mean = f"mean model val: {mean_val}\n"
            print(output_displ)
            f.write(output_displ)
            print(output_best)
            f.write(output_best)
            print(output_mean)
            f.write(output_mean)

            agg_types = ['fedavg', 'custom_fedavg', 'weighted_fedavg', 'custom_weighted_fedavg',
                         'hierarchical', 'hierarchical_not_encoder', 'weighted_hierarchical',
                         'weighted_hierarchical_not_encoder',
                         'hierarchical_mm', 'hierarchical_mm_not_encoder', 'weighted_hierarchical_mm_not_encoder',
                         'weighted_hierarchical_mm',
                         'hierarchical_train', 'hierarchical_mm_km',
                         'encoder', 'not_encoder', "weighted_encoder", 'weighted_not_encoder']
            custom_agg_types = ['custom_fedavg', 'custom_weighted_fedavg']  # leave best without agg
            if cfg.aggregation_type in agg_types:
                model_to_distribute = models[0].__class__(train_datasets[0].n_classes, cfg)
                state_dicts = [model.state_dict() for model in models]
                average_state_dict = {}
                stacked_state_dict = {}
                weights = [1.0] * cfg.client_num
                if 'weighted' in cfg.aggregation_type:  # todo:experimental
                    print(f"WEIGHTED VARIANT: {cfg.aggregation_type}")
                    # if cfg.aggregation_type == 'weighted_fedavg' or cfg.aggregation_type == 'custom_weighted_fedavg' or cfg.aggregation_type == 'weighted_encoder':
                    # weights = model_dist
                    weights = data_quantity

                print(f"weights pre-norm: {weights}")
                f.write(f"weights pre-norm: {weights}")
                weights = torch.tensor(weights)  # weight list to tensor
                weights = weights / weights.sum()  # weight normalization

                print(f"weights: {weights}")
                f.write(f"weights: {weights}")
                for key in state_dicts[
                    0]:  # todo: could this be WEIGHTED average instead of simple mean??? especially in custom_fedavg
                    stacked_state_dict[key] = torch.stack(
                        [state_dict[key] * weights[k] for k, state_dict in enumerate(state_dicts)],
                        dim=0)
                    average_state_dict[key] = torch.stack(
                        [state_dict[key] * weights[k] for k, state_dict in enumerate(state_dicts)],
                        dim=0).sum(dim=0)

                # if cfg.aggregation_type[:12] == 'hierarchical':
                if 'hierarchical' in cfg.aggregation_type:
                    # todo: create hierarchical clustering here..
                    print("Initiating cluster gathering..")
                    device = agg_clusters[0][0].device
                    last_centroids = [entry[-1] for entry in agg_clusters]
                    stacked_centroids = torch.stack(last_centroids, dim=0).reshape((-1, cfg.dim))
                    centroids_np = stacked_centroids.cpu().numpy()  # Convert to NumPy (if on GPU)

                    # if cfg.aggregation_type == 'hierarchical' or cfg.aggregation_type == 'hierarchical_not_encoder':
                    if 'mm' not in cfg.aggregation_type:
                        # Perform KMeans clustering to obtain 27 clusters
                        kmeans = KMeans(n_clusters=27, random_state=42, n_init=10)
                        kmeans.fit(centroids_np)
                        # Get the new cluster centers
                        new_centroids_np = kmeans.cluster_centers_
                        # Convert back to a PyTorch tensor
                        new_centroids = torch.tensor(new_centroids_np, device=device, dtype=stacked_centroids.dtype)
                    else:
                        new_centroids = maximin_clustering(stacked_centroids, num_clusters=27)

                    average_state_dict['cluster_probe.clusters'] = new_centroids

                model_to_distribute.load_state_dict(average_state_dict)

                for i in range(cfg.client_num):
                    if cfg.aggregation_type in custom_agg_types and i == max_index:  # leave best without agg
                        print(f"skipping replacement of model {i} since it has best contrastive performace..")
                        continue
                    if 'encoder' not in cfg.aggregation_type:
                        print("copying aggregated weights normally")
                        models[i].load_state_dict(model_to_distribute.state_dict())
                    else:
                        print('STARTING CLUSTER OR ENCODER ONLY')
                        cluster_keys = {'cluster_probe.clusters'}
                        current_state_dict = models[i].state_dict()
                        new_state_dict = model_to_distribute.state_dict()

                        for key in current_state_dict:
                            if "not_encoder" in cfg.aggregation_type:
                                print("starting copying cluster keys only..")
                                if key in cluster_keys:  # todo: this is a modification of encoder only to make it centroid only!
                                    current_state_dict[key] = new_state_dict[key]
                            else:
                                print("starting copying encoder keys only..")
                                if key not in cluster_keys:  # todo: this is a modification for encoder only
                                    current_state_dict[key] = new_state_dict[key]

                        models[i].load_state_dict(current_state_dict)
            else:
                model_to_distribute = copy.deepcopy(models[max_index])
                for i in range(cfg.client_num):
                    models[i].load_state_dict(model_to_distribute.state_dict())

            global_val = trainers[0].validate(model_to_distribute, val_loader)
            print(global_val)
            f.write(f"\nGlobal model agg: {j}\n")
            for key, value in global_val[0].items():
                f.write(f"{key}: {value}\n")  # Writes each key-value pair on a new line
            f.write("\n\n")  # Writes each key-value pair on a new line
            f.close()
        torch.save(model_to_distribute.state_dict(), agg_type_path + f"agg{j}_state_dict.pth")


if __name__ == "__main__":
    prep_args()
    my_app()
    with torch.no_grad():
        torch.cuda.empty_cache()

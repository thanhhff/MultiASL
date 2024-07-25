import os
import torch
import random
import numpy as np
import torch.utils.data as data
from utils import misc_utils
from dataset.mm_dataset import MMDataset
from model_factory import ModelFactory
from utils.loss import BinaryFocalLoss, CrossEntropyLoss, GeneralizedCE, TwoWayLoss
from config.config_mm import Config, parse_args
from tqdm import tqdm
import wandb


np.set_printoptions(formatter={'float_kind': "{:.2f}".format})

def load_weight(net, config):
    if config.load_weight:
        model_file = os.path.join(config.model_path, "best_model.pkl")
        print(">>> Loading weight from file: ", model_file)
        pretrained_params = torch.load(model_file)
        net.load_state_dict(pretrained_params, strict=False)
    else:
        print(">>> Training from scratch")


def get_dataloaders(config):
    train_loader = data.DataLoader(
        MMDataset(data_path=config.data_path, mode='train',
                      modal=config.modal, fps=config.fps,
                      num_frames=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='random', supervision='weak'),
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers)

    test_loader = data.DataLoader(
        MMDataset(data_path=config.data_path, mode='test',
                      modal=config.modal, fps=config.fps,
                      num_frames=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='uniform', supervision='weak'),
        batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers)
    
    ### Print length of train and test loader
    print("Length of train loader: ", len(train_loader))
    print("Length of test loader: ", len(test_loader))
    return train_loader, test_loader


def set_seed(config):
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False


class MMTrainer():
    def __init__(self, config):
        # config
        self.config = config

        # network
        self.net = ModelFactory.get_model(config.model_name, config)
        self.net = self.net.cuda()

        # data
        self.train_loader, self.test_loader = get_dataloaders(self.config)

        # loss, optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, 
                                           betas=(0.9, 0.999), weight_decay=0.0005)
        self.multi_class_criterion = TwoWayLoss()
        self.criterion_ASL = TwoWayLoss()
        self.Lgce = GeneralizedCE(q=self.config.q_val)

        # parameters
        self.total_loss_multi_per_epoch = 0
        self.total_loss_asl = 0
        self.best_mAP_multi_class = -1
        

    def test(self):
        ### mAP each class
        from torchmetrics.classification import MultilabelAveragePrecision
        mAP_each_class = MultilabelAveragePrecision(num_labels=12, average='none')
        load_weight(self.net, self.config)
        with torch.no_grad():
            targets, preds = [], []
            for _data, _label, _, _, _ in tqdm(self.test_loader):
                _data, _label = _data.cuda(), _label.cuda()
                x_cls, cas, action_flow, action_rgb = self.net(_data, is_training=False)
                combined_cas = misc_utils.instance_selection_function(torch.softmax(cas.detach(), -1), action_flow.permute(0, 2, 1).detach(), action_rgb.permute(0, 2, 1))
                _, topk_indices = torch.topk(combined_cas, self.config.num_segments // 8, dim=1)
                cas_top = torch.gather(cas, 1, topk_indices)
                cas_top = torch.mean(cas_top, dim=1)
                # Apply sigmoid
                score_supp = torch.sigmoid(cas_top)

                targets.append(_label.cpu())
                preds.append(score_supp.cpu())
            
            targets = torch.cat(targets).long()
            preds = torch.cat(preds)
            mAP_class = misc_utils.mAP(targets.numpy(), preds.numpy())
            mAP_sample = misc_utils.mAP(targets.t().numpy(), preds.t().numpy())

        mAP_score = mAP_each_class(preds, targets)
        print("Test mAP each class: ", mAP_score)
        print("Test mAP: ", mAP_class)
        print("Test mAP sample: ", mAP_sample)


    def evaluate_mutli_class(self, epoch=0, mode='test'):
        self.net = self.net.eval()

        data_loader = self.test_loader if mode == 'test' else self.train_loader
        with torch.no_grad():
            targets, preds = [], []
            for _data, _label, _, _, _ in tqdm(data_loader, desc="Evaluating '{}'".format(mode)):
                _data, _label = _data.cuda(), _label.cuda()
                x_cls, cas, action_flow, action_rgb = self.net(_data, is_training=False)
                combined_cas = misc_utils.instance_selection_function(torch.softmax(cas.detach(), -1), action_flow.permute(0, 2, 1).detach(), action_rgb.permute(0, 2, 1))
                _, topk_indices = torch.topk(combined_cas, self.config.num_segments // 8, dim=1)
                cas_top = torch.gather(cas, 1, topk_indices)
                cas_top = torch.mean(cas_top, dim=1)
                # Apply sigmoid
                score_supp = torch.sigmoid(cas_top)

                targets.append(_label.cpu())
                preds.append(score_supp.cpu())
            
            targets = torch.cat(targets).long()
            preds = torch.cat(preds)
            mAP_class = misc_utils.mAP(targets.numpy(), preds.numpy())
            mAP_sample = misc_utils.mAP(targets.t().numpy(), preds.t().numpy())

            # WANDB LOG
            print("Mode: {}, Epoch: {}, mAP: {:.5f}".format(mode, epoch, mAP_class))
            if mAP_class > self.best_mAP_multi_class and mode == 'test':
                self.best_mAP_multi_class = mAP_class
                print("New best test mAP: ", self.best_mAP_multi_class)
                torch.save(self.net.state_dict(), os.path.join(self.config.model_path, 'best_model.pkl'))
            wandb.log({f'{mode}_mAP': mAP_class, f'{mode}_mAP_sample': mAP_sample}, step=epoch)

        self.net = self.net.train()


    def calculate_pesudo_target(self, batch_size, label, topk_indices):
        cls_agnostic_gt = []
        cls_agnostic_neg_gt = []
        for b in range(batch_size):
            label_indices_b = torch.nonzero(label[b, :])[:,0]
            topk_indices_b = topk_indices[b, :, label_indices_b] # topk, num_actions
            cls_agnostic_gt_b = torch.zeros((1, 1, self.config.num_segments)).cuda()

            # positive examples
            for gt_i in range(len(label_indices_b)):
                cls_agnostic_gt_b[0, 0, topk_indices_b[:, gt_i]] = 1
            cls_agnostic_gt.append(cls_agnostic_gt_b)

        return torch.cat(cls_agnostic_gt, dim=0)  # B, 1, num_segments


    def calculate_all_losses(self, cas_top, _label, action_flow, action_rgb, cls_agnostic_gt):
        base_loss = self.criterion_ASL(cas_top, _label)
        cost = base_loss

        cls_agnostic_loss_flow = self.Lgce(action_flow.squeeze(1), cls_agnostic_gt.squeeze(1))
        cls_agnostic_loss_rgb = self.Lgce(action_rgb.squeeze(1), cls_agnostic_gt.squeeze(1))

        cost += cls_agnostic_loss_flow + cls_agnostic_loss_rgb
        return cost


    def forward_ASL(self, cas_fg, action_flow, action_rgb):
        combined_cas = misc_utils.instance_selection_function(torch.softmax(cas_fg.detach(), -1), 
                            action_flow.permute(0, 2, 1).detach(), action_rgb.permute(0, 2, 1))

        _, topk_indices = torch.topk(combined_cas, self.config.num_segments // 8, dim=1)
        cas_top = torch.mean(torch.gather(cas_fg, 1, topk_indices), dim=1)

        return cas_top, topk_indices, action_flow, action_rgb


    def train(self):
        # resume training
        load_weight(self.net, self.config)

        # training
        for epoch in range(self.config.num_epochs):
            for _data, _label, _, _, _ in tqdm(self.train_loader, desc='Training Epoch: {}'.format(epoch)):
                batch_size = _data.shape[0]
                _data, _label = _data.cuda(), _label.cuda()
                self.optimizer.zero_grad()
                # forward pass
                x_cls, cas_fg, action_flow, action_rgb = self.net(_data, is_training=True)
                loss_multi_class = self.multi_class_criterion(x_cls, _label)

                # ASL
                cas_top, topk_indices, action_flow, action_rgb = self.forward_ASL(cas_fg, action_flow, action_rgb)
                # calcualte pseudo target
                cls_agnostic_gt = self.calculate_pesudo_target(batch_size, _label, topk_indices)
                # losses
                loss_ASL = self.calculate_all_losses(cas_top, _label, action_flow, action_rgb, cls_agnostic_gt)

                loss = loss_multi_class + loss_ASL 
                loss.backward()

                self.optimizer.step()
                self.total_loss_multi_per_epoch += loss_multi_class.item()
                self.total_loss_asl += loss_ASL.item()

            # Log train loss
            wandb.log({'train_loss_multi': self.total_loss_multi_per_epoch, 
                       'train_loss_asl': self.total_loss_asl}, step=epoch)
            self.total_loss_multi_per_epoch, self.total_loss_asl = 0, 0

            self.evaluate_mutli_class(epoch=epoch, mode='test')
            if epoch % 10 == 0:
                self.evaluate_mutli_class(epoch=epoch, mode='train')


def main():
    args = parse_args()
    config = Config(args)
    set_seed(config)

    ### Wandb Initialization
    wandb.init(entity="thanhhff", 
               project="MM-Multi-Label-Action-Recognition", 
               group=args.model_name,
               name=args.exp_name, 
               config=config, 
               mode=args.wandb)

    trainer = MMTrainer(config)
    if args.inference_only:
        trainer.test()
    else:
        trainer.train()

    wandb.finish()

if __name__ == '__main__':
    main()

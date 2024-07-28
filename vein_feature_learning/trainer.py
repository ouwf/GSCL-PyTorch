import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from eval.evaluation import evaluate_verification
from torchvision.utils import save_image
import utils
import numpy as np
from loss.loss_functions import FusionLoss, CosFace, OnlineTripletLoss


class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.m = params['m']
        self.lr_scheduler = params["lr_scheduler"]
        self.batch_size = params['batch_size']
        self.save_image = params['save_image']
        self.args = params['args']

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, trainloader, testloader):
        loss_stats = utils.AverageMeter()
        self.initializes_target_network()
        best_result, best_snapshot = None, None
        for epoch in range(self.max_epochs):
            for batch_idx, ((batch_view_1, batch_view_2), _) in enumerate(trainloader):
                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)
                # save the first ten pairs of augmented images in the current mini-batch, keep 50 tiles at most
                if self.save_image:
                    sample = torch.cat((batch_view_1, batch_view_2), -1)
                    os.makedirs("./augmented_images", exist_ok=True)
                    save_image(sample[:10], "augmented_images/%d.bmp" % (batch_idx % 50), nrow=1, normalize=True, range=(-1.0, 1.0))
                loss = self.update(batch_view_1, batch_view_2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder

                loss_stats.update(loss.item())
                print(utils.dt(), 'Epoch:[%d]-[%d/%d] batchLoss:%.4f averLoss:%.4f' %
                      (epoch, batch_idx, len(trainloader), loss_stats.val, loss_stats.avg))

            roc, aver, auc = evaluate_verification(self.online_network, testloader, self.device)
            self.lr_scheduler.step()
            # save the current best model based on eer
            best_result, best_snapshot = \
                save_model(self.online_network, {'metrics': roc, 'eer': roc[0], 'epoch': epoch}, best_result, best_snapshot, self.args)
            print("End of epoch {}".format(epoch))

        print(utils.dt(), 'Training completed.')
        print(utils.dt(), '------------------Best Results---------------------')
        epoch, roc = best_result['epoch'], best_result['metrics']
        print(utils.dt(),
              'EER: %.2f%%, FPR100:%.2f%%, FPR1000:%.2f%%, FPR10000:%.2f%%, FPR0:%.2f%%, Aver: %.2f%% @ epoch %d' %
              (roc[0] * 100, roc[1] * 100, roc[2] * 100, roc[3] * 100, roc[4] * 100, np.mean(roc) * 100, epoch))

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1)[1])
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2)[1])

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)[1]
            targets_to_view_1 = self.target_network(batch_view_2)[1]

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1.detach())
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2.detach())
        return loss.mean()


class SimCLRTrainer:
    def __init__(self, online_network, optimizer, device, **params):
        self.online_network = online_network
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = params['max_epochs']
        self.lr_scheduler = params["lr_scheduler"]
        self.temperature = params['temperature']
        self.batch_size = params['batch_size']
        self.save_image = params['save_image']
        self.args = params['args']

    def train(self, trainloader, testloader):
        loss_stats = utils.AverageMeter()
        best_result, best_snapshot = None, None
        for epoch in range(self.max_epochs):
            for batch_idx, ((batch_view_1, batch_view_2), _) in enumerate(trainloader):
                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)
                # save the first ten pairs of augmented images in the current mini-batch, keep 50 tiles at most
                if self.save_image:
                    sample = torch.cat((batch_view_1, batch_view_2), -1)
                    os.makedirs("./augmented_images", exist_ok=True)
                    save_image(sample[:10], "augmented_images/%d.bmp" % (batch_idx % 50), nrow=1,
                               normalize=True, range=(-1.0, 1.0))
                loss = self.update(batch_view_1, batch_view_2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_stats.update(loss.item())
                print(utils.dt(), 'Epoch:[%d]-[%d/%d] batchLoss:%.4f averLoss:%.4f' %
                      (epoch, batch_idx, len(trainloader), loss_stats.val, loss_stats.avg))

            roc, aver, auc = evaluate_verification(self.online_network, testloader, self.device)
            self.lr_scheduler.step()
            # save the current best model based on eer
            best_result, best_snapshot = \
                save_model(self.online_network, {'metrics': roc, 'eer': roc[0], 'epoch': epoch}, best_result, best_snapshot, self.args)
            print("End of epoch {}".format(epoch))

        print(utils.dt(), 'Training completed.')
        print(utils.dt(), '------------------Best Results---------------------')
        epoch, roc = best_result['epoch'], best_result['metrics']
        print(utils.dt(),
              'EER: %.2f%%, FPR100:%.2f%%, FPR1000:%.2f%%, FPR10000:%.2f%%, FPR0:%.2f%%, Aver: %.2f%% @ epoch %d' %
              (roc[0] * 100, roc[1] * 100, roc[2] * 100, roc[3] * 100, roc[4] * 100, np.mean(roc) * 100, epoch))

    def update(self, batch_view_1, batch_view_2):
        # Simplified implementation: N-1 negative samples
        f_view_1 = self.online_network(batch_view_1)[1]
        f_view_2 = self.online_network(batch_view_2)[1]

        f_view_1_nml = F.normalize(f_view_1, p=2, dim=1)
        f_view_2_nml = F.normalize(f_view_2, p=2, dim=1)
        p = f_view_1_nml.matmul(f_view_2_nml.T)

        t = self.temperature
        loss_1 = -1 * F.log_softmax(p / t, dim=1).diag().mean()
        loss_2 = -1 * F.log_softmax(p / t, dim=0).diag().mean()
        loss = (loss_1 + loss_2) / 2

        # Full implementation: 2N-2 negative samples
        # f_view_1 = self.online_network(batch_view_1)[1]
        # f_view_2 = self.online_network(batch_view_2)[1]
        # bs = f_view_1.size(0)
        # f_view_1_nml = F.normalize(f_view_1, p=2, dim=1)
        # f_view_2_nml = F.normalize(f_view_2, p=2, dim=1)
        # p_ab = f_view_1_nml.matmul(f_view_2_nml.T)
        # p_aa = f_view_1_nml.matmul(f_view_1_nml.T)
        # p_bb = f_view_2_nml.matmul(f_view_2_nml.T)
        # non_diag_ind = torch.ones(bs).diag().bool().bitwise_not().cuda()
        # p_ab_aa = torch.cat((p_ab, p_aa[non_diag_ind].view(bs, -1)), 1)
        # p_ab_bb = torch.cat((p_ab, p_bb[non_diag_ind].view(bs, -1).T), 0)
        # t = self.temperature
        # loss_1 = -1 * F.log_softmax(p_ab_aa / t, dim=1).diag().mean()
        # loss_2 = -1 * F.log_softmax(p_ab_bb / t, dim=0).diag().mean()
        # loss = (loss_1 + loss_2) / 2
        return loss


class SupervisedTrainer:
    def __init__(self, network, loss, optimizer, device, **params):
        self.network = network
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = params['max_epochs']
        self.lr_scheduler = params["lr_scheduler"]
        self.batch_size = params['batch_size']
        self.save_image = params['save_image']
        if loss == 'softmax':
            self.loss = torch.nn.CrossEntropyLoss()
        elif loss == 'tripletloss':
            self.loss = OnlineTripletLoss(margin=params["hard_margin"])
        elif loss == 'cosface':
            self.loss = CosFace(s=params['s'], m=params['m'])
        elif loss == 'fusionloss':
            tripletloss = OnlineTripletLoss(margin=params['hard_margin'])
            cosface = CosFace(s=params['s'], m=params['m'])
            self.loss = FusionLoss(cls_loss=cosface, metric_loss=tripletloss, w_cls=params['w_cls'], w_metric=params['w_metric'])
        else:
            raise ValueError('Loss %s not supported!' % loss)
        self.args = params['args']

    def train(self, trainloader, testloader):
        loss_stats = utils.AverageMeter()
        best_result, best_snapshot = None, None
        for epoch in range(self.max_epochs):
            for batch_idx, (data, labels) in enumerate(trainloader):
                data, labels = data.to(self.device), labels.to(self.device)
                if self.save_image:
                    os.makedirs("./augmented_images", exist_ok=True)
                    save_image(data[:16], "augmented_images/%d.bmp" % (batch_idx % 50), nrow=4,
                               normalize=True, range=(-1.0, 1.0))
                loss = self.update(data, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_stats.update(loss.item())
                print(utils.dt(), 'Epoch:[%d]-[%d/%d] batchLoss:%.4f averLoss:%.4f' %
                      (epoch, batch_idx, len(trainloader), loss_stats.val, loss_stats.avg))

            roc, aver, auc = evaluate_verification(self.network, testloader, self.device)
            self.lr_scheduler.step()
            # save the current best model based on eer
            best_result, best_snapshot = \
                save_model(self.network, {'metrics': roc, 'eer': roc[0], 'epoch': epoch}, best_result, best_snapshot, self.args)
            print("End of epoch {}".format(epoch))

        print(utils.dt(), 'Training completed.')
        print(utils.dt(), '------------------Best Results---------------------')
        epoch, roc = best_result['epoch'], best_result['metrics']
        print(utils.dt(),
              'EER: %.2f%%, FPR100:%.2f%%, FPR1000:%.2f%%, FPR10000:%.2f%%, FPR0:%.2f%%, Aver: %.2f%% @ epoch %d' %
              (roc[0] * 100, roc[1] * 100, roc[2] * 100, roc[3] * 100, roc[4] * 100, np.mean(roc) * 100, epoch))

    def update(self, data, labels):
        features, logits = self.network(data)
        if isinstance(self.loss, OnlineTripletLoss):
            loss = self.loss(features, labels)
        elif isinstance(self.loss, (CosFace, torch.nn.CrossEntropyLoss)):
            loss = self.loss(logits, labels)
        elif isinstance(self.loss, FusionLoss):
            loss = self.loss((features, logits), labels)
        return loss


def save_model(model, current_result, best_result, best_snapshot, args):
    eer = current_result['eer']
    epoch = current_result['epoch']
    prefix = 'seed=%d_dataset=%s_network=%s_loss=%s' % (args.seed, args.dataset_name, args.network, args.loss)
    os.makedirs("snapshots", exist_ok=True)
    # save the current best model
    if best_result is None or eer <= best_result['eer']:
        best_result = current_result
        snapshot = {'model': model.state_dict(), 'epoch': epoch, 'args': args}
        if best_snapshot is not None:
            os.system('rm %s' % (best_snapshot))
        best_snapshot = './snapshots/%s_BestEER=%.2f_Epoch=%d.pth' % (prefix, eer * 100, epoch)
        torch.save(snapshot, best_snapshot)
    # always save the final model
    if epoch == args.max_epoch - 1:
        snapshot = {'model': model.state_dict(), 'epoch': epoch, 'args': args}
        last_snapshot = './snapshots/%s_FinalEER=%.2f_Epoch=%d.pth' % (prefix, eer * 100, epoch)
        torch.save(snapshot, last_snapshot)
    return best_result, best_snapshot
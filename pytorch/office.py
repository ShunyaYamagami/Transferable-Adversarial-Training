import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
import os
from collections import Counter
from utilities import *
from data import *
from networks import *
from typing import List
from myfunc import set_determinism, set_logger
from tqdm import tqdm
from datetime import datetime
import logging

# setGPU('0')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='Office31', choices=['Office31', 'OfficeHome', 'DomainNet'])
parser.add_argument("--task", type=str, default='true_domains')
parser.add_argument("--dset", type=str, default='amazon_dslr')
parser.add_argument("--train_batch_size", type=int, default=64)
# parser.add_argument("--total_epochs", type=int, default=2800)
parser.add_argument("--total_epochs", type=int, default=1500)
args = parser.parse_args()

args.checkpoints_dir = f'checkpoints/{args.dataset}/{args.task}/{args.dset}'
args.labeled_path = os.path.join(args.checkpoints_dir, 'labeled.pth')
args.unlabeled_path = os.path.join(args.checkpoints_dir, 'unlabeled.pth')
# args.test_path = os.path.join(args.checkpoints_dir, 'test.pth')

if args.dataset == 'Office31':
    args.num_classes = 31
    every_test_epoch = 10
elif args.dataset == 'OfficeHome':
    args.num_classes = 65
    every_test_epoch = 10
elif args.dataset == 'DomainNet':
    args.num_classes = 345
    every_test_epoch = 100
else:
    raise NotImplementedError


# create networks

set_determinism()
cuda = ''.join([str(i) for i in os.environ['CUDA_VISIBLE_DEVICES']])
exec_num = os.environ['exec_num'] if 'exec_num' in os.environ.keys() else 0
now = datetime.now().strftime("%y%m%d_%H:%M:%S")
log_dir = f'logs/{args.dataset}/{args.task}/{now}--c{cuda}n{exec_num}--{args.dset}--{args.task}'
set_logger(log_dir)
logger = logging.getLogger(__name__)
best_log_path = os.path.join(log_dir, 'best.txt')
latest_model_path = os.path.join(log_dir, 'model_latest.pth')
best_model_path = os.path.join(log_dir, 'model_best.pth')

cls = CLS(2048, args.num_classes, bottle_neck_dim = 256).cuda()
discriminator = LargeDiscriminator(2048).cuda()
scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=3000)
optimizer_cls = OptimWithSheduler(optim.Adam(cls.parameters(), weight_decay = 5e-4, lr = 1e-4),
                                  scheduler)
optimizer_discriminator = OptimWithSheduler(optim.Adam(discriminator.parameters(), weight_decay = 5e-4, lr = 1e-4),
                                  scheduler)

def get_dataset(path):
    lines: List[torch.Tensor, torch.Tensor, torch.Tensor] = torch.load(path)
    image = lines[0]
    label0 = lines[1].long()
    domain = lines[2].long()
    label = torch.zeros((len(label0), args.num_classes)).float()
    for i in range(len(label0)):	
        label[i] = one_hot(args.num_classes, label0[i])
    return image, label, domain

def save_models(model_path, k, best_acc, cls:nn.Module, discriminator:nn.Module):
    torch.save({
            'k': k,
            'best_acc': best_acc,
            'cls': cls.state_dict(),
            'discriminator': discriminator.state_dict(),
            }, model_path)

source_train, source_label, source_domain = get_dataset(args.labeled_path)
target_train, target_label, target_domain = get_dataset(args.unlabeled_path)

# =====================train
best_acc = 0.0
for k in tqdm(range(args.total_epochs)):
    mini_batches_source = get_mini_batches(source_train, source_label, source_domain, args.train_batch_size)
    mini_batches_target = get_mini_batches(target_train, target_label, target_domain, args.train_batch_size)
    for (i, ((im_source, label_source, domain_source), (im_target, label_target, domain_target))) in enumerate(
            zip(mini_batches_source, mini_batches_target)):
        
		
        # =========================generate transferable examples
        label_source_0 = Variable(label_source).cuda()
        feature_fooling = Variable(im_target.cuda(), requires_grad = True)
        feature_fooling_c = Variable(im_source.cuda(), requires_grad = True)
        feature_fooling_0 = feature_fooling.detach()
        feature_fooling_c1 = feature_fooling_c.detach()
        domain_source = domain_source.cuda()
        domain_target = domain_target.cuda()
		
        for i in range(20):
            # target discriminator
            # target domain labels -> 0
            scores = discriminator(feature_fooling)
            loss = BCELossForMultiClassification(domain_target.unsqueeze(1) , scores) - 0.1 * torch.sum((feature_fooling - feature_fooling_0) * (feature_fooling - feature_fooling_0))
            loss.backward()
            g = feature_fooling.requires_grad
            feature_fooling = feature_fooling + 2 * g 
            cls.zero_grad()
            discriminator.zero_grad()
            feature_fooling = feature_fooling.data.cpu().cuda()
        
        for xs in range(20):
            # source discriminator
            # source domain labels -> 1
            scorec = discriminator.forward(feature_fooling_c)
            losss = BCELossForMultiClassification(domain_source.unsqueeze(1),  scorec) - 0.1 * torch.sum((feature_fooling_c - feature_fooling_c1) * (feature_fooling_c - feature_fooling_c1))
            losss.backward()
            gss = feature_fooling_c.grad
            feature_fooling_c = feature_fooling_c +  2 * gss
            cls.zero_grad()
            discriminator.zero_grad()
            feature_fooling_c = Variable(feature_fooling_c.data.cpu().cuda(),requires_grad = True)
        
        for xss in range(20):
            # classifier
            _,_,_,scorec = cls.forward(feature_fooling_c)
            loss = CrossEntropyLoss(label_source_0, scorec) - 0.1 * torch.sum((feature_fooling_c - feature_fooling_c1) * (feature_fooling_c - feature_fooling_c1))
            loss.backward()
            gs = feature_fooling_c.grad
            feature_fooling_c = feature_fooling_c +  3 * gs
            cls.zero_grad()
            discriminator.zero_grad()
            feature_fooling_c = Variable(feature_fooling_c.data.cpu().cuda(),requires_grad = True)
        
		
		#==========================forward pass
        feature_source = Variable(im_source).cuda()
        label_source = Variable(label_source).cuda()
        feature_target = Variable(im_target).cuda()
        label_target = Variable(label_target).cuda()
        

        _, _, __, predict_prob_source = cls.forward(feature_source)
        _, _, __, predict_prob_target = cls.forward(feature_target)

        
        domain_prob_source = discriminator.forward(feature_source)
        domain_prob_target = discriminator.forward(feature_target)
        domain_prob_fooling = discriminator.forward(feature_fooling)
        domain_prob_fooling_c = discriminator.forward(feature_fooling_c)
        # domain_prob_fooling_c -> source
        # domain_prob_fooling -> target
        dloss_a = BCELossForMultiClassification(domain_source.unsqueeze(1), domain_prob_fooling_c.detach())
        dloss_a += BCELossForMultiClassification(domain_target.unsqueeze(1), domain_prob_fooling.detach())
        dloss = BCELossForMultiClassification(domain_source.unsqueeze(1), domain_prob_source)
        dloss += BCELossForMultiClassification(domain_target.unsqueeze(1), domain_prob_target)
        
        
        ce = CrossEntropyLoss(label_source, predict_prob_source)
        entropy = EntropyLoss(predict_prob_target)

        _, _, __, predict_prob_fooling = cls.forward(feature_fooling)
        _, _, __, predict_prob_fooling_c = cls.forward(feature_fooling_c)
        dis = torch.sum((predict_prob_fooling - predict_prob_target)*(predict_prob_fooling - predict_prob_target))
        ce_extra_c = CrossEntropyLoss(label_source, predict_prob_fooling_c)
        
		
		#=============================backprop
        with OptimizerManager([optimizer_cls , optimizer_discriminator]):
            loss = ce  + 0.5 * dloss + 0.5 * dloss_a + ce_extra_c + dis + 0.1 * entropy
            loss.backward()
                        
    if k % every_test_epoch == 0 or k == args.total_epochs:
        counter = AccuracyCounter()
        counter.addOntBatch(variable_to_numpy(predict_prob_source), variable_to_numpy(label_source))
        acc_train = Variable(torch.from_numpy(np.asarray([counter.reportAccuracy()], dtype=np.float32))).cuda()
        logger.info(f'Epoch[{k:4d}/{args.total_epochs}]')
        track_scalars(logger, ['ce', 'acc_train', 'dis', 'ce_extra_c', 'dloss', 'dloss_a', 'entropy'], globals())

        # ======================test
        with TrainingModeManager([cls], train=False) as mgr, Accumulator(['predict_prob','predict_index', 'label']) as accumulator:
            for (i, (im, label, _)) in enumerate(mini_batches_target):
                with torch.no_grad():
                    fs = Variable(im).cuda()
                    label = Variable(label).cuda()

                    __, fs,_,  predict_prob = cls.forward(fs)

                    predict_prob, label = [variable_to_numpy(x) for x in (predict_prob, label)]

                    label = np.argmax(label, axis=-1).reshape(-1, 1)
                    predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
                    accumulator.updateData(globals())

        for x in accumulator.keys():
            globals()[x] = accumulator[x]
            
        acc = float(np.sum(label.flatten() == predict_index.flatten()) )/ label.flatten().shape[0]
        logger.info(f'Epoch[{k:4d}/{args.total_epochs}]\tAccuracy: {acc:.4f}')

        if round(acc, 4) > round(best_acc, 4):
            best_acc = acc
            save_models(best_model_path, k, best_acc, cls, discriminator)
            with open(best_log_path, 'w') as f:
                f.write(f'Epoch[{k:4d}/{args.total_epochs}]\nAccuracy: {acc}\n')
        save_models(latest_model_path, k, best_acc, cls, discriminator)


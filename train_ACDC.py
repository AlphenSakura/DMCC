import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import (BaseDataSets, TwoStreamBatchSampler, WeakStrongAugment)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps, val_2d
from skimage.measure import label
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import  math
from einops import rearrange

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='DMCC', help='experiment_name')
parser.add_argument('--model', type=str, default='mcnet2d_v1', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=1, help='weight to balance all losses')
parser.add_argument('--block_size', type=int, default=16, help='block_size for masking operation')
parser.add_argument('--unsup_weight', type=float, default=1.0, help='weight for unsupervised loss')
parser.add_argument('--cp_weight', type=float, default=1.0, help='weight for copy paste loss')
# parser.add_argument('--threshold', type=float, default=0.75, help='threshold for pseudo label')

args = parser.parse_args()

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 47, "4": 111, "7": 191,
                    "11": 306, "14": 391, "18": 478, "35": 940}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i] #== c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()
    

def unsupervised_loss_by_threshold(pred, target, thresh):
    max_probs, targets_u = torch.max(target, dim=1)
    mask = max_probs.ge(thresh).bool() * (targets_u != 255).bool()
    targets_u[~mask] = 255
    loss = F.cross_entropy(pred, targets_u.detach(), ignore_index=255, reduction='none')
    return loss.mean()

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs

def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    dice_loss = losses.DiceLoss_bcp(n_classes=args.num_classes)
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    return loss_dice, loss_ce

def get_acdc_cp_loss(args, labeled_sub_bs, unlabeled_sub_bs, model, weak_batch, label_batch):
    img = weak_batch[:args.labeled_bs]
    uimg = weak_batch[args.labeled_bs:]
    lab = label_batch[:args.labeled_bs]
    ulab = label_batch[args.labeled_bs:]
    with torch.no_grad():
        pre_1, pre_2 = model(uimg)
        plab_1, plab_2 = get_ACDC_masks(pre_1, nms=1), get_ACDC_masks(pre_2, nms=1)
        img_mask, loss_mask = generate_mask(img)

    net_input_unl = uimg * img_mask + img * (1 - img_mask)
    net_input_l = img * img_mask + uimg * (1 - img_mask)
    out_unl_1, out_unl_2= model(net_input_unl)
    out_l_1, out_l_2= model(net_input_l)

    losses_u1 = mix_loss(out_unl_1, plab_1, lab, loss_mask, unlab=True)
    losses_u2 = mix_loss(out_unl_2, plab_2, lab, loss_mask, unlab=True)
    unl_ce, unl_dice = (losses_u1[0] + losses_u2[0]) / 2, (losses_u1[1] + losses_u2[1]) / 2
    losses_l1 = mix_loss(out_l_1, lab, plab_2, loss_mask)
    losses_l2 = mix_loss(out_l_2, lab, plab_1, loss_mask)
    l_ce, l_dice = (losses_l1[0] + losses_l2[0]) / 2, (losses_l1[1] + losses_l2[1]) / 2
    loss_ce = unl_ce + l_ce 
    loss_dice = unl_dice + l_dice
    return loss_ce,loss_dice

def train(args, snapshot_path):
    base_lr = args.base_lr
    labeled_bs = args.labeled_bs
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, 
                            split="train", 
                            num=None, 
                            transform=transforms.Compose([
        WeakStrongAugment(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    db_test = BaseDataSets(base_dir=args.root_path, split="test")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    model.train()
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,num_workers=1)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False,num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    consistency_criterion = losses.mse_loss
    dice_loss = losses.DiceLoss(n_classes=num_classes)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    labeled_sub_bs = int(labeled_bs // 2)
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            volume_batch_stong = sampled_batch['image_strong'].cuda()
           
            model.train()
            outputs = model(volume_batch)
            num_outputs = len(outputs)

            y_ori = torch.zeros((num_outputs,) + outputs[0].shape, device=outputs[0].device)
            y_pseudo_label = torch.zeros((num_outputs,) + outputs[0].shape, device=outputs[0].device)
            
            loss_seg = 0
            loss_seg_dice = 0 
            for idx in range(num_outputs):
                y = outputs[idx][:labeled_bs,...]

                loss_seg += ce_loss(y, label_batch[:labeled_bs][:].long())
                loss_seg_dice += dice_loss(F.softmax(y, dim=1), label_batch[:labeled_bs].unsqueeze(1))
   
                y_all = outputs[idx]
                y_prob_all = F.softmax(y_all, dim=1)
                y_ori[idx] = y_prob_all
                y_pseudo_label[idx] = sharpening(y_prob_all)
            
            loss_consist = 0
            loss_unsup = 0
            
            uncertainty = [-1.0 * torch.sum(y_ori[i] * torch.log(y_ori[i] + 1e-6), dim=1, keepdim=True) for i in range(num_outputs)]
            
            for i in range(num_outputs):
                j = (i + 1) % num_outputs
                uncertainty_o1 = uncertainty[i][labeled_bs:,...]
                uncertainty_o2 = uncertainty[j][labeled_bs:,...]
                mean_o1 = F.avg_pool2d(uncertainty_o1, kernel_size=args.block_size, stride=args.block_size)
                mean_o2 = F.avg_pool2d(uncertainty_o2, kernel_size=args.block_size, stride=args.block_size)
                mask = (mean_o1 < mean_o2).float().to(volume_batch_stong.device)
                mask_inter = F.interpolate(mask, size=(volume_batch_stong.shape[2], volume_batch_stong.shape[3]), mode='bilinear', align_corners=False)
                x = volume_batch_stong[labeled_bs:,...]  * mask_inter 
                
                outputs_strong = model(x)
                loss_consist += consistency_criterion(y_ori[i], y_pseudo_label[j])

                mask_pseudo = F.interpolate(mask, size=y_ori[i].shape[2:], mode='nearest')
                pseudo_label = y_ori[i][labeled_bs:,...]*mask_pseudo + y_ori[j][labeled_bs:,...]*(1-mask_pseudo)
                loss_unsup += sum(ce_loss(outputs_strong[idx], torch.argmax(pseudo_label, dim=1)) for idx in [i, j])
            
            iter_num = iter_num + 1
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            
            loss_ce, loss_dice = get_acdc_cp_loss(args, labeled_sub_bs=labeled_bs//2, unlabeled_sub_bs=(args.batch_size - labeled_bs)//2, model=model, weak_batch=volume_batch, label_batch=label_batch)
            loss_cp = (loss_ce + loss_dice) / 2
            loss = args.lamda * loss_seg_dice  + consistency_weight * loss_consist  + args.unsup_weight * loss_unsup + args.cp_weight * loss_cp 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logging.info('iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f, loss_unsup: %03f' % (iter_num, loss, loss_seg_dice, loss_consist, loss_unsup))
            
            writer.add_scalar('train/loss', loss, iter_num)
            writer.add_scalar('train/loss_seg', loss_seg, iter_num)
            writer.add_scalar('train/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/loss_consist', loss_consist, iter_num)
            writer.add_scalar('train/loss_unsup', loss_unsup, iter_num)
            writer.add_scalar('train/loss_cp', loss_cp, iter_num)

        

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/ACDC_{}_{}_labeled/{}".format(args.model, args.labelnum, args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

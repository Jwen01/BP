# Adaptive Mean Teacher with Structure-Aware Block Mixing for Semi-Supervised Medical Image Segmentation

import os
import sys
import random
import numpy as np
import warnings
import argparse
import logging
import shutil
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn

from utils import ramps, losses, test_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.mixmatch_util import mix_match_cross_select_pseudolabel2
from utils.util2 import WeightAdEMA
from utils.adamix_utils import AdaMix3D_boud

warnings.filterwarnings('ignore')

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LA', help='Pancreas_CT,LA,BraTS2019')
parser.add_argument('--root_path', type=str, default='../', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='debug', help='exp_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--max_iteration', type=int, default=30000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')  # 2
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')  # 4
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--sup_only', type=int, default=0, help='whether supervised training only')
parser.add_argument('--ad_ema', type=int, default=0, help='whether to use adaptive ema')
parser.add_argument('--ad_mix', type=int, default=0, help='whether to use adaptive mix')
parser.add_argument('--alpha', type=float, default=0.99, help='default alpha value')
parser.add_argument('--alpha_max', type=float, default=0.999, help='maximum value of alpha')
parser.add_argument('--alpha_min', type=float, default=0.99, help='minimum value of alpha')
parser.add_argument('--savefreq', type=int, default=200, help='frequency of model saving (in epochs)')
parser.add_argument('--topk', type=int, default=16, help='topk for adamix')
parser.add_argument('--per_num', type=int, default=8, help='each dimension of the feature map is divided into how many patches for adamix')
parser.add_argument('--weight_psemix', type=float, default=0.5, help='weight for pseudo_mix loss')
parser.add_argument('--adamw_lr', type=float, default=1e-4, help='learning rate for adamw')
parser.add_argument('--decay', type=float, default=1e-2, help='weight decay for adamw')
parser.add_argument('--snapshot_dir', type=str, default='', help='custom snapshot directory; if empty, auto-generated from args')
parser.add_argument('--amp', type=int, default=0, help='use CUDA automatic mixed precision to reduce memory')
parser.add_argument('--grad_accum_steps', type=int, default=1, help='micro-batches accumulated for one optimizer update')
parser.add_argument('--num_workers', type=int, default=0, help='number of data-loader workers')
parser.add_argument('--smoke_iterations', type=int, default=0, help='override max iterations for a short smoke test; 0 disables')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.snapshot_dir:
    snapshot_path = args.snapshot_dir
else:
    snapshot_path = args.root_path + "model/{}_{}_{}_labeled_suponly{}/{}/ad_ema{}-ad_mix{}-almax{}-almin{}-savefreq{}-topk{}-per_num{}-weight_psemix{}-adamw-{}-{}".format(
        args.dataset_name, args.exp, args.labelnum, args.sup_only,
        args.model, args.ad_ema, args.ad_mix, args.alpha_max, args.alpha_min,
        args.savefreq, args.topk, args.per_num, args.weight_psemix,
        args.adamw_lr, args.decay)
num_classes = 2
if args.dataset_name == "LA":
    # patch_size = (32, 32, 32)     #debug
    patch_size = (112, 112, 80)
    # args.root_path = args.root_path + 'data/LA'
    train_data_path = '../data/LA'
    args.max_samples = 80
    eval_start = 6000
    
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    # patch_size = (64, 64, 64)     #debug
    # args.root_path = args.root_path + 'data/Pancreas'
    train_data_path = '../data/Pancreas'
    args.max_samples = 62
    eval_start = 6000
    
elif args.dataset_name == "BraTS2019":
    patch_size = (96, 96, 96)
    # patch_size = (48, 48, 48) #debug
    # args.root_path = args.root_path + 'data/BraTS2019'
    train_data_path = '../data/BraTS2019'
    # args.model = 'vnets'
    args.max_samples = 250
    args.max_iteration = 60000
    eval_start = 26000
# train_data_path = args.root_path

if args.smoke_iterations > 0:
    args.max_iteration = args.smoke_iterations
    eval_start = args.max_iteration + 1

if args.grad_accum_steps < 1:
    raise ValueError('--grad_accum_steps must be at least 1')
if args.batch_size != 2 * args.labeled_bs:
    raise ValueError('SA-AMT expects equal labeled/unlabeled micro-batches: --batch_size must equal 2 * --labeled_bs')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def compute_alpha_from_softdice(student_logits,
                                teacher_logits,
                                alpha_min=0.99,
                                 alpha_max=0.999,
                                eps=1e-6):

    # Keep the overlap calculation in FP32 even when model inference uses AMP.
    ps = F.softmax(student_logits.float(), dim=1)
    pt = F.softmax(teacher_logits.float(), dim=1)

    # remove background
    ps = ps[:, 1:]
    pt = pt[:, 1:]

    B, C = ps.shape[:2]

    ps = ps.reshape(B, C, -1)
    pt = pt.reshape(B, C, -1)

    intersection = (ps * pt).sum(dim=2)

    denominator = (
        (ps * ps).sum(dim=2) +
        (pt * pt).sum(dim=2)
    )

    soft_dice = (2 * intersection + eps) / (denominator + eps)

    # mean over batch & class
    soft_dice = soft_dice.mean()

     # adaptive EMA (paper Eq.23: alpha = alpha_min + (alpha_max - alpha_min) * d_t)
    # d_t = soft_dice (teacher-student consistency); higher -> higher alpha -> slower EMA
    alpha = alpha_min + (alpha_max - alpha_min) * soft_dice

    alpha = alpha.clamp(alpha_min, alpha_max)

    return soft_dice.detach(), alpha.detach()

# from utils.util2 import FullAugmentor
# augmentor = FullAugmentor()

if __name__ == "__main__":
    
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(args)
#     logging.info("%s alpha=%s", args, alpha)


    def create_model(ema=False):
        # Network definition
        net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    model = create_model(ema=False)
    ema_model = create_model(ema=True)
    if args.dataset_name == "LA":
        db_train = LAHeart_no_read(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]),with_idx=True)
    elif args.dataset_name == "Pancreas_CT":   # Pancreas_no_read # Pancreas
        db_train = Pancreas_no_read(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]),with_idx=True)
    elif args.dataset_name == "BraTS2019":
        db_train = BraTS2019_no_read(base_dir=train_data_path,
                             split='train',
                             transform=transforms.Compose([
                                RandomRotFlip(),
                                # RandomRot(),
                                RandomCrop(patch_size),
                                ToTensor(),
                             ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.adamw_lr, betas=(0.9, 0.999), weight_decay=args.decay)
    
    
    ema_optimizer = WeightAdEMA(model, ema_model)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    consistency_criterion = losses.mse_loss
    # dice_loss = losses.Binary_dice_loss
    iter_num = 0
    best_dice = 0

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    amp_enabled = bool(args.amp) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    max_micro_iterations = max_iterations * args.grad_accum_steps
    max_epoch = max_micro_iterations // len(trainloader) + 1
    optimizer_steps_per_epoch = max(1, len(trainloader) // args.grad_accum_steps)
    
    iterator = tqdm(range(max_epoch), ncols=70)
    augu = torch.tensor(1).cuda()
    start = datetime.now()
    time_start = start.strftime("%Y-%m-%d %H:%M:%S")
    print('start-time', time_start)
    adamix = AdaMix3D_boud(total_steps=max(1, max_iterations // optimizer_steps_per_epoch), num_classes=num_classes, image_size=patch_size, patch_size=args.per_num, topk=args.topk)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    alpha_accumulator = 0.0
    logging.info('AMP=%s, micro_batch=%s (%s labeled + %s unlabeled), grad_accum_steps=%s, effective_batch=%s',
                 amp_enabled, args.batch_size, args.labeled_bs, args.batch_size - args.labeled_bs,
                 args.grad_accum_steps, args.batch_size * args.grad_accum_steps)

    for epoch_num in iterator:
        
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch = volume_batch.cuda(non_blocking=True)
            label_batch = label_batch.cuda(non_blocking=True)

            l_image = volume_batch[:args.labeled_bs]
            l_label = label_batch[:args.labeled_bs]

            ul_image = volume_batch[args.labeled_bs:]
            X = list(zip(l_image, l_label))

            # The teacher path is no-grad; autocast reduces its temporary activation memory.
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                _, _, _, pseudo_label, dl_label, _, t_u, _, _ = mix_match_cross_select_pseudolabel2(
                    X, ul_image, eval_net=ema_model, K=1, alpha=0.75,
                    mixup_mode='_x', aug_factor=augu)

            model.train()

            X_data = l_image
            X_label = l_label.float()
            U_data = ul_image
            U_data_pseudo = pseudo_label.float()
            dl_label = dl_label.float()

            X = torch.cat((X_data, U_data), 0)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                output_all = model(X)
            # Dice and adaptive-EMA reductions stay in FP32 for numerical stability.
            output_all_softmax = torch.softmax(output_all.float(), dim=1)
            conf_l = output_all_softmax[:args.labeled_bs].max(dim=1)[0]
            conf_u = output_all_softmax[args.labeled_bs:args.labeled_bs*2].max(dim=1)[0]
            do_ad_mix = bool(args.ad_mix) and not bool(args.sup_only)
            if do_ad_mix:
                U_data_m, U_mix_label, _, adloss, tok_list = adamix(
                    oimage=U_data, aimage=X_data, olabel=pseudo_label.long(),
                    alabel=dl_label.long(), oconf=conf_u, aconf=conf_l,
                    prediction=output_all[args.labeled_bs:args.labeled_bs*2],
                    cur_step=epoch_num)
                U_mix_label = U_mix_label.float()

            output_all_u = output_all[args.labeled_bs:args.labeled_bs*2]
            out_soft_u = output_all_softmax[args.labeled_bs:2*args.labeled_bs]

            if args.ad_ema:
                softdice, ad_alpha = compute_alpha_from_softdice(output_all_u, t_u, alpha_min=args.alpha_min, alpha_max=args.alpha_max, eps=1e-6)
            else:
                ad_alpha = args.alpha

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                loss_seg_ce_lab = ce_loss(output_all[:args.labeled_bs], X_label.long())
                loss_seg_dice_lab = dice_loss(output_all_softmax[:args.labeled_bs], X_label.long().unsqueeze(1))
                supervised_loss = 0.5 * (loss_seg_ce_lab + loss_seg_dice_lab)
                u_loss = 0.5 * (ce_loss(output_all_u, U_data_pseudo.long()) +
                                dice_loss(out_soft_u, U_data_pseudo.long().unsqueeze(1)))

                if args.sup_only:
                    first_loss = supervised_loss
                elif do_ad_mix:
                    first_loss = supervised_loss + u_loss * (1 - args.weight_psemix)
                else:
                    first_loss = supervised_loss + u_loss

            supervised_loss_value = float(supervised_loss.detach())
            u_loss_value = float(u_loss.detach())
            softdice_value = float(softdice) if args.ad_ema else None
            current_alpha = float(ad_alpha)

            # Backpropagate the regular branch before the mixed branch. This is
            # mathematically the same gradient sum, but frees one 3D V-Net graph
            # before allocating the next one.
            scaler.scale(first_loss / args.grad_accum_steps).backward()
            first_loss_value = float(first_loss.detach())
            del first_loss, supervised_loss, u_loss, output_all, output_all_u
            del output_all_softmax, out_soft_u, conf_l, conf_u, t_u

            if do_ad_mix:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                    output_m = model(U_data_m)
                output_m_softmax = torch.softmax(output_m.float(), dim=1)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
                    umix_loss = 0.5 * (ce_loss(output_m, U_mix_label.long()) +
                                       dice_loss(output_m_softmax, U_mix_label.long().unsqueeze(1)))
                    weighted_umix_loss = umix_loss * args.weight_psemix
                scaler.scale(weighted_umix_loss / args.grad_accum_steps).backward()
                umix_loss_value = float(umix_loss.detach())
                loss_value = first_loss_value + float(weighted_umix_loss.detach())
                del output_m, output_m_softmax, umix_loss, weighted_umix_loss
            else:
                umix_loss_value = None
                loss_value = first_loss_value

            micro_step += 1
            alpha_accumulator += current_alpha
            if micro_step % args.grad_accum_steps != 0:
                continue

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            step_was_skipped = amp_enabled and scaler.get_scale() < scale_before
            optimizer.zero_grad(set_to_none=True)
            ema_alpha = alpha_accumulator / args.grad_accum_steps
            alpha_accumulator = 0.0
            if step_was_skipped:
                logging.warning('AMP skipped an optimizer update because of non-finite gradients')
                continue

            iter_num += 1
            ema_optimizer.step(alpha=ema_alpha)

            if args.ad_ema:
                writer.add_scalar('ad_alpha', current_alpha, iter_num)
                writer.add_scalar('softdice', softdice_value, iter_num)
            if do_ad_mix:
                for i in range(len(adloss)):
                    v = adloss[i]
                    writer.add_scalar(f'adloss/adloss{i+1}', v.item() if hasattr(v, 'item') else v, iter_num)
                for i in range(len(tok_list)):
                    v = tok_list[i]
                    writer.add_scalar(f'topk/topk{i+1}', v.item() if hasattr(v, 'item') else v, iter_num)
                writer.add_scalar('umix_loss', umix_loss_value, iter_num)
            writer.add_scalar('sup_loss', supervised_loss_value, iter_num)
            writer.add_scalar('u_loss', u_loss_value, iter_num)
            writer.add_scalar('loss', loss_value, iter_num)
            writer.add_scalar('memory/max_allocated_gb', torch.cuda.max_memory_allocated() / (1024 ** 3), iter_num)
        
            if iter_num >= eval_start and iter_num % args.savefreq == 0:                
                now = datetime.now()
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                print(time_str)
                
                model.eval()

                if args.dataset_name =="LA":
                    dice_sample, jc_sample, hd_sample, asd_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4, dataset_name = 'LA')
                elif args.dataset_name =="Pancreas_CT":
                    dice_sample, jc_sample, hd_sample, asd_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=32, stride_z=32, dataset_name = 'Pancreas_CT')
                elif args.dataset_name =="BraTS2019":
                    dice_sample, jc_sample, hd_sample, asd_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=16, stride_z=16, dataset_name = 'BraTS2019')

                if dice_sample > best_dice:
                    best_dice = dice_sample
                    if args.dataset_name =="BraTS2019":
                        save_mode_path = os.path.join(snapshot_path,  f'{args.exp}_{iter_num}_best_model.pth')
                    else:
                        save_mode_path = os.path.join(snapshot_path,  f'{args.exp}_best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info(f"save best model to {save_mode_path}")
                logging.info(f"iter {iter_num}, dice_sample: [{dice_sample*100:.2f}/{best_dice*100:.2f}]")
                
                
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                
                model.train()
            
            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                logging.info('peak CUDA memory: %.2f GiB', torch.cuda.max_memory_allocated() / (1024 ** 3))
                print('best_dice',best_dice)
                break
            
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

# -*- coding: utf-8 -*-
import os
import argparse
import torch.backends.cudnn as cudnn
import torch.utils.data
from calculate_error import *
from datasets.datasets_list import DoTADatasetSubMPM, DADADatasetSubMPM
from pathlib import Path
from utils import *
from tadclip import *
import joblib
from tqdm import tqdm
import yaml


parser = argparse.ArgumentParser(description='CLIP for Traffic Anomaly Detection',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--accident_templates', nargs='+', type=str, default=['The {} vehicle collision with another {}', 'The {} vehicle out-of-control and leaving the roadway', 'the {} vehicle has an unknown accident', 'The vehicle is running normally on the road'])
parser.add_argument('--accident_prompt', nargs='+', type=str, default=['A traffic anomaly occurred in the scene', 'The traffic in this scenario is normal'])
parser.add_argument('--accident_classes', nargs='+', type=str, default=['ego', 'non-ego', 'vehicle', 'pedestrian', 'obstacle'])
parser.add_argument('--multi_class', action='store_true', help='multi class')
parser.add_argument('--prompt_len', type=int, default=6, help='prompt_len')  # 4 better that 16?
parser.add_argument('--ctx_init', type=str, default="The traffic in this scenario is")
parser.add_argument('--temperature', type=float, default=0.1)

# Directory setting
parser.add_argument('--models_list_dir', type=str, default='')
parser.add_argument('--model_dir', type=str, default='/ssd/qh/DoTA/data/TTHF/')
parser.add_argument('--other_method', type=str, default='TDAFF_BASE')  # default='MonoCLIP'
parser.add_argument('--base_model', type=str, default='RN50', help='base model: RN50, ViT-B-16, ViT-B-32, RN50x64, ViT-L-14')
parser.add_argument('--trainfile_dota', type=str,
                    default="/ssd/qh/DoTA/data/metadata/train_split.txt")
parser.add_argument('--testfile_dota', type=str,
                    default="/ssd/qh/DoTA/data/metadata/val_split.txt")

# Dataloader setting
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epoch_size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr_clip', default=5e-5, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr_other', default=5e-5, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--batch_size', default=96, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--dataset', type=str, default="DoTA")  # FIXME KITTI, NYU, DoTA
parser.add_argument('--wd', default=1e-4, type=float, help='Weight decay')
parser.add_argument("--warmup_length", type=int, default=500)

# Logging setting
parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency')
parser.add_argument('--val_in_train', type=bool, default=False, help='validation process in training')

# Model setting
parser.add_argument('--height', type=int, default=224)  # default 224(RN50), 448(RN50x64)
parser.add_argument('--width', type=int, default=224)  # default 224(RN50), 448(RN50x64)
parser.add_argument('--normal_class', type=int, default=1)
parser.add_argument('--fg', action='store_true', help='fine-grained prompts')
parser.add_argument('--general', action='store_true', help='general prompts')
parser.add_argument('--aafm', action='store_true', help='attentive anomaly focused mechanism')
parser.add_argument('--hf', action='store_true', help='high frequency information in temporal')
parser.add_argument('--classifier', action='store_true', help='classifier')

# Evaluation setting
parser.add_argument('--evaluate', action='store_true', help='evaluate score')
parser.add_argument('--eval_every', type=int, default=1000)
parser.add_argument('--multi_test', type=bool, default=False, help='evaluate score')

# Training setting
parser.add_argument('--train', action='store_true', help='training mode')
parser.add_argument('--exp_name', type=str, default='TDAFF_BASE_RN50')
parser.add_argument('--kl_div', action='store_true', help='inter frame kl divergence')

# GPU parallel process setting
parser.add_argument('--gpu_num', type=str, default="1", help='force available gpu index')
parser.add_argument('--rank', type=int, help='node rank for distributed training', default=0)


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def validate(args, val_loader, model, dataset='KITTI'):
    paths = dict(log_dir="%s/%s" % (args.model_dir, args.exp_name))
    os.makedirs(paths["log_dir"], exist_ok=True)
    ##global device
    if dataset in ['DoTA', 'DADA']:
        scores = []

    length = len(val_loader)
    # switch to evaluate mode
    model.eval()
    for i, batch in enumerate(val_loader):
        if args.other_method in ['TDAFF_BASE']:
            (rgb_data, rgb_data_c, _, _, _) = batch
            rgb_data = rgb_data.cuda()
            frame_c = rgb_data_c.cuda()
            input_img = rgb_data
            with torch.no_grad():
                if args.other_method == 'TDAFF_BASE':
                    if args.fg and args.general:
                        # from calflops import calculate_flops 
                        # flops, macs, params = calculate_flops(model = model, args=[input_img, frame_c, None,'eval'])
                        output_logits_m, output_logits_s = model(input_img, frame_c, mode='eval')
                    elif args.fg:
                        output_logits_m = model(input_img, frame_c, mode='eval')
                    else:
                        output_logits_s = model(input_img, frame_c, mode='eval')
                else:
                    raise ModuleNotFoundError("method not found")

        if dataset in ['DoTA', 'DADA']:
            if args.other_method in ['TDAFF_BASE']:
                if args.other_method == 'TDAFF_BASE':
                    if args.fg and args.general:
                        output_logits_s = output_logits_s.cpu().numpy()
                        output_logits_m = output_logits_m.cpu().numpy()

                        coarse_score = 1 - output_logits_s[:, -1]
                        refine_score = 1 - output_logits_m[:, -1]
                        frame_score = (coarse_score + refine_score) / 2
                    elif args.fg:
                        output_logits_m = output_logits_m.cpu().numpy()

                        frame_score = 1 - output_logits_m[:, -1]
                    else:
                        output_logits_s = output_logits_s.cpu().numpy()

                        frame_score = 1 - output_logits_s[:, -1]
        if i % 100 == 0:
            print('valid: {}/{}'.format(i, length))
        scores = np.append(scores, frame_score)

    if dataset == 'DoTA':
        joblib.dump(scores, os.path.join(paths['log_dir'], "frame_scores_%s_%s.json" % (
            args.height, args.width)))
        gt = joblib.load(
            open(os.path.join('/ssd/qh/DoTA//data', "ground_truth_demo/gt_label.json"),
                 "rb"))  # change path to DoTA dataset
        TAD_result = compute_tad_scores(scores, gt, args, sub_test=False)
        return TAD_result, scores
    elif dataset == 'DADA':
        joblib.dump(scores, os.path.join(paths['log_dir'], "frame_scores_%s_%s_dada.json" % (
            args.height, args.width)))
        gt = joblib.load(
            open(os.path.join('/data/lrq/DADA-2000', "ground_truth_demo/gt_label.json"),
                 "rb"))   # change path to DADA dataset
        TAD_result = compute_tad_scores(scores, gt, args, sub_test=False, dataset='dada')
        return TAD_result, scores

# debug:查看每部分时间
from time import perf_counter
flag_debug_t = False
debug_t = 0
def updata_debug_t():
        global debug_t
        debug_t = perf_counter()
        
def print_t(process='unknown process'):
    if flag_debug_t:
        print(f"{process} takes {(perf_counter() - debug_t):.4f}")
        updata_debug_t()    

def train_dota(args, train_loader, val_loader, model, optimizer, scheduler=None):
    if args.other_method in ['TDAFF_BASE']:
        loss_frame_m = nn.CrossEntropyLoss()
        loss_frame_s = nn.CrossEntropyLoss()
        loss_text_s = nn.CrossEntropyLoss()
        loss_text_m = nn.CrossEntropyLoss()
    paths = dict(log_dir="%s/%s" % (args.model_dir, args.exp_name),
                 ckpt_dir="%s/%s" % (args.model_dir, args.exp_name))
    os.makedirs(paths["ckpt_dir"], exist_ok=True)
    os.makedirs(paths["log_dir"], exist_ok=True)

    with open(os.path.join(paths["log_dir"], "clip_for_dota_cfg.yaml"), 'w') as f:
        yaml.dump(args, f)

    length = len(train_loader)
    best_auc = 0.0
    batch_idx = 0
    for epoch in range(args.epochs):
        model.train()
        total_losses = 0
        
        print_t(process="enter train")
        for i, batch in tqdm(enumerate(train_loader), desc="Training Epoch %d" % (epoch + 1),total=length):
                                                     
            print_t(process="Get batch")
            optimizer.zero_grad()
            if args.other_method in ['TDAFF_BASE']:
                frames, frame_c, one_hot_label_m, one_hot_label_s, _ = batch
                frame_c = frame_c.cuda()
                frames = frames.cuda()
                label_m = one_hot_label_m.cuda()
                label_s = one_hot_label_s.cuda()
                if args.other_method == 'TDAFF_BASE':
                    if args.fg and args.general:
                        logits_per_frame_m, logits_per_text_m, logits_per_frame_s, logits_per_text_s = model(
                            frames, frame_c, mode='train')
                        loss_img_m = loss_frame_m(logits_per_frame_m, label_m.long())
                        loss_img_s = loss_frame_s(logits_per_frame_s, label_s.long())
                        labels_m = label_m.t()
                        labels_m_ = torch.unique(labels_m, dim=0)
                        tmp_loss_m = []
                        logits_per_text_m_ = logits_per_text_m.gather(0, labels_m_.unsqueeze(-1).expand(-1,
                                                                                                        logits_per_text_m.shape[
                                                                                                            -1]))
                        for idx, tmp_class_idx in enumerate(labels_m_):
                            cur_tmp_loss = [logits_per_text_m_[idx][labels_m == tmp_class_idx].mean().unsqueeze(0)]
                            for cur_tmp_inner_idx in range(logits_per_text_m.shape[0]):
                                if cur_tmp_inner_idx == tmp_class_idx:
                                    continue
                                cur_tmp_loss.append(
                                    logits_per_text_m_[idx][labels_m == cur_tmp_inner_idx].mean().unsqueeze(0))
                            tmp_loss_m.append(torch.cat(cur_tmp_loss))
                        loss_t_m = loss_text_m(torch.stack(tmp_loss_m),
                                               torch.zeros(logits_per_text_m_.shape[0]).long().to(labels_m.device))

                        labels_s = label_s.t()
                        labels_s_ = torch.unique(labels_s, dim=0)
                        tmp_loss_s = []
                        logits_per_text_s_ = logits_per_text_s.gather(0, labels_s_.unsqueeze(-1).expand(-1,
                                                                                                        logits_per_text_s.shape[
                                                                                                            -1]))
                        for idx, tmp_class_idx in enumerate(labels_s_):
                            cur_tmp_loss = [logits_per_text_s_[idx][labels_s == tmp_class_idx].mean().unsqueeze(0)]
                            for cur_tmp_inner_idx in range(logits_per_text_s.shape[0]):
                                if cur_tmp_inner_idx == tmp_class_idx:
                                    continue
                                cur_tmp_loss.append(
                                    logits_per_text_s_[idx][labels_s == cur_tmp_inner_idx].mean().unsqueeze(0))
                            tmp_loss_s.append(torch.cat(cur_tmp_loss))
                        loss_t_s = loss_text_s(torch.stack(tmp_loss_s),
                                               torch.zeros(logits_per_text_s_.shape[0]).long().to(labels_s.device))

                        losses_m = loss_t_m + loss_img_m if not torch.isnan(loss_t_m).any() else loss_img_m
                        losses_s = loss_t_s + loss_img_s if not torch.isnan(loss_t_s).any() else loss_img_s
                        losses = (losses_m + losses_s) / 2
                    elif args.fg:
                        logits_per_frame_m, logits_per_text_m = model(
                            frames, frame_c, mode='train')
                        loss_img_m = loss_frame_m(logits_per_frame_m, label_m.long())
                        labels_m = label_m.t()
                        labels_m_ = torch.unique(labels_m, dim=0)
                        tmp_loss_m = []
                        logits_per_text_m_ = logits_per_text_m.gather(0, labels_m_.unsqueeze(-1).expand(-1,
                                                                                                        logits_per_text_m.shape[
                                                                                                            -1]))
                        for idx, tmp_class_idx in enumerate(labels_m_):
                            cur_tmp_loss = [logits_per_text_m_[idx][labels_m == tmp_class_idx].mean().unsqueeze(0)]
                            for cur_tmp_inner_idx in range(logits_per_text_m.shape[0]):
                                if cur_tmp_inner_idx == tmp_class_idx:
                                    continue
                                cur_tmp_loss.append(
                                    logits_per_text_m_[idx][labels_m == cur_tmp_inner_idx].mean().unsqueeze(0))
                            tmp_loss_m.append(torch.cat(cur_tmp_loss))
                        loss_t_m = loss_text_m(torch.stack(tmp_loss_m),
                                               torch.zeros(logits_per_text_m_.shape[0]).long().to(labels_m.device))

                        losses = loss_t_m + loss_img_m if not torch.isnan(loss_t_m).any() else loss_img_m
                    else:
                        logits_per_frame_s, logits_per_text_s = model(
                            frames, frame_c, mode='train')
                        loss_img_s = loss_frame_s(logits_per_frame_s, label_s.long())
                        labels_s = label_s.t()
                        labels_s_ = torch.unique(labels_s, dim=0)
                        tmp_loss_s = []
                        logits_per_text_s_ = logits_per_text_s.gather(0, labels_s_.unsqueeze(-1).expand(-1, logits_per_text_s.shape[-1]))
                        for idx, tmp_class_idx in enumerate(labels_s_):
                            cur_tmp_loss = [logits_per_text_s_[idx][labels_s == tmp_class_idx].mean().unsqueeze(0)]
                            for cur_tmp_inner_idx in range(logits_per_text_s.shape[0]):
                                if cur_tmp_inner_idx == tmp_class_idx:
                                    continue
                                cur_tmp_loss.append(
                                    logits_per_text_s_[idx][labels_s == cur_tmp_inner_idx].mean().unsqueeze(0))
                            tmp_loss_s.append(torch.cat(cur_tmp_loss))
                        loss_t_s = loss_text_s(torch.stack(tmp_loss_s),
                                               torch.zeros(logits_per_text_s_.shape[0]).long().to(labels_s.device))

                        losses = loss_t_s + loss_img_s if not torch.isnan(loss_t_s).any() else loss_img_s
                else:
                    raise ModuleNotFoundError("method not found")
            print_t(process=f"loop step")
            total_losses += losses.item()
            losses.backward()
            optimizer.step()

            if batch_idx % 200 == 0:
                print(
                    "[Step: {}/ Epoch: {}]: T_Loss: {:.4f}".format(
                        i + 1, epoch + 1,
                        losses))
                with open(os.path.join(paths['ckpt_dir'], "loss.txt"), 'a') as f:
                    if args.other_method in ['TDAFF_BASE']:
                        if args.other_method == 'TDAFF_BASE':
                            if args.fg and args.general:
                                f.write(
                                    "[Step: {}/ Epoch: {}]: T_Loss: {:.4f}, Loss_m: {:.4f}, Loss_s: {:.4f}, learning_rate: {:.6f}".format(
                                        i + 1, epoch + 1,
                                        losses,
                                        losses_m, losses_s,
                                        optimizer.param_groups[0]['lr']) + '\n')
                            else:
                                f.write(
                                    "[Step: {}/ Epoch: {}]: T_Loss: {:.4f}, learning_rate: {:.6f}".format(
                                        i + 1, epoch + 1,
                                        losses,
                                        optimizer.param_groups[0]['lr']) + '\n')
                    f.close()
            if batch_idx > 0 and batch_idx % args.eval_every == 0:
                torch.save({
                    'step': i,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(paths['ckpt_dir'], "epoch_%d_step_%d.pt" % (epoch + 1, i)))
                save_one_model(paths['ckpt_dir'], max_to_save=5)

                ############### eval ######################
                auc, scores = validate(args, val_loader, model, 'DoTA')
                model.train()  # FIXME !!!!!!!
                print(' AUC result: ', auc)
                with open(os.path.join(paths['ckpt_dir'], "loss.txt"), 'a') as f:
                    f.write(
                        "[Step: {}/ Epoch: {}]: Eval AUC: {:.4f}".format(
                            i + 1, epoch + 1,
                            auc) + '\n')
                    f.close()
                if auc >= best_auc:
                    best_auc = auc
                    print('Best AUC: ', auc)
                    with open(os.path.join(paths['ckpt_dir'], "loss.txt"), 'a') as f:
                        f.write(
                            "[Step: {}/ Epoch: {}]: Best AUC: {:.4f}".format(
                                i + 1, epoch + 1,
                                best_auc) + '\n')
                        f.close()
                    joblib.dump(scores, os.path.join(paths['log_dir'], "frame_scores_%s_%s_best.json" % (
                        args.height, args.width)))
                    only_model_saver(model.state_dict(), os.path.join(paths["ckpt_dir"], "best.pth"))
                scheduler.step(auc)  # For ReduceLROnPlateau
            batch_idx += 1

    return


def main():
    args = parser.parse_args()
    print("=> No Distributed Training")
    print('=> Index of using GPU: ', args.gpu_num)

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if args.dataset == 'DoTA':
        if args.other_method in ['TDAFF_BASE']:
            train_set = DoTADatasetSubMPM(args, train=True)
            # # train_set.__getitem__(0)
            test_set = DoTADatasetSubMPM(args, train=False)
        else:
            raise ModuleNotFoundError("method not found")
    elif args.dataset == 'DADA':
        if args.other_method in ['TDAFF_BASE']:
            train_set = None
            test_set = DADADatasetSubMPM(args, train=False)
            # test_set.__getitem__(0)

    print("=> Dataset: ", args.dataset)
    print("=> Data height: {}, width: {} ".format(args.height, args.width))
    if train_set:
        print('=> train  samples_num: {}  '.format(len(train_set)))
    if test_set:
        print('=> test  samples_num: {}  '.format(len(test_set)))

    train_sampler = None
    test_sampler = None

    if train_set:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, sampler=train_sampler)
    else:
        train_loader = None
    if test_set:
        val_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, sampler=test_sampler)
    else:
        val_loader = None

    cudnn.benchmark = True

    ###################### setting Network part ###################

    print("=> creating model")
    if args.other_method in ['TDAFF_BASE']:
        clip_model, _, _ = clip.create_model_and_transforms(args.base_model, pretrained='openai', jit=False,
                                                            cache_dir='/data/qh/pretrain_models/')
        tokenizer = clip.get_tokenizer(args.base_model)

        if args.other_method == 'TDAFF_BASE':
            Model = TDAFF_BASE(args, clip_model, tokenizer)
    else:
        raise ModuleNotFoundError("method not found")

    num_params = 0
    for p in Model.parameters():
        num_params += (p.numel() if p.requires_grad else 0)

    print("===============================================")
    print("Total parameters: {}".format(num_params))
    print("===============================================")

    Model = Model.cuda()

    args.evaluate = True

    if args.evaluate is True:
        ###################### setting model list #################################
        if args.multi_test is True:
            print("=> all of model tested")
            models_list_dir = Path(args.models_list_dir)
            models_list = sorted(models_list_dir.files('*.pkl'))
        else:
            print("=> just one model tested")
            models_list = [args.model_dir]
        test_model = Model

        print("Model Initialized")

        test_len = len(models_list)
        print("=> Length of model list: ", test_len)

        for i in range(test_len):
            if args.other_method in ['TDAFF_BASE']:
                print('loaded model')
                test_model.load_state_dict(
                    torch.load(os.path.join(args.model_dir, args.exp_name, 'best.pth'))["model_state_dict"])
            else:
                raise ModuleNotFoundError("method not found")
            test_model.eval()
            if args.dataset == 'DoTA':
                errors, scores = validate(args, val_loader, test_model, 'DoTA')
            elif args.dataset == 'DADA':
                errors, scores = validate(args, val_loader, test_model, 'DADA')
            print(' * model: {}'.format(models_list[i]))
            print("")
            print(' AUC result: ', errors)
            print("")
        print(args.dataset, " valdiation finish")
    else:
        print("Model Initialized")
        train_model = Model
        optimizer = torch.optim.Adam(train_model.parameters(), lr=args.lr_clip, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-7)
        print("=> Training")
        if args.dataset == 'DoTA':
            train_dota(args, train_loader, val_loader, train_model, optimizer, scheduler)
        print("")


if __name__ == "__main__":
    main()




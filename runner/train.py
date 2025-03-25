import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from tqdm import tqdm
import datetime
import os
import yaml
import copy
from runner.src.tools import CEloss , NLLloss
from runner.src.dota import gt_cls_target
from runner.src.utils import debug_weights,debug_guess,get_result_filename,load_results
from runner.src.metrics import evaluation, write_results ,  evaluation_per_scene , evaluation_on_obj
from runner.test import pama_test
from models.componets import HungarianMatcher

def save_checkpoint(cfg, e , model, optimizer, lr_scheduler, index_video, index_frame ):
    dir_chk = os.path.join(cfg.output, 'checkpoints')
    os.makedirs(dir_chk, exist_ok=True)
    path = os.path.join(dir_chk, 'model-{:02d}.pt'.format(e+1))
    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'index_video': index_video,
        'index_frame': index_frame,
    }, path)

def check_unused_para(model):
    for name, param in model.named_parameters():
        if param.grad is None:
             print(name)

class FrameState():
    def __init__(self) -> None:
        self.t = 0
        self.begin_t = 0
        self.T = 0
        pass

# debug:查看每部分时间
from time import perf_counter
flag_debug_t = False
debug_t = 0
def updata_debug_t():
        global debug_t
        debug_t = perf_counter()
        
def print_t(process='unknown process'):
    global flag_debug_t
    if flag_debug_t:
        print(f"{process} takes {(perf_counter() - debug_t):.4f}",force=True)
        updata_debug_t()     

def pama_train(cfg, model, train_sampler, traindata_loader, scaler, optimizer,lr_scheduler, 
               test_sampler=None, testdata_loader=None , index_video = 0 , index_frame= 0, ):

    # log writer
    if cfg.local_rank == 0:
        # Tensorboard
        writer = SummaryWriter(cfg.output + '/tensorboard/train_{}'.format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        # add custom title to distinguish
        writer.add_scalar(cfg.output, 0, 0)

    # matcher
    apply_ins_loss = cfg.get('apply_ins_loss',False)
    if apply_ins_loss:
        matcher = HungarianMatcher()

    # NF for vst
    fb = cfg.get('NF',0)
    # clip model use two image in every iteration 
    clip_model =  'clip' in cfg.get('model_type')

    celoss = CEloss(cfg)
    begin_epoch , total_epoch =  cfg.epoch , cfg.total_epoch 
    model.train(True)
    for e in range(begin_epoch, total_epoch):
        # DDP sampler
        if dist.is_initialized():
            train_sampler.set_epoch(e)
        
        # tqdm
        if cfg.local_rank == 0:
            pbar = tqdm(total=len(traindata_loader),desc='Epoch: %d / %d' % (e + 1, total_epoch))
        
        print_t(process="-----")
        # run in single video
        for j, (video_data, data_info, yolo_boxes, frames_boxes, video_name) in enumerate(traindata_loader):
            print_t(process="Get batch")
            # prepare data for model and loss func
            video_data = video_data.to(cfg.device, non_blocking=True) # [B,T,C,H,W]
            data_info = data_info.to(cfg.device, non_blocking=True)
            # yolo_boxes : list B x list T x nparray(N_obj, 4)
            yolo_boxes = np.array(yolo_boxes,dtype=object)
            # matcher between yolo and gt
            if apply_ins_loss:
                match_index = [matcher(yolo,gt) for yolo , gt in zip(yolo_boxes,frames_boxes)]

            # record whole video data
            B,T = video_data.shape[:2]
            t_shape = (B,T-fb)
            targets = torch.full(t_shape, -100).to(video_data.device)
            outputs = torch.full(t_shape, -100, dtype=float).to(video_data.device)        
            video_len_orig,toa_batch,tea_batch = data_info[:, 0] , data_info[:, 2] , data_info[:, 3]

            # loop in video frames
            rnn_state , frame_state = None , FrameState()
            for i in range(fb + clip_model , T):
                # preparation
                frame_state.t = i-fb
                frame_state.begin_t = 1 if clip_model else 0
                frame_state.T = T-1-fb
                #clip 
                if clip_model:
                    target = gt_cls_target(i, toa_batch, tea_batch).long()
                    batch_image_data , batch_boxes = video_data[:,i-1:i+1] , yolo_boxes[:,i]
                # vst
                elif fb: 
                    target = gt_cls_target(i-1, toa_batch, tea_batch).long()          
                    batch_image_data , batch_boxes = video_data[:,i-fb:i] , yolo_boxes[:,i-1]
                # dinov2  
                else:
                    target = gt_cls_target(i, toa_batch, tea_batch).long()
                    batch_image_data , batch_boxes = video_data[:,i] , yolo_boxes[:,i]

                with torch.cuda.amp.autocast(enabled=cfg.fp16):
                    optimizer.zero_grad()
                    if clip_model:
                        logits_per_image, logits_per_text = model(batch_image_data, batch_boxes, frame_state)
                        output = logits_per_image
                    else:
                        ret = model(batch_image_data, batch_boxes, rnn_state, frame_state)
                        output , rnn_state, outputs_ins_anormal= ret['output'] , ret['rnn_state'] , ret['ins_anomaly']

                    flt = i >= video_len_orig 
                    target = torch.where(flt, torch.full(target.shape, -100).to(video_data.device), target)
                    output = torch.where(flt.unsqueeze(dim=1).expand(-1,2), torch.full(output.shape, -100, dtype=output.dtype).to(video_data.device), output)
                    loss_frame = celoss(output,target)
                    loss = loss_frame
                    # instance anomaly loss
                    if apply_ins_loss:
                        if fb: 
                            now_match_index = [ x[i-1] for x in match_index ]
                        # dinov2  
                        else:
                            now_match_index = [ x[i] for x in match_index ]
                        '''
                        根据每帧yolo预测的object构造实例损失,例如:
                        outputs_ins_anormal: list[batch: list[object_nums,2]]
                        '''
                        # 构造gt objects
                        gt_zeros = [torch.zeros(single.shape[0]).type(torch.int64).to(video_data.device) for single in outputs_ins_anormal]                        
                        gt_objects = copy.deepcopy(gt_zeros)
                        for i,single in enumerate(gt_objects):
                            single[now_match_index[i][0]]=1
                        gt_objects = torch.cat(gt_objects, dim=0)
                        # batch single frame objects:  [ [box_num1,2], [box_num2,2], [box_num3,2]]  
                        src_objects = torch.cat(outputs_ins_anormal, dim=0)
                        loss_ins_anomaly = celoss(src_objects, gt_objects)
                        loss = loss + cfg.ins_loss_weight * loss_ins_anomaly
         
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # record data (loss) per iteration 
                loss_dict = {}
                loss_dict['loss_frame'] = loss_frame

                if cfg.local_rank == 0:
                    debug_weights(cfg.model_type, writer, model, loss_dict , index_frame, **cfg.train_debug)

                index_frame+=1

                # record whole video target and output
                targets[:, i-fb] = target.clone()
                out = output.softmax(dim=-1).max(1)[1] # select output(normal or anomaly)
                out[target == -100] = -100
                outputs[:, i-fb] = out    
                print_t(process=f"loop step{i}")
            # update for scheduler
            lr_scheduler.step()

            # filter first index
            if clip_model:
                outputs = outputs[:,1:]
                targets = targets[:,1:]

            # record data(lr,val) per epoch 
            if cfg.local_rank == 0: 
                writer.add_scalar('lr',optimizer.param_groups[-1]['lr'],index_video)    
                debug_guess(writer, outputs, targets, index_video)
                pbar.set_description('Epoch: %d / %d, Loss: %.4f' % (e + 1, total_epoch, loss))
                pbar.update(1)
            
            index_video+=1

        # save checkpoint
        if cfg.local_rank == 0 and (e+1) % cfg.snapshot_interval == 0:
            save_checkpoint(cfg, e , model, optimizer, lr_scheduler, index_video, index_frame )

        # test
        if cfg.test_inteval != -1 and (e + 1) % cfg.test_inteval == 0:
            filename = get_result_filename(cfg, e + 1)
            with torch.no_grad():
                pama_test(cfg, model, test_sampler, testdata_loader,  e + 1, filename)
                # updatate model stage
                model.train(True)
            if dist.is_initialized():
                dist.barrier()
            # record eval data
            if cfg.local_rank == 0: 
                txt_folder = os.path.join(cfg.output, 'evaluation')
                os.makedirs(txt_folder,exist_ok=True)
                txt_path = os.path.join(txt_folder, 'eval.txt')
                content = load_results(filename)
                per_class = cfg.get("dataset_type",'dota') == 'dota'
                write_results(txt_path, e + 1, *evaluation(**content,post_process=False,per_class=per_class))
                # instance level eval
                if 'obj_targets' in content and sum([len(x) for x in content['obj_targets']]):
                    write_results(txt_path,e + 1,*evaluation_on_obj(content['obj_outputs'],content['obj_targets'],content['video_name']),eval_type='instacne')
                # frame level eval
                if 'fra_outputs' in content and content['fra_outputs'][0][0] != -100:
                    write_results(txt_path, e + 1 , *evaluation(outputs = content['fra_outputs'], targets = content['targets']) , eval_type ='prompt frame')

        #  close an epoch bar 
        if cfg.local_rank == 0:
            pbar.close()
        

                   
                    



    

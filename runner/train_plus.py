import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from tqdm import tqdm
import datetime
import os
import yaml
import pickle

from runner.src.tools import CEloss
from runner.src.dota import gt_cls_target
from runner.src.utils import debug_weights,debug_guess,get_result_filename,load_results
from runner.src.metrics import evaluation, write_results ,  evaluation_per_scene , evaluation_on_obj

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


# debug:查看每部分时间
from time import perf_counter
flag_debug_t = False
debug_t = 0
def updata_debug_t():
        global debug_t
        debug_t = perf_counter()
        
def print_t(process='unknown process'):
    if flag_debug_t:
        print(f"{process} takes {(perf_counter() - debug_t):.4f}",force=True)
        updata_debug_t()     

'''
for multi model running in a dataloader

'''
class FrameState():
    def __init__(self) -> None:
        self.t = 0
        self.begin_t = 0
        self.T = 0
        pass

class train_runner():
    def __init__(self, cfg, model , scaler, optimizer, lr_scheduler, index_video = 0 , index_frame= 0 ) -> None:
        self.cfg = cfg
        self.model = model
        self.scaler = scaler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.index_video =index_video
        self.index_frame = index_frame

        self.model.train(True)
        self.celoss = CEloss(cfg)
        # log writer
        if cfg.local_rank == 0:
            # Tensorboard
            self.writer = SummaryWriter(cfg.output + '/tensorboard/train_{}'.format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
            # add custom title to distinguish
            self.writer.add_scalar(cfg.output, 0, 0)
    
    def runing_single_video(self,e,total_epoch,video_data,data_info,yolo_boxes):
        # record whole video data
        B,T = video_data.shape[:2]
        t_shape = (B,T)
        targets = torch.full(t_shape, -100).to(video_data.device)
        outputs = torch.full(t_shape, -100, dtype=float).to(video_data.device)        
        video_len_orig,toa_batch,tea_batch = data_info[:, 0] , data_info[:, 2] , data_info[:, 3]

        # loop in video frames
        rnn_state , frame_state = None , FrameState()
        for i in range(T):
            target = gt_cls_target(i, toa_batch, tea_batch).long()
            batch_image_data , batch_boxes = video_data[:,i] , yolo_boxes[:,i]          
            with torch.cuda.amp.autocast(enabled=self.cfg.fp16):
                self.optimizer.zero_grad()
                ret = self.model(batch_image_data, batch_boxes, rnn_state, frame_state)
                output , rnn_state= ret['output'] , ret['rnn_state']
                # frame loss
                flt = i >= video_len_orig           
                target = torch.where(flt, torch.full(target.shape, -100).to(video_data.device), target)
                output = torch.where(flt.unsqueeze(dim=1).expand(-1,2), torch.full(output.shape, -100, dtype=output.dtype).to(video_data.device), output)
                loss_frame = self.celoss(output,target)

                loss = loss_frame  
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # record data (loss) per iteration 
            loss_dict = {}
            loss_dict['loss_frame'] = loss_frame

            if self.cfg.local_rank == 0:
                debug_weights(self.cfg.model_type, self.writer, self.model, loss_dict , self.index_frame, **self.cfg.train_debug)

            self.index_frame+=1

            # record whole video target and output
            targets[:, i] = target.clone()
            out = output.softmax(dim=-1).max(1)[1] # select output(normal or anomaly)
            out[target == -100] = -100
            outputs[:, i] = out    
            print_t(process=f"loop step{i}")
        # update for scheduler
        self.lr_scheduler.step()

        # record data(lr,val) per epoch 
        if self.cfg.local_rank == 0: 
            self.writer.add_scalar('lr',self.optimizer.param_groups[-1]['lr'],self.index_video)    
            debug_guess(self.writer, outputs, targets, self.index_video)
        
        self.index_video+=1

    def reset_test_date(self,epoch):
        self.model.eval()
        self.filename_rank = self.filename
        # DDP sampler
        if dist.is_initialized():
            self.filename_rank = self.filename_rank.replace('.pkl', f'_rank{str(dist.get_rank())}.pkl')
        self.epoch = epoch
        self.targets_all , self.outputs_all , self.bbox_all= [] , [] , []
        self.obj_targets_all , self.obj_outputs_all , self.frame_outputs_all = [] , [] , []
        self.toas_all, self.teas_all,  self.idxs_all , self.info_all = [] , [] , [] , [] 
        self.frames_counter = []
        self.video_name_all = []

    def test_post_procesor(self,):
        # updatate model stage
        self.model.train(True)

        # collect results for all dataset
        # self.toas_all = np.array(self.toas_all).reshape(-1)
        # self.teas_all = np.array(self.teas_all).reshape(-1)
        # self.idxs_all = np.array(self.idxs_all).reshape(-1)
        # self.info_all = np.array(self.info_all).reshape(-1, 4)
        self.toas_all = np.array([item for sublist in self.toas_all for item in sublist]).reshape(-1)
        self.teas_all = np.array([item for sublist in self.teas_all for item in sublist]).reshape(-1)
        self.idxs_all = np.array([item for sublist in self.idxs_all for item in sublist]).reshape(-1)
        self.info_all = np.array([item for sublist in self.info_all for item in sublist]).reshape(-1, 4)
        self.frames_counter = np.array(self.frames_counter).reshape(-1)
        self.video_name_all = np.array(self.video_name_all).reshape(-1)

        print(f'save file {self.filename_rank}')
        with open(self.filename_rank, 'wb') as f:
            pickle.dump({
                'targets': self.targets_all,
                'outputs': self.outputs_all,
                'bbox_all':self.bbox_all,
                'obj_targets':self.obj_targets_all,
                'obj_outputs':self.obj_outputs_all,
                'fra_outputs':self.frame_outputs_all,
                'toas': self.toas_all,
                'teas': self.teas_all,
                'idxs': self.idxs_all,
                'info': self.info_all,
                'frames_counter': self.frames_counter,
                'video_name':self.video_name_all,
            }, f)

        if dist.is_initialized():
            dist.barrier()
        
            # combine all .pkl data 
        if self.cfg.local_rank == 0 and dist.is_initialized():
            device = torch.cuda.device_count()
            contents = []
            for i in range(device):
                with open(self.filename.replace('.pkl', f'_rank{str(i)}.pkl'), 'rb') as f:
                    contents.append(pickle.load(f))
                os.remove(self.filename.replace('.pkl', f'_rank{str(i)}.pkl'))

            targets_all = []
            outputs_all = []
            bbox_all = []
            obj_targets_all = []
            obj_outputs_all = []
            frame_outputs_all = []
            toas_all = []
            teas_all = []
            idxs_all = []
            info_all = []
            frames_counter = []
            video_name_all = []
            for i in range(device):
                ## ['targets'、'outputs']是由N（scene数量）个组成，所以用extend 追加 
                targets_all.extend(contents[i]['targets'])
                outputs_all.extend(contents[i]['outputs'])
                ## contents[i]['bbox_all'] 是N （scene数量）个 list ,每个list中有 M （代表帧数）个 list ,数据为shape=[M(当前帧的物体数量) ，4 ]的array
                ## 所以用extend 拆分为单个video
                bbox_all.extend(contents[i]['bbox_all'])
                obj_targets_all.extend(contents[i]['obj_targets'])
                obj_outputs_all.extend(contents[i]['obj_outputs'])
                frame_outputs_all.extend(contents[i]['fra_outputs'])
                toas_all.append(contents[i]['toas'])
                teas_all.append(contents[i]['teas'])
                idxs_all.append(contents[i]['idxs'])
                info_all.append(contents[i]['info'])
                frames_counter.append(contents[i]['frames_counter'])
                video_name_all.append(contents[i]['video_name'])

            
            toas_all = np.array(toas_all).reshape(-1)
            teas_all = np.array(teas_all).reshape(-1)
            idxs_all = np.array(idxs_all).reshape(-1)
            info_all = np.array(info_all).reshape(-1, 4)
            frames_counter = np.array(frames_counter).reshape(-1)
            video_name_all = np.array(video_name_all).reshape(-1)
            print(f'save file {self.filename_rank}')
            with open(self.filename, 'wb') as f:
                pickle.dump({
                    'targets': targets_all,
                    'outputs': outputs_all,
                    'bbox_all':bbox_all,
                    'obj_targets':obj_targets_all,
                    'obj_outputs':obj_outputs_all,
                    'fra_outputs':frame_outputs_all,
                    'toas': toas_all,
                    'teas': teas_all,
                    'idxs': idxs_all,
                    'info': info_all,
                    'frames_counter': frames_counter,
                    'video_name':video_name_all,
                }, f)

    def testing_single_video(self,video_data,data_info,yolo_boxes,video_name):
        # record whole video data
        B,T = video_data.shape[:2]
        t_shape = (B,T)

        # video data
        targets = torch.full(t_shape, -100).to(video_data.device)
        outputs = torch.full(t_shape, -100, dtype=float).to(video_data.device)  

        idx_batch,toa_batch,tea_batch = data_info[:, 1] , data_info[:, 2] , data_info[:, 3]
        info_batch = data_info[:, 7:11]

        batch_bbox = []
        batch_obj_targets = []
        batch_obj_outputs = []
        frame_outputs = torch.full(t_shape, -100, dtype=float).to(video_data.device)

        # loop in video frames
        rnn_state , frame_state = None , FrameState()
        for i in range(T):
            target = gt_cls_target(i, toa_batch, tea_batch).long()
            batch_image_data , batch_boxes = video_data[:,i] , yolo_boxes[:,i]        
            with torch.cuda.amp.autocast(enabled=self.cfg.fp16):
                ret = self.model(batch_image_data, batch_boxes, rnn_state, frame_state)
                output , rnn_state= ret['output'] , ret['rnn_state']
                loss_frame = self.celoss(output,target)
                loss = loss_frame
                output = output.softmax(dim=-1)
                     
            targets[:, i] = target.clone()
            outputs[:, i] = output[:, 1].clone()

        # concate bbox in batch into video -> batch_bbox : list ( T x list ( B x (np.ndarray(N_obj, 4) ) )  )
        video_bbox = [[] for _ in range(video_data.shape[0])]
        for sub_bbox in batch_bbox:  # split to time-frame -level
            for ind,bbox in enumerate(sub_bbox):  # split to batch-frame -level
                video_bbox[ind].append(bbox)

        # concate obj score from batch into video: 即 video 的 obj score : [ N  x [N-frame x [N-obj] ] ] 
        video_obj_targets = [[] for _ in range(video_data.shape[0])]
        video_obj_outputs = [[] for _ in range(video_data.shape[0])]
        for sub_target, sub_output in zip(batch_obj_targets,batch_obj_outputs): # split to time-frame -level
            for ind,(tar,out) in enumerate(zip(sub_target,sub_output)): # split to batch-frame -level
                video_obj_targets[ind].append(tar)
                video_obj_outputs[ind].append(out)
        
        # collect results for each video
        video_len_batch = data_info[:, 0].int() # delet padding part
        for i in range(targets.shape[0]):
            self.targets_all.append(targets[i][:video_len_batch[i]].view(-1).tolist())
            self.outputs_all.append(outputs[i][:video_len_batch[i]].view(-1).tolist())  
            self.frames_counter.append(video_len_batch[i].tolist())
            self.video_name_all.append(video_name[i])
            self.bbox_all.append(video_bbox[i][:video_len_batch[i]])
            self.obj_targets_all.append(video_obj_targets[i][:video_len_batch[i]])
            self.obj_outputs_all.append(video_obj_outputs[i][:video_len_batch[i]])
            self.frame_outputs_all.append(frame_outputs[i][:video_len_batch[i]].view(-1).tolist())
        self.toas_all.append(toa_batch.tolist())
        self.teas_all.append(tea_batch.tolist())
        self.idxs_all.append(idx_batch.tolist())
        self.info_all.append(info_batch.tolist())

    def write_eval_result(self,e):
        # record eval data
        if self.cfg.local_rank == 0: 
            txt_folder = os.path.join(self.cfg.output, 'evaluation')
            os.makedirs(txt_folder,exist_ok=True)
            txt_path = os.path.join(txt_folder, 'eval.txt')
            content = load_results(self.filename)
            write_results(txt_path, e, *evaluation(**content))
            # instance level eval
            if 'obj_targets' in content and sum([len(x) for x in content['obj_targets']]):
                write_results(txt_path,e,*evaluation_on_obj(content['obj_outputs'],content['obj_targets'],content['video_name']),eval_type='instacne')
            # frame level eval
            if 'fra_outputs' in content and content['fra_outputs'][0][0] != -100:
                write_results(txt_path, e , *evaluation(outputs = content['fra_outputs'], targets = content['targets']) , eval_type ='prompt frame')

                
def pama_test_plus(cfg, trainers, test_sampler, testdata_loader, epoch):
    
    # tqdm
    if cfg.local_rank == 0:
        pbar = tqdm(total=len(testdata_loader),desc='Test Epoch: %d / %d' % (epoch, cfg.total_epoch))

    for trainer in trainers.values():
        trainer.reset_test_date(epoch)
    
    for j, (video_data, data_info, yolo_boxes, frames_boxes, video_name) in enumerate(testdata_loader):
        # prepare data for model and loss func
        video_data = video_data.to(cfg.device, non_blocking=True) # [B,C,H,W]
        data_info = data_info.to(cfg.device, non_blocking=True)
        # yolo_boxes : list B x list T x nparray(N_obj, 4)
        yolo_boxes = np.array(yolo_boxes,dtype=object)

        # loop in trainers
        for trainer in trainers.values():
            trainer.testing_single_video(video_data,data_info,yolo_boxes,video_name)
            if dist.is_initialized():
                dist.barrier()

        if cfg.local_rank == 0:  
            pbar.update(1)
    
    for trainer in trainers.values():
        trainer.test_post_procesor()
        trainer.write_eval_result(epoch)

def pama_train_plus( cfg, trainer_dict, train_sampler, traindata_loader, test_sampler=None, testdata_loader=None ):
    
    begin_epoch , total_epoch =  cfg.epoch , cfg.total_epoch

    # prepare trainers 
    trainers = {}
    for name, para in trainer_dict.items():
        print(f'loader trainer {name=}')
        trainers[name] = train_runner(**para)

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
            video_data = video_data.to(cfg.device, non_blocking=True) # [B,T,...]
            data_info = data_info.to(cfg.device, non_blocking=True)
            # yolo_boxes : list B x list T x nparray(N_obj, 4)
            yolo_boxes = np.array(yolo_boxes,dtype=object)

            # loop in trainers
            for trainer in trainers.values():
                 trainer.runing_single_video(e,total_epoch,video_data,data_info,yolo_boxes)

            # record data(lr,val) per epoch 
            if cfg.local_rank == 0: 
                pbar.set_description('Epoch: %d / %d' % (e + 1, total_epoch))
                pbar.update(1)

        # save checkpoint
        if cfg.local_rank == 0 and (e+1) % cfg.snapshot_interval == 0:
            for trainer in trainers.values():
                save_checkpoint(trainer.cfg, e , trainer.model, trainer.optimizer, trainer.lr_scheduler, trainer.index_video, trainer.index_frame )

        # test
        if cfg.test_inteval != -1 and (e + 1) % cfg.test_inteval == 0:
            for trainer in trainers.values():
                filename = get_result_filename(trainer.cfg, e + 1)
                trainer.filename = filename

            with torch.no_grad():
                pama_test_plus(cfg, trainers, test_sampler, testdata_loader,  e + 1)

            if dist.is_initialized():
                dist.barrier()
                
        #  close an epoch bar 
        if cfg.local_rank == 0:
            pbar.close()
        

                   
                    



    

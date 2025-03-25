import cv2
import numpy as np
from tqdm import tqdm
import torch
from typing import List, Optional
import matplotlib.pyplot as plt
import os
import math

class ResizeTransform:
    def __init__(self, im_h: int, im_w: int , t_type = 'Frame_loss'): # t_type:  Frame_loss Instance_loss
        self.t_type = t_type
        h0 , h1 = self.feature_size(im_h)
        w0 , w1 = self.feature_size(im_w)
        self.height = h1 if t_type == 'Frame_loss' else h0
        self.width = w1 if t_type == 'Frame_loss' else w0

    @staticmethod
    def feature_size(s):
        s = math.ceil(s / 4)  # PatchEmbed
        s1 = math.ceil(s / 2)  # PatchMerging1
        s2 = math.ceil(s1 / 2)  # PatchMerging2
        s3 = math.ceil(s2 / 2)  # PatchMerging3
        return s1,s3

    def __call__(self, x ):      
        if self.t_type == 'Frame_loss' :
            # b,t,h,w,c = x.shape
            # [batch_size, T , H , W , C] -> [batch, C, T,  H,  W]
            result = x.permute(0, 4, 1, 2, 3)

        elif self.t_type == 'Instance_loss':
            b,hw,c = x.shape # [N_object,  H , W , C]
            result = x.reshape(b,self.height,self.width,c)
            # [N_object,  H , W , C] -> [N_object, C, H,  W]
            result = result.permute(0, 3, 1, 2)

        return result




class FrameState():
    def __init__(self) -> None:
        self.t = 0
        self.begin_t = 0
        self.T = 0
        pass

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))
        pass

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, imgs , boxes , rnn_state = None, frame_state = None):
        self.gradients = []
        self.activations = []
        return self.model( imgs, boxes, rnn_state, frame_state)
     
    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 cfg,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=True,
                 ):
        self.cfg = cfg
        # eval for reprodution
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        self.device = 'cpu'
        if self.cuda:
            self.model = model.cuda()
            self.device = 'cuda'
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        if len(grads.shape)==4:
            return np.mean(grads, axis=(2, 3), keepdims=True)
        elif len(grads.shape)==5:
            return np.mean(grads, axis=(2, 3, 4), keepdims=True)
        else:
            raise ValueError("grads.shape must be either 4 or 5 dimensions, received {}".format(len(grads.shape)))

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        # weighted_activations = activations
        if len(grads.shape)==4:
            cam = weighted_activations.sum(axis=1)
        elif len(grads.shape)==5:
            cam = weighted_activations.sum(axis=1).mean(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(video_data):
        width, height = video_data.size(-1), video_data.size(-2)
        return width, height

    def compute_cam_per_layer(self, video_data, index=None):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(video_data)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            # 观察具体值
            if index != None:
                # folder = "/data/qh/DoTA/output/debug/gradcam_compare/cam_data/[v2]base/D_pyFV4nKd4_001836/"
                folder = "/data/qh/DoTA/output/debug/gradcam_compare/cam_data/[v2]fullmodel/D_pyFV4nKd4_001836/"
                os.makedirs(folder, exist_ok=True)
                np.save(os.path.join(folder,f'{index}.npy'), cam)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, video_data, yolo_boxes = None , target_category = None , loss_type = 'Frame_loss', vst_norm = False):
        
        if self.cuda:
            video_data = video_data.cuda()
        yolo_boxes =  np.expand_dims(np.array(yolo_boxes,dtype=object),axis=0) # [T,] -> [B=1,T,]

        B,T,C,H,W = video_data.shape
        assert B == 1, f'Batch size must be 1, but got {B=}'
        v_len = T + 1 # add last frame
        NF = fb = 4
        rnn_state , frame_state = None , FrameState()
        snippets_gradcam = []
        outputs = torch.full( (B, v_len-NF), -100, dtype=float).to(self.device)
        for i in tqdm(range(NF,v_len),desc="Grad cam: "):        
            # preparation
            frame_state.t = i - fb
            frame_state.begin_t =  0
            frame_state.T = v_len - 1 - fb
            batch_image_data , batch_boxes = video_data[:,i-fb:i] , yolo_boxes[:,i-1]
            ret = self.activations_and_grads(batch_image_data, batch_boxes, rnn_state, frame_state)
            output , rnn_state, outputs_ins_anormal= ret['output'] , ret['rnn_state'] , ret['ins_anomaly']  
            
            # output = self.activations_and_grads(video_data)
            if isinstance(target_category, int):
                target_category = [target_category] * video_data.size(0)

            if target_category is None:
                target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
                print(f"category id: {target_category}")
            else:
                assert (len(target_category) == video_data.size(0))

            if loss_type == 'Frame_loss':
                self.model.zero_grad()
                loss = self.get_loss(output, target_category)
                loss.backward()
                cam_per_layer = self.compute_cam_per_layer(video_data)
                snippets_gradcam.append(self.aggregate_multi_layers(cam_per_layer))                 
                outputs[:, i-NF] = output.softmax(dim=1)[:, 1].detach().data
            elif loss_type == 'Instance_loss':
                # frame_level gradcam
                self.model.zero_grad()
                loss = self.get_loss(output, target_category)
                loss.backward(retain_graph=True)
                cam_per_layer = self.compute_cam_per_layer(video_data)
                frame_gradcam = self.aggregate_multi_layers(cam_per_layer)
                outputs[:, i-NF] = output.softmax(dim=1)[:, 1].detach().data

                # instance_level gradcam
                # outputs_ins_anormal ： [tensor,shape(N，2]  length = 1
                instance_outputs = outputs_ins_anormal[0]
                N_instance = instance_outputs.shape[0]
                instance_score , frame_instance_gradcam = [] , []
                if N_instance > 0 :
                    instance_score = instance_outputs.softmax(dim=1)[:,1].detach().data.tolist() 
                    for i in range(N_instance):
                        # clean grad
                        self.activations_and_grads.gradients = []
                        self.model.zero_grad()
                        loss = self.get_loss(instance_outputs[i].unsqueeze(0), target_category)
                        if i != N_instance-1:
                            loss.backward(retain_graph=True)
                        else:
                            loss.backward()
                        cam_per_layer = self.compute_cam_per_layer(video_data)
                        # only select the box_index layer when applying Instance Encoder
                        if not vst_norm :                  
                            cam_per_layer = [np.expand_dims(x[i],axis=0) for x in cam_per_layer]
                        
                        frame_instance_gradcam.append(self.aggregate_multi_layers(cam_per_layer))                         
                snippets_gradcam.append({'frame_gradcam':frame_gradcam,'obj_score':instance_score,'obj_gradcam':frame_instance_gradcam})        
        return snippets_gradcam,outputs.view(-1).tolist()
    
    
    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      crop_size: tuple = (None,None)) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :crop_size if crop from padding image , tuple (target_size,original_size)
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)

    if isinstance(crop_size[0],tuple):
        target_size, original_size = crop_size
        cam = center_crop_img(cam,target_size,original_size)

    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, target_size , original_size ):
    imgh,imgw = img.shape[:2]
    oldh,oldw =  original_size
    tarh,tarw =  target_size
    assert imgh==tarh and imgw==tarw
    scale = min ( tarh * 1.0 / oldh, tarw * 1.0 / oldw )
    newh, neww = oldh * scale, oldw * scale 
    delta_h = int((target_size[0] - newh)//2) 
    delta_w = int((target_size[1] - neww)//2)
    if delta_h>0:
        img = img[delta_h:-delta_h]
    if delta_w>0:
        img = img[:,delta_w:-delta_w]

    return img

def gradcam_vis(visualization:np.ndarray,
                outputs:Optional[List[float]],
                savepath: str,
                index:int=0,
                scenes_gt: dict =None,
                level: str = 'frame',
                obj_dic: dict = {},
                ):
    NF = 4
    if level ==  'frame':
        savepath = os.path.join(savepath,f'gradcam_{index+NF-1}.jpg')
    elif level == 'instance':
        obj_ind = obj_dic['obj_ind']
        obj_score = obj_dic['obj_score']*100
        savepath = os.path.join(savepath,f'frame_{index+NF-1}_object_{obj_ind}.jpg')
    # don't draw curve
    if scenes_gt is None or level == 'instance':
        plt.imshow(visualization)
        plt.axis('off')
        if level == 'instance':
            plt.title('frame {:3d}  score {:.1f} |  obj {:3d}  score {:.1f}'.format(index+NF-1,outputs[index]*100,obj_ind,obj_score))
        else:
            plt.title('frame {:3d} score {:03f}'.format(index+NF-1,outputs[index]))
        plt.savefig(savepath,dpi=150)  
        plt.clf()
    else:
        # grad_cam    
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12, 12))
        ax1.imshow(visualization)
        ax1.axis('off')
        ax1.set_title('frame {:3d} score {:.1f}'.format(index+NF-1,outputs[index]*100))
        #curve plot
        toa , tea = scenes_gt['anomaly_start']-NF+1 , scenes_gt['anomaly_end']-NF+1
        toa = max(toa,0)
        n_frames = len(outputs)
        xvals = np.arange(n_frames)
        ax2.set_aspect(n_frames*0.18)

        ax2.set_ylim(0, 1.0)
        ax2.set_xlim(0, n_frames)
        # Adjust x-axis ticks to be offset by 3  
        offset = NF-1   
        current_ticks  = ax2.get_xticks()
        shift_ticks = [tick - offset for tick in current_ticks  if tick>=offset ]
        # add 0 position
        shift_ticks.insert(0,0)
        ax2.set_xticks(shift_ticks)
        offset_labels = [int(x + offset) for x in shift_ticks]
        ax2.set_xticklabels(offset_labels) 
        ax2.plot(xvals[:index+1],outputs[:index+1], linewidth=3.0, color='r')
        ax2.axhline(y=0.5, xmin=0, xmax=n_frames, linewidth=3.0, color='g', linestyle='--')
        if toa >= 0 and tea >= 0:
            ax2.axvline(x=toa, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
            ax2.axvline(x=tea, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
            x = [toa, tea]
            y1 = [0, 0]
            y2 = [1, 1]
            ax2.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)
        plt.tight_layout()
        plt.savefig(savepath,dpi=150,bbox_inches='tight', pad_inches=0.1)  
        plt.close('all')



                
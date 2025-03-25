
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, \
                    average_precision_score, classification_report  
from datetime import datetime
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d

from runner.src import utils
from runner.src.dota import anomalies

'''
前向中值和后向中值滤波（不用）
def forward_medfilt(signal, kernel_size):
    n = len(signal)
    filtered_signal = np.zeros_like(signal)
    
    for i in range(n):
        if i < kernel_size - 1:
            # 在前面没有足够的点时，考虑所有可用点
            filtered_signal[i] = np.median(signal[:i+1])
        else:
            # 只考虑前面的 kernel_size 个点
            filtered_signal[i] = np.median(signal[i-kernel_size+1:i+1])
    
    return filtered_signal

def backward_medfilt(signal, kernel_size):
    n = len(signal)
    filtered_signal = np.zeros_like(signal)
    
    for i in range(n):
        # 确保只使用当前点和其后的点来计算中值
        end_idx = min(i + kernel_size, n)
        filtered_signal[i] = np.median(signal[i:end_idx])
    
    return filtered_signal
'''

def normalize_video(video_scores):
    video_scores_max = video_scores.max()
    video_scores_min = video_scores.min()
    if video_scores_max - video_scores_min > 0:
        video_scores = (video_scores - video_scores_min) / (video_scores_max - video_scores_min)
    return video_scores

def evaluation_on_obj(outputs,targets,video_name,eva_type='all'):
    if eva_type == 'all':
        # concate video output: [B,[T,[N]]] -> [B,sum N]
        outputs = [np.concatenate(x,axis=0) for x in outputs]
        targets = [np.concatenate(x,axis=0) for x in targets]
        # concate whole video output
        outputs = np.concatenate(outputs,axis=0) 
        targets = np.concatenate(targets,axis=0)
        return evaluation(outputs,targets)

    elif eva_type == 'scene':
        # concate video output: [B,[T,[N]]] -> [B,sum N]
        outputs = [np.concatenate(x,axis=0) for x in outputs]
        targets = [np.concatenate(x,axis=0) for x in targets]
        return evaluation_per_scene(outputs,targets,video_name)

    else:
        raise Exception(f'not supported instance eval type {eva_type}')

def evaluation_per_scene(outputs, targets, video_name, auc_type = 'frame', metric_type = 'AUC', post_process=False , kernel_size = 31, **kwargs):
    scene_score ={}
    assert len(outputs) == len(targets)," length of outputs is not equal to targets' "
    for i in range(len(outputs)):
        if auc_type == 'frame' : 
            preds = np.array(outputs[i])
            gts = np.array(targets[i])
            if post_process:
                if len(preds)>kernel_size:
                    preds = signal.medfilt(preds, kernel_size=kernel_size)
                else:
                    ks = len(preds) if len(preds)%2!=0 else len(preds)-1
                    preds = signal.medfilt(preds, kernel_size=ks)
                preds = normalize_video(np.array(preds))
        elif auc_type == 'instance':
            preds = np.concatenate(outputs[i])
            gts = np.concatenate(targets[i])
            # 计算AUC必须要有两个类
            if not np.any(gts):
                continue
        # 计算AUC必须要有两个类
        if len(np.unique(gts)) == 1:
            red_data = gts[0]==0
            gts = np.append(gts,red_data)
            preds = np.append(preds,red_data)
        
        # metric_type
        if metric_type == 'AUC':
            scene_score[video_name[i]]= roc_auc_score(gts, preds)
        elif metric_type == 'Accuracy':
            scene_score[video_name[i]]= accuracy_score(gts, preds > 0.5)

    scene_score = dict(sorted(scene_score.items(), key=lambda item: item[1]))
    return scene_score



'''
对每个scene计算AUC.
return : Dict{'scene_name' : auc_score,}
'''
def AUC_on_scene(pkl_path , auc_type = 'frame' , post_process=False):
    content = utils.load_results(pkl_path)
    if auc_type == 'frame':
        return evaluation_per_scene(**content , post_process = post_process)
    elif auc_type == 'instance':
        if 'obj_targets' in content and sum([len(x) for x in content['obj_targets']]):
            return evaluation_per_scene(content['obj_outputs'],content['obj_targets'],content['video_name'],auc_type)
        else:
            raise Exception(f'set AUC type is instance , but no obj_targets in {pkl_path}')
    raise Exception(f'unsupported AUC type {auc_type}, must be frame or instance')

'''
对每个scene计算 Accuracy.
return : Dict{'scene_name' : acc_score,}
'''
def Accuracy_on_scene(pkl_path , acc_type = 'frame' , post_process=False):
    content = utils.load_results(pkl_path)
    if acc_type == 'frame':
        return evaluation_per_scene(**content , post_process = post_process, metric_type='Accuracy')
    elif acc_type == 'instance':
        if 'obj_targets' in content and sum([len(x) for x in content['obj_targets']]):
            return evaluation_per_scene(content['obj_outputs'],content['obj_targets'],content['video_name'],acc_type, metric_type='Accuracy')
        else:
            raise Exception(f'set Accuracy type is instance , but no obj_targets in {pkl_path}')
    raise Exception(f'unsupported Accuracy type {acc_type}, must be frame or instance')


def evaluation(outputs, targets, info=None, post_process=False, kernel_size = 31, popr_type = 'mid' , per_class=False, **kwargs):
    if post_process:
        post_outputs = []
        for preds in outputs:
            ks = len(preds) if len(preds)%2!=0 else len(preds)-1 # make sure odd
            now_ks = min(kernel_size,ks)
            if popr_type == 'mid':
                preds = signal.medfilt(preds, kernel_size=now_ks)     
            elif popr_type == 'mean':
                preds = uniform_filter1d(preds, size=kernel_size)
            scores_each_video = normalize_video(np.array(preds))
            post_outputs.append(scores_each_video)
        outputs = post_outputs

    preds = np.array(utils.flat_list(outputs))
    gts = np.array(utils.flat_list(targets))
    F1_mean, _, F1_one = f1_mean(gts, preds)
    return (
        roc_auc_score(gts, preds),
        average_precision_score(gts, preds),
        F1_one,
        F1_mean,
        accuracy_score(gts, preds > 0.5),
        classification_report(
            gts, preds > 0.5, target_names=['normal', 'anomaly']),
        get_eval_per_class(outputs, targets, info, utils.split_by_class) if per_class else None,
        get_eval_per_class(outputs, targets, info, utils.split_by_class_ego) if per_class else None,
    )


def print_results(cfg, AUC_frame, PRAUC_frame, f1score, f1_mean, accuracy,
                  report, eval_per_class, eval_per_class_ego , per_class=True):
    
    print("[Correctness] f-AUC = %.5f" % (AUC_frame))
    print("             PR-AUC = %.5f" % (PRAUC_frame))
    print("           F1-Score = %.5f" % (f1score))
    print("           F1-Mean  = %.5f" % (f1_mean))
    print("           Accuracy = %.5f" % (accuracy))
    #  print("      accident pred = %.5f" % (acc_pred))
    #  print("        normal pred = %.5f" % (nor_pred))
    print()
    print(report)
    print()

    if per_class:
        print("***************** per class eval *****************\n")
        if eval_per_class is not None:
            print("---------------------- ALL ----------------------\n")         
            print('F1-mean per class\n')
            for key, values in eval_per_class.items():
                print('{:03f} F1-mean class {}\n'.format(list(values)[2], anomalies[int(key)-1]))
            print("\n")
            
            print('f-AUC per class\n')
            for key, values in eval_per_class.items():
                print('{:03f} f-AUC class {}\n'.format(list(values)[0], anomalies[int(key)-1]))

        if eval_per_class_ego is not None:
            clss = sorted(set([cls for cls, _ in eval_per_class_ego.keys()]))
            print("\n")
            # non_involve ego
            print("---------------- Non-involve ego ----------------\n")    
            fauc_ego = [list(eval_per_class_ego[(cls, 0.)])[0] for cls in clss]
            for key, values in zip(clss,fauc_ego):
                print('{:03f} f-AUC class {}\n'.format(
                    values, anomalies[int(key)-1]))
            print("\n")
            
            print("------------------ Involve ego ------------------\n")       
            fauc_ego = [list(eval_per_class_ego[(cls, 1.)])[0] for cls in clss]
            for key, values in zip(clss,fauc_ego):
                print('{:03f} f-AUC class {}\n'.format(
                    values, anomalies[int(key)-1]))  
                     
def write_results(file_name, epoch, AUC_frame, PRAUC_frame, f1score, f1_mean, accuracy,
                  report, eval_per_class, eval_per_class_ego, eval_type = 'frame', prefix = None, per_class=True):
    # 获取当前日期和时间
    now = datetime.now()
    # 格式化日期和时间
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

    with open(file_name, 'a') as file:
        file.write("\n######################### EPOCH #########################\n")
        # 写入评估指标
        file.write(f'{formatted_time}')
        if prefix:
            file.write(f'\n{prefix}')  
        file.write(f"\n----------------{eval_type} eval on epoch = {epoch}----------------\n")
        file.write("[Correctness] f-AUC = %.5f\n" % (AUC_frame))
        file.write("             PR-AUC = %.5f\n" % (PRAUC_frame))
        file.write("           F1-Score = %.5f\n" % (f1score))
        file.write("           F1-Mean  = %.5f\n" % (f1_mean))
        file.write("           Accuracy = %.5f\n" % (accuracy))
        # file.write("      accident pred = %.5f\n" % (acc_pred))
        # file.write("        normal pred = %.5f\n" % (nor_pred))
        file.write("\n")
        file.write(report + "\n")
        file.write("\n")
        
        if per_class:
            file.write("***************** per class eval *****************\n")
            if eval_per_class is not None:
                file.write("---------------------- ALL ----------------------\n")         
                file.write('F1-mean per class\n')
                for key, values in eval_per_class.items():
                    file.write('{:03f} F1-mean class {}\n'.format(list(values)[2], anomalies[int(key)-1]))
                file.write("\n")
                
                file.write('f-AUC per class\n')
                for key, values in eval_per_class.items():
                    file.write('{:03f} f-AUC class {}\n'.format(list(values)[0], anomalies[int(key)-1]))

            if eval_per_class_ego is not None:
                clss = sorted(set([cls for cls, _ in eval_per_class_ego.keys()]))
                file.write("\n")
                # non_involve ego
                file.write("---------------- Non-involve ego ----------------\n")    
                fauc_ego = [list(eval_per_class_ego[(cls, 0.)])[0] for cls in clss]
                for key, values in zip(clss,fauc_ego):
                    file.write('{:03f} f-AUC class {}\n'.format(
                        values, anomalies[int(key)-1]))
                file.write("\n")
                
                file.write("------------------ Involve ego ------------------\n")       
                fauc_ego = [list(eval_per_class_ego[(cls, 1.)])[0] for cls in clss]
                for key, values in zip(clss,fauc_ego):
                    file.write('{:03f} f-AUC class {}\n'.format(
                        values, anomalies[int(key)-1]))        
                
def f1_mean(gts, preds):
    F1_one = f1_score(gts, preds > 0.5)
    F1_zero = f1_score(
        (gts.astype('bool') == False).astype('long'),
        preds <= 0.5)
    F1_mean = 2 * (F1_one * F1_zero) / (F1_one + F1_zero)
    return F1_mean, F1_zero, F1_one


def get_eval_per_class(outputs, targets, info, split_fun):
    # retrocompat
    if info is None:
        return None
    # outputs/targets split per class
    ot = split_fun(outputs, targets, info)
    data = {}
    for cls, vals in ot.items():
        if vals['outputs'].shape[0]>0:
            data[cls] = { roc_auc_score(vals['targets'], vals['outputs']),
                        average_precision_score(vals['targets'], vals['outputs']),
                        f1_mean(vals['targets'], vals['outputs'])[0],
                        accuracy_score(vals['targets'], vals['outputs'] > 0.5),
                        } 
        else:
           data[cls] = [0,0,0,0]  # not existed the class
    return data
    # return {
    #     cls: {
    #         roc_auc_score(vals['targets'], vals['outputs']),
    #         average_precision_score(vals['targets'], vals['outputs']),
    #         f1_mean(vals['targets'], vals['outputs'])[0],
    #         accuracy_score(vals['targets'], vals['outputs'] > 0.5),
    #     } for cls, vals in ot.items()
    # }

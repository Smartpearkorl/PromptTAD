import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip_local import SimpleTokenizer as _Tokenizer
import open_clip_local as clip
import numpy as np
import copy
import torch.nn.init as init

_tokenizer = _Tokenizer()


def zeroshot_text_tad(accident_classes, templates):
    texts_list = []
    for i, template in enumerate(templates):
        if i == 0:
            for type in accident_classes[:2]:
                for obj in accident_classes[2:]:
                    texts = template.format(type, obj)
                    texts_list.append(texts)
        elif i == 1 or i == 2:
            for type in accident_classes[:2]:
                texts = template.format(type)
                texts_list.append(texts)
        elif i == 3:
            texts = template
            texts_list.append(texts)
    return texts_list


def zeroshot_text_tad_bi(templates):
    texts_list = []
    for i, template in enumerate(templates):
        texts = template
        texts_list.append(texts)
    return texts_list


def initialize_model(model):
    for name, param in model.named_parameters():
        try:
            if 'weight' in name:
                if 'batch_norm' not in name and 'bn' not in name:
                    init.xavier_uniform_(param)
            elif 'bias' in name:
                init.constant_(param, 0.0)
        except:
            init.normal_(param, mean=0, std=0.01)


class TDAFF_BASE(nn.Module):
    def __init__(self, args, pretrain_model, tokenizer):
        super(TDAFF_BASE, self).__init__()
        self.args = args
        self.fg = args.fg
        self.general = args.general
        self.aafm = args.aafm
        self.hf = args.hf
        self.classifier = args.classifier
        vis_dim = pretrain_model.visual.output_dim
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()
        if self.classifier:
            if self.hf:
                self.clip = pretrain_model
                self.encode_hp = copy.deepcopy(self.clip.visual)
                self.fuse_layer = nn.Sequential(
                    nn.Linear(2 * vis_dim, vis_dim),
                    nn.Dropout(0.2)
                )
                # init
                initialize_model(self.encode_hp)
            self.encoder = pretrain_model.visual
            # init
            initialize_model(self.encoder)
            if self.fg:
                self.linear_classifier = nn.Linear(vis_dim, 11)
            else:
                self.linear_classifier = nn.Linear(vis_dim, 2)
        else:
            self.accident_templates = args.accident_templates
            self.accident_classes = args.accident_classes
            self.accident_prompt = args.accident_prompt

            self.clip = pretrain_model

            if self.hf and self.aafm:
                self.encode_hp = copy.deepcopy(self.clip.visual)
                # frozen viusal
                for name, params in self.clip.visual.named_parameters():
                    params.requires_grad = False
                self.clip.positional_embedding.requires_grad = False
                self.clip.text_projection.requires_grad = False

                if self.args.base_model in ['RN50', 'RN50x64']:
                    if self.args.base_model == 'RN50':
                        self.patch_head = nn.Sequential(
                            nn.Linear(vis_dim * 2, vis_dim),
                            nn.Dropout(0.2)
                        )
                    else:
                        self.patch_head = nn.Sequential(
                            nn.Linear(vis_dim * 4, vis_dim),
                            nn.Dropout(0.2))
                elif self.args.base_model in ['ViT-B-16', 'ViT-B-32']:
                    self.patch_ln = nn.Sequential(
                        nn.Linear(768, vis_dim),
                        nn.Dropout(0.2)
                    )
                elif self.args.base_model in ['ViT-L-14']:
                    self.patch_ln = nn.Sequential(
                        nn.Linear(1024, vis_dim),
                        nn.Dropout(0.2)
                    )

                # VFR
                self.cattention_f = CAttention(vis_dim, 8, vis_dim // 2, vis_dim // 2)
                # LFR
                self.cattention_t = CAttention(vis_dim, 8, vis_dim // 2, vis_dim // 2)
                self.fusion = nn.Sequential(
                    nn.Linear(3 * vis_dim, vis_dim),
                    nn.Dropout(0.2)  # default 0.2
                )
            elif self.aafm:
                if self.args.base_model in ['RN50', 'RN50x64']:
                    if self.args.base_model == 'RN50':
                        self.patch_head = nn.Sequential(
                            nn.Linear(vis_dim * 2, vis_dim),
                            nn.Dropout(0.2)
                        )
                    else:
                        self.patch_head = nn.Sequential(
                            nn.Linear(vis_dim * 4, vis_dim),
                            nn.Dropout(0.2))
                elif self.args.base_model in ['ViT-B-16', 'ViT-B-32']:
                    self.patch_ln = nn.Sequential(
                        nn.Linear(768, vis_dim),
                        nn.Dropout(0.2)
                    )
                elif self.args.base_model in ['ViT-L-14']:
                    self.patch_ln = nn.Sequential(
                        nn.Linear(1024, vis_dim),
                        nn.Dropout(0.2)
                    )

                self.cattention_f = CAttention(vis_dim, 8, vis_dim // 2, vis_dim // 2)
                self.cattention_t = CAttention(vis_dim, 8, vis_dim // 2, vis_dim // 2)
                self.fusion = nn.Sequential(
                    nn.Linear(2 * vis_dim, vis_dim),
                    nn.Dropout(0.2)  # default 0.2
                )
            elif self.hf:
                # way 1
                self.encode_hp = copy.deepcopy(self.clip.visual)
                for name, params in self.clip.visual.named_parameters():
                    params.requires_grad = False
            if self.fg:
                self.text_f_m = tokenizer(zeroshot_text_tad(self.accident_classes, self.accident_templates))
            if self.general:
                self.text_f_s = tokenizer(zeroshot_text_tad_bi(self.accident_prompt))

    def forward(self, img_p, img_c, label=None, mode=None):
        if self.classifier:
            img_p_f, _ = self.encoder(img_p)
            img_c_f, _ = self.encoder(img_c)
            img_f = self.avgpool(torch.stack([img_p_f, img_c_f], 2)).squeeze(-1)
        else:
            img_p_f, img_p_token = self.clip.encode_image(img_p)
            img_c_f, img_c_token = self.clip.encode_image(img_c)
            (b, c, hw) = img_p_token.shape
            img_f = self.avgpool(torch.stack([img_p_f, img_c_f], 2)).squeeze(-1)
            img_token = self.avgpool(torch.stack([img_p_token, img_c_token], 3).reshape(b, -1, 2)).reshape(b, c, hw)
        if self.hf:
            diff_img = img_c - img_p

            img_diff_f, _ = self.encode_hp(diff_img)
            img_f_m = img_f + img_diff_f
        if self.aafm:
            if self.args.base_model in ['RN50', 'RN50x64']:
                img_token = self.patch_head(img_token.permute(0, 2, 1))

            elif self.args.base_model in ['ViT-B-16', 'ViT-B-32', 'ViT-L-14']:
                img_token = self.patch_ln(img_token)

        if self.hf:
            img_f_ = img_f_m / img_f_m.norm(dim=-1, keepdim=True)  # normalize img_f
        else:
            img_f_ = img_f / img_f.norm(dim=-1, keepdim=True)  # normalize img_f

        if self.classifier:
            if self.hf:
                scores = self.linear_classifier(img_f_m)
            else:
                scores = self.linear_classifier(img_f)
            if mode == 'eval':
                scores = self.sigmoid(scores)
            return scores
        else:
            if self.fg:
                text_f_m = self.clip.encode_text(self.text_f_m.to(img_f.device))
                text_f_m_ = text_f_m / text_f_m.norm(dim=-1, keepdim=True)
            if self.general:
                text_f_s = self.clip.encode_text(self.text_f_s.to(img_f.device))
                text_f_s = text_f_s / text_f_s.norm(dim=-1, keepdim=True)

            if self.aafm:
                # VFR
                if self.hf:
                    img_token_f = self.cattention_f(img_f.unsqueeze(1), img_token, img_token).squeeze(1)
                    img_token_f = img_token_f.squeeze(1)
                else:
                    img_token_f = self.cattention_f(img_f.unsqueeze(1), img_token, img_token).squeeze(1)

            if mode == 'eval':
                if self.fg:
                    logits_per_image_m = 100. * img_f_ @ text_f_m_.t()
                    logits_per_image_m = F.softmax(logits_per_image_m, dim=-1)

                if self.aafm and self.fg:
                    # LFR
                    attn_t = logits_per_image_m
                    # way 2
                    soft_text_f = attn_t @ text_f_m
                    img_token_t = self.cattention_t(soft_text_f.unsqueeze(1), img_token, img_token).squeeze(1)
                    img_token_t = img_token_t.squeeze(1)
                    if self.hf:
                        img_f_s = self.fusion(torch.cat([img_token_f, img_token_t, img_f_m], -1))
                    else:
                        img_f_s = self.fusion(torch.cat([img_token_f, img_token_t], -1))

                    img_f_s = img_f_s / img_f_s.norm(dim=-1, keepdim=True)

                    logits_per_image_s = 100. * img_f_s @ text_f_s.t()
                    logits_per_image_s = F.softmax(logits_per_image_s, dim=-1)

                elif self.general:
                    logits_per_image_s = 100. * img_f_ @ text_f_s.t()
                    logits_per_image_s = F.softmax(logits_per_image_s, dim=-1)
                if self.fg and self.general:
                    return logits_per_image_m, logits_per_image_s
                elif self.fg:
                    return logits_per_image_m
                else:
                    return logits_per_image_s
            else:
                logit_scale = self.clip.logit_scale.exp()
                if self.fg:
                    logits_per_image_m = logit_scale * img_f_ @ text_f_m_.t()
                    logits_per_text_m = logit_scale * text_f_m_ @ img_f_.t()

                if self.aafm and self.fg:
                    # LFR
                    if label is not None:
                        soft_text_f = torch.index_select(text_f_m_, 0, label.long())
                    else:
                        attn_t = F.softmax(logits_per_image_m, dim=-1)
                        soft_text_f = attn_t @ text_f_m
                    img_token_t = self.cattention_t(soft_text_f.unsqueeze(1), img_token, img_token).squeeze(1)
                    if self.hf:
                        img_f_s = self.fusion(torch.cat([img_token_f, img_token_t, img_f_m], -1))
                    else:
                        img_f_s = self.fusion(torch.cat([img_token_f, img_token_t], -1))

                    img_f_s = img_f_s / img_f_s.norm(dim=-1, keepdim=True)

                    logits_per_image_s = logit_scale * img_f_s @ text_f_s.t()
                    logits_per_text_s = logit_scale * text_f_s @ img_f_s.t()
                elif self.general:
                    logits_per_image_s = logit_scale * img_f_ @ text_f_s.t()
                    logits_per_text_s = logit_scale * text_f_s @ img_f_.t()

                if self.fg and self.general:
                    return logits_per_image_m, logits_per_text_m, logits_per_image_s, logits_per_text_s
                elif self.fg:
                    return logits_per_image_m, logits_per_text_m
                else:
                    return logits_per_image_s, logits_per_text_s


class CAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(CAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        batch_size_q = input_Q.size(0)
        batch_size_k = input_K.size(0)
        batch_size_v = input_V.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size_q, -1, self.n_heads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size_k, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size_v, -1, self.n_heads, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size_q, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        output = self.dropout(output)
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


if __name__ == "__main__":
    # import open_clip as clip
    import argparse
    parser = argparse.ArgumentParser(description='CLIP for Traffic Anomaly Detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--prompt_len', type=int, default=16, help='prompt_len')
    parser.add_argument('--accident_templates', nargs='+', type=str,
                        default=['The {} vehicle collision with another {}',
                                 'The {} vehicle out-of-control and leaving the roadway',
                                 'the {} vehicle has an unknown accident',
                                 'The vehicle is running normally on the road'])
    parser.add_argument('--accident_prompt', nargs='+', type=str,
                        default=['A traffic anomaly occurred in the scene', 'The traffic in this scenario is normal'])
    parser.add_argument('--accident_classes', nargs='+', type=str,
                        default=['ego', 'non-ego', 'vehicle', 'pedestrian', 'obstacle'])
    parser.add_argument('--ctx_init', type=str, default='a photo of a', help='ctx_init')
    parser.add_argument('--multi_class', action='store_true', help='multi class')
    parser.add_argument('--base_model', type=str, default='RN50', help='base model: RN50, ViT-B-16, ViT-B-32')
    args = parser.parse_args()
    clip_model, _, _ = clip.create_model_and_transforms(args.base_model, pretrained='openai', jit=False, cache_dir='./pretrain_models')
    tokenizer = clip.get_tokenizer(args.base_model)
    Model = TDAFF_BASE(args, clip_model, tokenizer)
    print("Turning off gradients in both the image and the text encoder")
    params = sum(p.numel() for p in Model.parameters() if p.requires_grad)
    a = torch.zeros([4, 3, 224, 224])
    b = torch.ones([4, 3, 224, 224])
    c = torch.zeros([4, 3, 224, 224])
    label = torch.tensor([1, 0, 0, 1])
    indices = np.random.permutation(a.shape[0] * 2)
    texf_m = ['The ego vehicle collision with another vehicle',
              'The ego vehicle collision with another vehicle',
              'The ego vehicle collision with another vehicle',
              'The ego vehicle collision with another vehicle']
    texf_s = ['A traffic anomaly occurred in the scene',
              'The traffic in this scenario is normal',
              'A traffic anomaly occurred in the scene',
              'A traffic anomaly occurred in the scene']
    output = Model(a, b)
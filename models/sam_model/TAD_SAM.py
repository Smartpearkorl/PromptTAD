# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
from typing import Optional, Tuple , Union

from .modeling import Sam
from .utils.transforms import ResizeLongestSide
from .build_sam import sam_model_registry


class SamTAD(nn.Module):
    def __init__(
        self,
        input_type, # 'rgb' or 'emb'
        decoder_type , # 'full' 'TAD_base' 'TAD_prompt' 
        sam_type, 
        checkpoint,
        return_ins_tokens = True,
    ) -> None:
        """
        Uses SAM to tokenize Image-Aware embedding and Intance-Aware embedding for an image
        and use bottle token to augment Intance-Aware embedding
        Arguments:
          sam_model (Sam): The model to use for mask prediction.
          bottle_aug (bool): If uses bottle token to augment Intance-Aware embedding , else just get mean

        """
        super().__init__()
        self.input_type = input_type # 'rgb' or 'emb'
        self.return_ins_tokens = return_ins_tokens
        self.model = sam_model_registry[sam_type](checkpoint=checkpoint, no_load_vit = input_type == 'emb' , decoder_type = decoder_type)
        self.transform = ResizeLongestSide(1024)
        self.reset_image()
      
        # prepare for mask-generation (for movad)
        # if self.input_type == 'emb':

        self.original_size = (720, 1280)
        self.input_size = (576, 1024)
        # self.is_image_set = True

    '''
    step1: Uses SAM to tokenize Image-Aware embedding and Intance-Aware embedding for an image
           return : image_emb shape = [1,C,H,W] , instance_token shape = [N_object,C], instance_emb shape = [N_object,C,H,W]
    step2: Combine instance level to object_tokens

    '''
    def forward(
        self,
        images: torch.Tensor,
        boxes_batch: list[np.ndarray] = None,
        multimask_output: bool = False,
    ):
        # Calculates the image embeddings
        if self.input_type == 'rgb':
          input_image = self.model.preprocess(images)
          self.features = self.model.image_encoder(input_image)
          self.is_image_set = True

        elif self.input_type == 'emb':
            self.features = images # [B,C,H,W]
        
        # base model (without promts)
        if not self.return_ins_tokens:
            return self.features, None , None 

        # need to loop in batch for building instnce-aware embedding 
        batch_ins_tokens , batch_ins_embs = [], []
        for frame_emb , frame_boxes  in zip(self.features,boxes_batch):  

          frame_boxes = self.transform.apply_boxes(frame_boxes, self.original_size)
          box_torch = torch.as_tensor(frame_boxes, dtype=torch.float, device=self.device)

          # Embed prompts
          sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=None, boxes=box_torch, masks=None)
          instance_tokens, instacne_embs = self.model.mask_decoder(
              image_embeddings=frame_emb.unsqueeze(dim=0),
              image_pe=self.model.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=multimask_output)

          batch_ins_tokens.append(instance_tokens)
          batch_ins_embs.append(instacne_embs)
        return  self.features, batch_ins_tokens , batch_ins_embs

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        if len(image.shape) == 3: 
          input_image = self.transform.apply_image(image)
          input_image_torch = torch.as_tensor(input_image, device=self.device)
          input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        elif len(image.shape) == 4:
            input_image_torch = []
            for frame in image:
                input_image = self.transform.apply_image(frame)
                input_image_torch.append(torch.as_tensor(input_image, device=self.device))
            input_image_torch = torch.stack(input_image_torch,dim=0)
            input_image_torch = input_image_torch.permute(0,3,1,2)

        self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    @torch.no_grad()
    def predict_torch(
        self,
        images:  Union[np.ndarray,torch.Tensor],
        boxes: Optional[torch.Tensor] ,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,  
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          images (np.ndarray or torch.Tensor):  np.ndarray for rgb , torch.Tensor for 
            vit embedding
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        # Calculates the image embeddings
        if self.input_type == 'rgb':
            input_image = self.model.preprocess(images)
            self.features = self.model.image_encoder(input_image)
            self.is_image_set = True

        elif self.input_type == 'emb':
            self.features = images
            self.is_image_set = True

        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            ruturn_mask = True
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None

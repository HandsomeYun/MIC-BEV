import numpy as np
import torch
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
from ..modules.bipartite_cam_bev_gat import BipartiteCamBEVGAT, compute_cam_bev_weights

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, embed_dims, pc_range=None, num_points_in_pillar=4, 
                 return_intermediate=False, dataset_type='nuscenes', num_cameras=4, 
                 with_gat=False, **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.with_gat = with_gat
        self.num_cameras = num_cameras
        if self.with_gat:
            self.cam_bev_gat = BipartiteCamBEVGAT(embed_dims, self.num_cameras)
        for layer in self.layers:
            for name, module in layer.named_modules():
                if 'TemporalSelfAttention' in str(type(module)):
                    for param in module.parameters():
                        param.requires_grad = False
                        
    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):
        """
        Project 3D reference points into each camera to get (u,v) pixel coords
        and build a visibility mask.

        Args:
            reference_points (Tensor): (bs, D, num_query, 3) normalized in [0,1].
            pc_range (list[float]): The 3D point cloud range to denormalize.
            img_metas (list[dict]): Each contains 'lidar2img' (N_cam, 4x4) per camera.

        Returns:
            reference_points_cam: (N_cam, bs, num_query, 2) normalized u,v in [0,1].
            bev_mask: (N_cam, bs, num_query, N_depth) boolean mask of visibility.
        """
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Extract lidar2img matrices and convert to Tensor: shape (B, N_cam, 4, 4)
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        # Denormalize x, y, z from [0,1] to actual coordinates in pc_range
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        # Update visibility mask: also require 0 <= u,v <= 1
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        
        # Fix any NaNs in mask (older PyTorch may not support nan_to_num)
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(self,
                bev_query,
                key, # feat_flatten
                value, #
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Either last-layer output (num_query, bs, embed_dims) or
            stack of intermediate outputs if return_intermediate=True.
        """

        # Start with the input BEV query
        output = bev_query
        intermediate = []
        
        # Clear camera validity masks from previous forward pass
        self.cam_validity_masks = []

        # 1) Generate 3D reference points for SCA: shape (bs, D, bev_h*bev_w, 3)
        ref_3d = self.get_reference_points(
            bev_h, bev_w,
            Z=self.pc_range[5] - self.pc_range[2],        # total Z range
            num_points_in_pillar=self.num_points_in_pillar,
            dim='3d',
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype
        )
        # 2) Generate 2D reference points for TSA: shape (bs, bev_h*bev_w, 1, 2)
        ref_2d = self.get_reference_points(
            bev_h, bev_w,
            dim='2d',
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype
        )

        # 3) Project 3D points into each camera; get pixel coords and a visibility mask
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas']
        )
        # collapse over the Z dimension so that each (cam, batch, query) is one bool:
        bev_mask_3d = bev_mask.any(dim=-1)  
        assert bev_mask_3d.ndim == 3
        
        # 3.5) Build camera tokens from `key` or `value` (multi-camera features) 
        #       and image metadata, then run GAT to get per-query per-camera logits.
        bev_query = bev_query.permute(1, 0, 2)  # (bs, num_query, embed_dims)
        img_feats = [key[c] for c in range(key.size(0))]    
        
        # Compute geometric information for GAT
        # Extract camera positions from img_metas
        cam_positions = []
        cam_yaw = []
        cam_pitch = [] 
        for img_meta in kwargs['img_metas']:
            # Extract camera positions from lidar2img transformation matrices
            # lidar2img transforms from lidar to camera coordinates
            # We need the inverse (camera to lidar) to get camera positions
            lidar2cam = torch.tensor(img_meta['lidar2cam'], device=bev_query.device, dtype=bev_query.dtype)
            num_cams = lidar2cam.size(0)
            
            # Compute camera positions by inverting the transformation
            cam_positions_batch = []
            cam_yaw_batch   = []   # list per batch → later stack to (B, N_cam)
            cam_pitch_batch = []
            for c in range(num_cams):
                E = lidar2cam[c]          # 4×4 pure extrinsic (LiDAR→CAM)
                R_ext = E[:3, :3]         # rotation
                t_ext = E[:3,  3]         # translation (in metres)
                # camera center in LiDAR frame:
                cam_pos = -R_ext.t() @ t_ext   # (x,y,z) in metres
                cam_pos[2] = cam_pos[2] + 1.5
                cam_positions_batch.append(cam_pos)
                
                R_c2w = R_ext.t()                  # Camera→World rotation  (inverse)
                fwd = R_c2w[:, 2]                  # camera optical axis (3,)
                # ---- yaw (heading) ----------------------------------------------------
                # angle between forward’s projection onto xy-plane and +X axis (East)
                yaw   = torch.atan2(fwd[1], fwd[0])          # rad, range [-π, π]
                
                # ---- pitch (tilt) -----------------------------------------------------
                # positive when looking *down* (because z-up world, z component negative)
                horiz_len = torch.norm(fwd[:2])               # √(fx²+fy²)
                pitch = torch.atan2(-fwd[2], horiz_len)       # rad
                
                cam_yaw_batch.append(yaw)
                cam_pitch_batch.append(pitch)
            
            cam_positions_batch = torch.stack(cam_positions_batch, dim=0)  # (N_cam, 3)
            cam_yaw_batch   = torch.stack(cam_yaw_batch)          # (N_cam,)
            cam_pitch_batch = torch.stack(cam_pitch_batch)        # (N_cam,)
            #==========pad===============================================
            desired_N = self.num_cameras #4
            actual_N = cam_positions_batch.size(0)
            
            # Create camera validity mask BEFORE padding
            cam_validity_mask = torch.ones(actual_N, device=cam_positions_batch.device, dtype=torch.bool)
            
            if actual_N < desired_N:
                pad = torch.zeros(desired_N - actual_N, 3,
                                device=cam_positions_batch.device,
                                dtype=cam_positions_batch.dtype)
                cam_positions_batch = torch.cat([cam_positions_batch, pad], dim=0)
                pad_yaw   = torch.zeros(desired_N - actual_N,
                        device=cam_yaw_batch.device,
                        dtype=cam_yaw_batch.dtype)
                cam_yaw_batch   = torch.cat([cam_yaw_batch, pad_yaw], 0)
                pad_pitch   = torch.zeros(desired_N - actual_N,
                        device=cam_pitch_batch.device,
                        dtype=cam_pitch_batch.dtype)
                cam_pitch_batch = torch.cat([cam_pitch_batch, pad_pitch], 0)
                # Extend validity mask with False for padded cameras
                pad_validity = torch.zeros(desired_N - actual_N, device=cam_validity_mask.device, dtype=torch.bool)
                cam_validity_mask = torch.cat([cam_validity_mask, pad_validity], dim=0)
            elif actual_N > desired_N:
                cam_positions_batch = cam_positions_batch[:desired_N]
                cam_yaw_batch   = cam_yaw_batch[:desired_N]
                cam_pitch_batch = cam_pitch_batch[:desired_N]
                cam_validity_mask = cam_validity_mask[:desired_N]
            # now cam_positions_batch is always (4,3) and cam_validity_mask is (4,)
            #=============================================
            cam_positions.append(cam_positions_batch)
            cam_yaw.append(cam_yaw_batch)    # inside your batch loop
            cam_pitch.append(cam_pitch_batch)
            # Store validity mask for this batch
            if not hasattr(self, 'cam_validity_masks'):
                self.cam_validity_masks = []
            self.cam_validity_masks.append(cam_validity_mask)
        
            
        cam_positions = torch.stack(cam_positions, dim=0)  # (B, N_cam, 3)
        cam_yaw   = torch.stack(cam_yaw,   dim=0)   # (B, N_cam)
        cam_pitch = torch.stack(cam_pitch, dim=0)   # (B, N_cam)
        cam_validity_masks = torch.stack(self.cam_validity_masks, dim=0)  # (B, N_cam)
        
        # Process each batch separately since compute_cam_bev_weights expects single batch
        if self.with_gat:
            cam_bev_weights_list = []
            # Build the grid coords ONCE, with explicit indexing
            grid_x, grid_y = torch.meshgrid(
                torch.arange(bev_w, device=bev_query.device, dtype=bev_query.dtype),
                torch.arange(bev_h, device=bev_query.device, dtype=bev_query.dtype)
            )
            bev_grid_coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (bev_h*bev_w,2)

            #Compute spans, resolution, min as tensors
            span = torch.tensor([
                self.pc_range[3] - self.pc_range[0],
                self.pc_range[4] - self.pc_range[1]
            ], device=bev_query.device, dtype=bev_query.dtype)           # (2,)

            resolution = span / torch.tensor([bev_w, bev_h],
                device=bev_query.device, dtype=bev_query.dtype)        # (2,)

            world_min = torch.tensor(self.pc_range[:2],
                device=bev_query.device, dtype=bev_query.dtype)        # (2,)
            
            for b in range(bev_query.size(0)):            
                batch_weights = compute_cam_bev_weights(
                    self.cam_bev_gat, 
                    bev_mask_3d[:, b:b+1],  # (N_cam, 1, num_bev)
                    bev_query[b:b+1],       # (1, num_bev, embed_dims)
                    [feat[:, b:b+1] for feat in img_feats],  # list of (num_tokens, 1, embed_dims)
                    cam_positions[b],       # (N_cam, 3)
                    cam_yaw[b],
                    cam_pitch[b],
                    bev_grid_coords, resolution, span, world_min
                )
                cam_bev_weights_list.append(batch_weights)

            # Combine batch results
            cam_bev_weights = torch.cat(cam_bev_weights_list, dim=0)  # (B, num_bev, N_cam)
            
            # Apply camera validity mask BEFORE normalization to ensure attention sums to 1 across all cameras
            # cam_validity_masks: (B, N_cam), cam_bev_weights: (B, num_bev, N_cam)
            cam_bev_weights = cam_bev_weights * cam_validity_masks.unsqueeze(1)  # Broadcast (B, N_cam) to (B, num_bev, N_cam)
            
            # Re-normalize after masking to ensure attention sums to 1 across all cameras (including padded ones)
            cam_bev_weights = cam_bev_weights / (cam_bev_weights.sum(-1, keepdim=True).clamp_min(1e-6))
            
            # Store weights for hook access
            self.cam_bev_weights = cam_bev_weights
        else:
            # fallback to equal weighting across all cams
            B, Q = bev_mask_3d.shape[1], bev_mask_3d.shape[2]
            Ncam = key.shape[0]
            cam_bev_weights = bev_mask_3d.float().permute(1,2,0)  # (B, Q, Ncam)
            
            # Apply camera validity mask BEFORE normalization to ensure attention sums to 1 across all cameras
            cam_bev_weights = cam_bev_weights * cam_validity_masks.unsqueeze(1)  # Broadcast (B, N_cam) to (B, Q, Ncam)
            
            # Re-normalize after masking to ensure attention sums to 1 across all cameras (including padded ones)
            cam_bev_weights = cam_bev_weights / (cam_bev_weights.sum(-1, keepdim=True).clamp_min(1e-6))
            
            # Store weights for hook access
            self.cam_bev_weights = cam_bev_weights
            # print(f"equal weights {cam_bev_weights}")
        # ---------------------------------------------------------------------------

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_pos = bev_pos.permute(1, 0, 2)

        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(bs*2, len_bev, num_bev_level, 2)
        else: #  If no previous BEV, duplicate current ref_2d twice
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(bs*2, len_bev, num_bev_level, 2)
        
        # 4) Iterate through each transformer layer in the sequence
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                cam_bev_weights=cam_bev_weights,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """
    Single BEVFormerLayer:  
    - TSA (temporal self-attention) over BEV queries and prev_bev  
    - FFN normalization & feed-forward  
    - SCA (spatial cross-attention) over multi-camera features  
    - Another FFN normalization & feed-forward  
    The operation order is defined by `operation_order` and must contain
    exactly ['self_attn','norm','cross_attn','ffn'] repeated appropriately.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        """
        Args:
            attn_cfgs (list[dict]): Configs for each attention block.
            feedforward_channels (int): Hidden dim of FFN.
            ffn_dropout (float): Dropout rate in FFN.
            operation_order (tuple[str]): Execution order, e.g. ('self_attn','norm','cross_attn','ffn','norm','ffn').
            act_cfg, norm_cfg, ffn_num_fcs: standard FFN configs.
        """

        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                cam_bev_weights=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    cam_bev_weights=cam_bev_weights,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query




from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention


@TRANSFORMER_LAYER.register_module()
class MM_BEVFormerLayer(MyCustomBaseTransformerLayer):
    """multi-modality fusion layer.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 lidar_cross_attn_layer=None,
                 **kwargs):
        super(MM_BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.cross_model_weights = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True) 
        if lidar_cross_attn_layer:
            self.lidar_cross_attn_layer = build_attention(lidar_cross_attn_layer)
            # self.cross_model_weights+=1
        else:
            self.lidar_cross_attn_layer = None


    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                debug=False,
                depth=None,
                depth_z=None,
                lidar_bev=None,
                radar_bev=None,
                cam_bev_weights=None, 
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    lidar_bev=lidar_bev,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                new_query1 = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    depth=depth,
                    lidar_bev=lidar_bev,
                    depth_z=depth_z,
                    **kwargs)

                if self.lidar_cross_attn_layer:
                    bs = query.size(0)
                    new_query2 = self.lidar_cross_attn_layer(
                        query,
                        lidar_bev,
                        lidar_bev,
                        reference_points=ref_2d[bs:],
                        spatial_shapes=torch.tensor(
                            [[bev_h, bev_w]], device=query.device),
                        level_start_index=torch.tensor([0], device=query.device),
                        )
                query = new_query1 * self.cross_model_weights + (1-self.cross_model_weights) * new_query2
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
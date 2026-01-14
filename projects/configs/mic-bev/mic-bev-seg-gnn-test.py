_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# -----------------------------------------------------------------------------
point_cloud_range = [-51.2, -51.2, -7.0, 51.2, 51.2, 1.0]
post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
voxel_size = [0.2, 0.2, 8]

class_names = [
    'car', 'bicycle', 'truck', 'pedestrian'
]
num_classes=len(class_names)
max_cam = 4
#========MAP=======
with_seg=True #Whether to predict map
# BEV‐SEMANTIC MAP CLASSES 
sem_seg_classes = [
    'background', 'driving', 'sidewalk', 'crosswalk',
    'shoulder', 'border', 'parking'
    ]
num_map_classes = len(sem_seg_classes)
#========GAT========
with_gat=True
#===================
file_client_args = dict(backend='disk')
dataset_type = 'M2IDataset'

# # Set dataset roots before running
data_root = '/data4/yun/M2I_dataset/M2I_split_dataset'
pkl_root = '/data4/yun/M2I_dataset/M2I_pkl/'
# for absolute path remapping. If you move the dataset to a different location, you need to change this.
path_prefix_replace = [('/data3/', '/data4/')] 

ann_file_train=pkl_root + "/v2xset_infos_temporal_train.pkl"
ann_file_val=pkl_root + "/v2xset_infos_temporal_val.pkl"
ann_file_test=pkl_root + "/v2xset_infos_temporal_test.pkl"
total_epochs = 5

eval_cfg = {
    "dist_ths": [0.5, 1.0, 2.0, 4.0],
    "dist_th_tp": 2.0,
    "min_recall": 0.1,
    "min_precision": 0.1,
    "mean_ap_weight": 5,
    "class_names": class_names,
    "tp_metrics":['trans_err', 'scale_err', 'orient_err', 'vel_err'],
    "err_name_maping":{'trans_err': 'mATE','scale_err': 'mASE','orient_err': 'mAOE','vel_err': 'mAVE','attr_err': 'mAAE'},
    "class_range":{'car':(50,50), 'truck':(50,50),'bicycle':(40,40),'pedestrian':(40,40)}
    }
# ----------

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 1 # each sequence contains `queue_length` frames.

model = dict(
    type='MICBEV',
    use_grid_mask=True,
    video_test_mode=False,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='MICBEVHead',
        with_seg=with_seg, 
        with_gat=with_gat,
        num_cams=max_cam,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=200,
        num_classes=num_classes,
        num_map_classes = num_map_classes, 
            loss_seg_cfg = dict( 
            type='CrossEntropyLoss',
            loss_weight=2.0,
            ignore_index=255),
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=False,
            use_shift=False,
            use_can_bus=False,
            num_cams=max_cam,
            embed_dims=_dim_,
            encoder=dict(
                type='BEVFormerEncoder',
                embed_dims=256,
                num_layers=6,
                num_cameras=max_cam,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                with_gat=with_gat,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            num_cams = max_cam, 
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=post_center_range,
            pc_range=point_cloud_range,
            max_num=100,
            voxel_size=voxel_size,
            num_classes=len(class_names)),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0), #Stable gradient
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True, max_views = max_cam),
    dict(
        type='LoadBEVSegFromFile',
        file_client_args=dict(backend='disk')
    ),
    dict(type='RandomMaskMultiView', mask_prob = 0.25, blur_kernel_size = 11, blur_sigma=(3.0, 10.0), deterministic=False),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='GetBEVObjLabel', bev_h=bev_h_, bev_w=bev_w_, pc_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True, max_views = max_cam),
    dict(
        type='LoadBEVSegFromFile',
        file_client_args=dict(backend='disk')
    ),
    dict(type = 'RandomMaskMultiView', mask_prob = 1, blur_kernel_size = 11, blur_sigma=(10.0, 10.0), deterministic=True), #Tested for robust, uncomment for normal
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(800, 600),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D',  keys=['img']) #Load map
        ])
]


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        path_prefix_replace=path_prefix_replace,
        ann_file= ann_file_train,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # use_valid_flag=True,
        use_valid_flag=False,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        pc_range = point_cloud_range,
        eval_cfg=eval_cfg,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             path_prefix_replace=path_prefix_replace,
             ann_file=ann_file_val,
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             eval_cfg=eval_cfg,
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              path_prefix_replace=path_prefix_replace,
              ann_file=ann_file_test,
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              eval_cfg=eval_cfg,
              classes=class_names, modality=input_modality, test_mode=False,),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

#learning policy
lr_config = dict(
    by_epoch=False,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-3)

# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict( 
    type='OptimizerHook',  # or Fp16OptimizerHook if using fp16
    grad_clip=dict(max_norm=1, norm_type=2),
)


evaluation = dict(
    interval=10,
    pipeline=test_pipeline,
)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
checkpoint_config = dict(interval=1, max_keep_ckpts=3,  
                         by_epoch=True)
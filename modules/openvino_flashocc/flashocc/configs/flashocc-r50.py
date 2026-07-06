data_config = {
    "cams": [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
    ],
    "Ncams": 6,
    "input_size": (256, 704),
    "src_size": (900, 1600),
    "resize": (-0.06, 0.11),
    "rot": (-5.4, 5.4),
    "flip": True,
    "crop_h": (0.0, 0.0),
    "resize_test": 0.00,
}

grid_config = {
    "x": [-40, 40, 0.4],
    "y": [-40, 40, 0.4],
    "z": [-1, 5.4, 6.4],
    "depth": [1.0, 45.0, 0.5],
}

numC_Trans = 64

model = dict(
    type="BEVDetOCC",
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style="pytorch",
        pretrained=None,
    ),
    img_neck=dict(
        type="CustomFPN",
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0],
    ),
    img_view_transformer=dict(
        type="LSSViewTransformer",
        grid_config=grid_config,
        input_size=data_config["input_size"],
        in_channels=256,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=True,
        downsample=16,
    ),
    img_bev_encoder_backbone=dict(
        type="CustomResNet",
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8],
    ),
    img_bev_encoder_neck=dict(
        type="FPN_LSS",
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256,
    ),
    occ_head=dict(
        type="BEVOCCHead2D",
        in_dim=256,
        out_dim=256,
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_balance=False,
        loss_occ=dict(type="CrossEntropyLoss", use_sigmoid=False, ignore_index=255, loss_weight=1.0),
    ),
)

train_args = dict(
    # model parameter
    in_channels=3,
    num_class=4,
    num_joints=14,
    graph_args={'layout': 'noeye_ear_coco', 'strategy': 'distance'},
    edge_importance_weighting=True,

    # training parameter
    device='cuda:0',
    trainset_path='/root/autodl-tmp/wmh/sgcn/data/escalator_dataset/labels/aug_trainset.json',
    valset_path='/root/autodl-tmp/wmh/sgcn/data/escalator_dataset/labels/val.json',
    log_dir='log/',
    weights_dir='weights/',
    lr=0.1,
    batch_size=32,
    total_epoch=200,
    steps=[50, 100, 150, 180],
    save_best=True,
)

test_args = dict(
    # model parameter
    in_channels=3,
    num_class=4,
    num_joints=14,
    graph_args={'layout': 'noeye_ear_coco', 'strategy': 'distance'},
    edge_importance_weighting=True,

    # test parameter
    device='cuda:0',
    testset_path='/root/autodl-tmp/wmh/sgcn/data/escalator_dataset/labels/test.json',
    log_dir='log/',
    weights_path='weights/sgcn_4_layer_distance_V1.pt',
    batch_size=32,
)



# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer = dict(
    type='AdamW', lr=2e-05, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, 
    clip_grad=dict(max_norm=0.01, norm_type=2))

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-8,
        power=0.9,
        begin=0,
        end=160000,
        by_epoch=False)
]
# training schedule for 90k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

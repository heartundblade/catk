from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiplicativeLR, CosineAnnealingLR

def CosineAnnealingWarmUpRestarts(optimizer, epochs, warm_up_epoch, start_factor=0.1, eta_min=1e-6):
    assert epochs >= warm_up_epoch
    T_warmup = warm_up_epoch
    
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warm_up_epoch - 1)
    # fixed_scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warm_up_epoch,
        eta_min=eta_min,
    )

    scheduler = SequentialLR(optimizer, 
                             schedulers=[warmup_scheduler, cosine_scheduler], 
                             milestones=[T_warmup])
    
    return scheduler
# 1. setup and training v2 = 7-8 hours of training (1 epoch, 19M config)
# 2. prepare supervised training dataset + preprocessing pipeline
# 3. setup multiple inference strategy (argmax & top-k, idk else)
# 4. improve model  (flash attention, MoE, simpler FFN, hyperparams)
# 5. save epoch metrics to checkpoint, make checkpoint more verbose for large model, save scheduler state, remove old callbacks (basically new checkpoint callbacks)
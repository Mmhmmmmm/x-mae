CUDA_VISIBLE_DEVICES=1 python main_finetune.py \
--data_path ../few-shot-data/pokemon \
--type codebook \
--resume ../pretrain_mae/mae_pretrain_vit_base_full.pth  \
--output_dir ../MAE_FT/Pokemon_Codebook_5 \
--blr 1e-4 \
# --resume_add ../pretrain/vit-b-300ep.pth.tar \

# --start_epoch 300 \

# --norm_pix_loss \
# --blr 1e-4 \
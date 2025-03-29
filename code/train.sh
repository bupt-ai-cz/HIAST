setting_file=configs/hiast_setting.yaml
work_dir=../log/gtav-to-citiscapes/hiast

# sl_1
# generate pseudo labels with pseudo_resume_from.pth and resume_from.pth from warmup phase
python generate_pseudo_labels.py \
    --config_file configs/sl_1.yaml \
    --pseudo_resume_from ../pretrained/gtav-to-cityscapes/pseudo_resume_from.pth \
    --pseudo_save_dir $work_dir/sl_1/pseudo_label/gray_label

python train.py \
    --config_file configs/sl_1.yaml \
    --setting_file $setting_file \
    --resume_from ../pretrained/gtav-to-cityscapes/resume_from.pth \
    --pseudo_save_dir $work_dir/sl_1/pseudo_label/gray_label \
    --work_dir $work_dir/sl_1

# sl_2
# generate pseudo labels with ema_model_last.pth of momentum model from self-training phase 1
python generate_pseudo_labels.py \
    --config_file configs/sl_2.yaml \
    --pseudo_resume_from $work_dir/sl_1/checkpoints/ema_model_last.pth \
    --pseudo_save_dir $work_dir/sl_2/pseudo_label/gray_label

python train.py \
    --config_file configs/sl_2.yaml \
    --setting_file $setting_file \
    --resume_from $work_dir/sl_1/checkpoints/model_last.pth \
    --pseudo_save_dir $work_dir/sl_2/pseudo_label/gray_label \
    --work_dir $work_dir/sl_2

# sl_3
# generate pseudo labels with ema_model_last.pth of momentum model from self-training phase 2
python generate_pseudo_labels.py \
    --config_file configs/sl_3.yaml \
    --pseudo_resume_from $work_dir/sl_2/checkpoints/ema_model_last.pth \
    --pseudo_save_dir $work_dir/sl_3/pseudo_label/gray_label

python train.py \
    --config_file configs/sl_3.yaml \
    --setting_file $setting_file \
    --resume_from $work_dir/sl_2/checkpoints/model_last.pth \
    --pseudo_save_dir $work_dir/sl_3/pseudo_label/gray_label \
    --work_dir $work_dir/sl_3
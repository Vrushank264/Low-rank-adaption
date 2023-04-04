export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export INSTANCE_DIR="/content/data/ajxd"
export OUTPUT_DIR="../exps"

#pip install -r ../requirements.txt

accelerate launch ../lora_diffusion/cli_lora_pti.py train 
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --train_inpainting \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --scale_lr \
  --learning_rate_unet=2e-4 \
  --learning_rate_text=1e-6 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --lr_scheduler_lora="constant" \
  --lr_warmup_steps_lora=100 \
  --placeholder_tokens="ajxd" \
  --save_steps=100 \
  --max_train_steps_ti=3000 \
  --max_train_steps_tuning=3000 \
  --perform_inversion=True \
  --clip_ti_decay \
  --weight_decay_ti=0.000 \
  --weight_decay_lora=0.000 \
  --device="cuda:0" \
  --lora_rank=8 \
  --use_face_segmentation_condition \
  --lora_dropout_p=0.1 \
  --lora_scale=2.0 \
  --log_wandb

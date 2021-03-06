python ./run_readmission.py \
  --task_name readmission \
  --do_train \
  --do_eval \
  --data_dir ./discharge \
  --bert_model ./model/pretraining \
  --max_seq_length 512 \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir ./result_new

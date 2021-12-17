# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
export DATA_DIR="data/"
export OUTPUT_DIR="test/"

# make predictions on test set for a model trained with Pytorch DDP
python task.py --append_descr 1 --append_triples --append_retrieval 1 --data_version csqa_ret_3datasets --append_answer_text 1 --model_type electra --batch_size 1 --max_seq_length 512 --vary_segment_id --bert_model_dir test/ --mission output --predict_dir $OUTPUT_DIR/prediction/ --pred_file_name pred_test.csv --bert_vocab_dir google/electra-large-discriminator


# make predictions on test set for a model trained with DeepSpeed
deepspeed --include="localhost:0" task.py --append_descr 1 --append_triples --append_retrieval 1 --data_version csqa_ret_3datasets --append_answer_text 1 --model_type debertav2 --batch_size 1 --max_seq_length 512 --vary_segment_id --ddp --deepspeed --bert_model_dir test/ --predict_dir $OUTPUT_DIR/prediction/ --pred_file_name pred_test.csv --mission output --bert_vocab_dir microsoft/deberta-v2-xxlarge --deepspeed_config debertav2-test
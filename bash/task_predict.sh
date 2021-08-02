# Example predict script

export DATA_DIR="data/"
export OUTPUT_DIR="output/"

python task.py\
 --batch_size 8\
 --pred_file_name pred_1.csv\
 --output_model_dir output/\
 --bert_model_dir output/model/\
 --append_descr 1\
 --max_seq_length 192\
 --append_answer_text 1\
 --mission output\
 --predict_dev


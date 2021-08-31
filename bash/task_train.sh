# Example training script

export DATA_DIR="data/"
export OUTPUT_DIR="output/"


python task.py --append_answer_text 1 --batch_size 8 --append_descr 1 --max_seq_length 192 --lr 1e-5 --weight_decay 0.15


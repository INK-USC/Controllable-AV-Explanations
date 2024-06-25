# CAVE - (C)ontrollable (A)uthorship (V)erification (E)xplanations
This is the official code and associated datasets for the paper titled

>[CAVE: Controllable Authorship Verification Explanations. *Sahana Ramnath, Kartik Pandey, Elizabeth Boschee, Xiang Ren.*](https://arxiv.org/abs/2406.16672)

## Dataset 
We provide our train/val/test datasplits in folder ``data/[dataset-name]``. The ``instruct-llama-3-8b`` and ``gpt-4-turbo`` subfolders have the sampled test-set responses from [Llama-3-8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and GPT-4-Turbo respectively. 

## Training commands
``cd llama3``
* to train a single-dataset CAVE:
``python train_lora.py \
--data_path "../data/imdb62/train_i2ro.csv" \
--train_batch_size 1 \
--epochs 10 \
--lora_r 128 \
--lora_alpha 256 \
--output_dir save_models`` 

* to train a multi-dataset CAVE:
``python train_lora_combined.py \
--data_path "../data/imdb62/train_i2ro.csv,../data/blog-auth/train_i2ro.csv,../data/pan20-fanfic/train_i2ro.csv" \
--train_batch_size 1 \
--epochs 10 \
--lora_r 128 \
--lora_alpha 256 \
--output_dir save_models`` 

## Inference commands
* The command below runs the model on the test set and saves the generations as a csv file in the checkpoint folder itself. The prefix argument can be used as a naming convention to identify which dataset's test set is evaluated. The csv file is best opened with pandas to ensure that the document structure and answer structure is retained. 
``python inference_llama3.py --model_path /path/to/model/checkpoint --dataset-val ../data/imdb62/test_i2ro.csv --do_val 0 --prefix="imdb_"``

* To obtain metrics ``ACCURACY`` and ``CONSISTENCY`` (as defined in the paper), and to obtain a csv file that can be opened with Excel/Google Sheets for easy analysis:
``python check_metrics.py --pred_data /path/to/csv/file --gold_data ../data/imdb62/test_i2ro.csv --human /path/to/human-eval/csv``


# should execute pre-training in advance. Also be careful layer_number
# 1) BERT *************************
CUDA_VISIBLE_DEVICE=7 python3 run_glue.py --data_dir='../dataset/STS-B/MSRvid' 
                                              --model_type='bert' 
                                              --model_name_or_path='bert-base-uncased' 
                                              --task_name='STS-B' 
                                              --output_dir='../pretrain_MSRvid_bert' 
                                              --do_train --do_eval

CUDA_VISIBLE_DEVICE=7 python3 run_expected.py --data_dir='../dataset/STS-B/Images' 
                                              --model_type='bert' 
                                              --model_name_or_path='bert-base-uncased' 
                                              --task_name='STS-B' 
                                              --output_dir='../expected_Images_bert' 
                                              --do_train --do_eval

CUDA_VISIBLE_DEVICE=7 python3 run_finetuning.py --data_dir='../dataset/STS-B/Images' 
                                              --model_type='bert' 
                                              --model_name_or_path='bert-base-uncased' 
                                              --task_name='STS-B' 
                                              --output_dir='../finetune_Images_bert' 
                                              --do_train --do_eval                                             
# 2) Roberta **********************
CUDA_VISIBLE_DEVICE=7 python3 run_glue.py --data_dir='../dataset/STS-B/MSRvid'
                                              --model_type='roberta' 
                                              --model_name_or_path='roberta-base' 
                                              --task_name='STS-B' 
                                              --output_dir='../pretrain_MSRvid_roberta' 
                                              --do_train --do_eval

CUDA_VISIBLE_DEVICE=7 python3 run_expected.py --data_dir='../dataset/STS-B/Images'
                                              --model_type='roberta' 
                                              --model_name_or_path='roberta-base' 
                                              --task_name='STS-B' 
                                              --output_dir='../expected_Images_roberta' 
                                              --do_train --do_eval                                              

CUDA_VISIBLE_DEVICE=7 python3 run_finetuning.py --data_dir='../dataset/STS-B/Images'
                                              --model_type='roberta' 
                                              --model_name_or_path='roberta-base' 
                                              --task_name='STS-B' 
                                              --output_dir='../finetune_Images_roberta' 
                                              --do_train --do_eval 
# 3) Distilbert *******************
CUDA_VISIBLE_DEVICE=7 python3 run_glue.py --data_dir='../dataset/STS-B/MSRvid'
                                          --model_type='distilbert' 
                                          --model_name_or_path='distilbert-base-uncased' 
                                          --task_name='STS-B' 
                                          --output_dir='../pretrain_MSRvid_distilbert' 
                                          --do_train --do_eval

CUDA_VISIBLE_DEVICE=7 python3 run_expected.py --data_dir='../dataset/STS-B/Images'
                                              --model_type='distilbert' 
                                              --model_name_or_path='distilbert-base-uncased' 
                                              --task_name='STS-B' 
                                              --output_dir='../expected_Images_distilbert' 
                                              --do_train --do_eval 

CUDA_VISIBLE_DEVICE=7 python3 run_finetuning.py --data_dir='../dataset/STS-B/Images'
                                              --model_type='distilbert' 
                                              --model_name_or_path='distilbert-base-uncased' 
                                              --task_name='STS-B' 
                                              --output_dir='../finetune_Images_distilbert' 
                                              --do_train --do_eval   
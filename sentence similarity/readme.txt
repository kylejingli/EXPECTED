This group of experiments are implemented based on the work "Pretrained Transformers Improve Out-of-Distribution Robustness" (ACl-20), which requires Python 3+ and PyTorch 1.0+.

We tested our model on sts-b dataset with BERT, RoBERTa, and DistilBERT being examined. One can reproduce our experimental results by starting with /src/myrun.sh. Pretrain and fine-tune on are recorded with checkpoint-350 while expected is with checkpoint-5000.


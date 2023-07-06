This experiment is implemented based on pycls package (https://github.com/facebookresearch/pycls).

We do not put all the dependent files here as they are not our contributions. What we mainly change is to modify the trainer.py under the path:
 
/pycls/core/trainer.py (see the presented one)

You may also need to create a configs file which is typically under the path of /pycls/core/ formatted with yaml.

Please consider amending meter.py if you wish to reproduce Fault-intolerant Evaluation (Section 5.3.2 of our paper) by using top-k error.

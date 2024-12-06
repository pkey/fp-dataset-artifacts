### Setting up locally

```bash
make env
make activate
make install
```

🦑 HIGHLY RECOMMENDING USING WANDB see all execution graphs 🦑 
```bash
wandb login
```

### Running evaluations & training

```bash

# EVAL squad on baseline model
make eval-squad-on-baseline

# IF YOU HAVE A MANUAL SQUAD 1.0 FORMAT DATASET FOR EVAL/TRAINING
# - PUT IT TO /datasets/{EXPERIMENT_NAME}/{training | validation}.json

# EVAL custom on baseline model (validation.json)
make eval-custom-on-baseline EXPERIMENT_NAME='example'

# TRAIN custom on baseline model (training.json)
make train-custom-on-baseline EXPERIMENT_NAME='example'

# EVAL custom on trained-baseline model (validation.json)
make eval-custom-on-trained-baseline EXPERIMENT_NAME='example'
```

#### Helper commands

```bash
# If you have a dataset with only CONTEXT + ANSWER and want to convert to SQUAD 1.0
# - PUT IT TO /datasets/{EXPERIMENT_NAME}/convert_test.json (will create a test.json in same folder)
python3 scripts/to_squad_format.py EXPERIMENT_NAME='example'

# If you have a big dataset and want to split automatically to 70:30 training/validation split
# - PUT IT TO /datasets/{EXPERIMENT_NAME}/test.json
python3 scripts/split_dataset.py EXPERIMENT_NAME='example'

# If you want to generate some questions for EXPERIMENT with AI 
python3 scripts/human_in_the_loop.py EXPERIMENT_NAME='human_in_the_loop'
```
 
#### Other dataset commands 
```bash 
make eval-bert-on-baseline
make eval-bidaf-on-baseline
make eval-roberta-on-baseline

make eval-squad_v2-on-baseline (added support)
make eval-hotpot-on-baseline (added suport)
```
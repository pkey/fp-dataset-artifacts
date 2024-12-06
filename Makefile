.PHONY: env activate install install/colab

ifneq (,$(wildcard .env))
    include .env
    export $(shell sed 's/=.*//' .env)
endif


# -- SCRIPTS FOR LOCAL SETUP
env:
	python3 -m venv venv

activate:
	source ./venv/bin/activate

install:
	pip3 install -r requirements.txt


# -- SCRIPTS FOR GOOGLE COLAB SETUP
install/colab:
	apt install python3.10-venv
	python -m venv tf-venv
	. tf-venv/bin/activate

	pip install --upgrade pip
	pip install -r requirements.txt


# -- SCRIPTS FOR TRAINING BASE MODELS

# POSSIBLE OPTIONS: small / base / large
ELECTRA_TYPE = small
BASE_MODEL_PATH = "$(MODEL_TRAINING_PATH)/train/electra_$(ELECTRA_TYPE)-squad/"

train-electra-squad:
	python3 run.py --do_train --task qa --dataset squad --output_dir $(BASE_MODEL_PATH) --per_device_train_batch_size $(PER_DEVICE_BATCH)

train-electra-squad_v2:
	python3 run.py --do_train --task qa --dataset squad_v2 --output_dir "$(MODEL_TRAINING_PATH)/train/electra_$(ELECTRA_TYPE)-squad_v2/" --per_device_train_batch_size $(PER_DEVICE_BATCH)


# -- SCRIPTS FOR TRAINING/EVALUATING EXPERIMENTS
eval-squad-on-baseline:
ifeq ($(strip $(EXPERIMENT_NAME)),)
	python3 run.py --do_eval --task qa \
		--dataset squad \
		--model $(BASE_MODEL_PATH) \
		--output_dir "$(MODEL_TRAINING_PATH)/eval/electra_$(ELECTRA_TYPE)-squad-ON-SQUAD"
else
	python3 run.py --do_eval --task qa \
		--dataset squad \
		--model "$(MODEL_TRAINING_PATH)/train/electra_$(ELECTRA_TYPE)-squad-$(EXPERIMENT_NAME)" \
		--output_dir "$(MODEL_TRAINING_PATH)/eval/electra_$(ELECTRA_TYPE)-squad+custom-ON_SQUAD-$(EXPERIMENT_NAME)"
endif

eval-custom-on-baseline:
ifeq ($(strip $(EXPERIMENT_NAME)),)
	$(error EXPERIMENT_NAME is required. Please provide it using 'make eval-custom-on-baseline EXPERIMENT_NAME=your_experiment_name')
else
	python3 run.py --do_eval --task qa \
		--dataset "$(LOCAL_DATASET_PATH)/$(EXPERIMENT_NAME)/validation.json" \
		--model $(BASE_MODEL_PATH) \
		--output_dir "$(MODEL_TRAINING_PATH)/eval/electra_$(ELECTRA_TYPE)-squad-ON_CUSTOM-$(EXPERIMENT_NAME)"
endif

train-custom-on-baseline:
ifeq ($(strip $(EXPERIMENT_NAME)),)
	$(error EXPERIMENT_NAME is required. Please provide it using 'make train-dataset-on-baseline EXPERIMENT_NAME=your_experiment_name')
else
	python3 run.py --do_train --task qa \
		--dataset "$(LOCAL_DATASET_PATH)/$(EXPERIMENT_NAME)/training.json" \
		--model $(BASE_MODEL_PATH) \
		--output_dir "$(MODEL_TRAINING_PATH)/train/electra_$(ELECTRA_TYPE)-squad_WITH_CUSTOM-$(EXPERIMENT_NAME)"
endif

eval-custom-on-trained-baseline:
ifeq ($(strip $(EXPERIMENT_NAME)),)
	$(error EXPERIMENT_NAME is required. Please provide it using 'make train-dataset-on-baseline EXPERIMENT_NAME=your_experiment_name')
else
	python3 run.py --do_eval --task qa \
		--dataset "$(LOCAL_DATASET_PATH)/$(EXPERIMENT_NAME)/validation.json" \
		--model "$(MODEL_TRAINING_PATH)/train/electra_$(ELECTRA_TYPE)-squad_WITH_CUSTOM-$(EXPERIMENT_NAME)" \
		--output_dir "$(MODEL_TRAINING_PATH)/eval/electra_$(ELECTRA_TYPE)-squad+WITH_CUSTOM-ON_CUSTOM-$(EXPERIMENT_NAME)"
endif


# -- SCRIPTS FOR OTHER DATASET RUNS
eval-squad_v2-on-baseline:
	python3 run.py --do_eval --task qa --dataset squad_v2 --model "$(MODEL_TRAINING_PATH)/train/electra_$(ELECTRA_TYPE)-squad_v2/" --output_dir "$(MODEL_TRAINING_PATH)/eval_output_squad_v2/"

eval-hotpot-on-baseline:
	python3 run.py --do_eval --task qa --dataset hotpot_qa:distractor --model $(BASE_MODEL_PATH) --output_dir "$(MODEL_TRAINING_PATH)/eval_output_hotpot/"

eval-bert-on-baseline:
	python3 run.py --do_eval --task qa --dataset adversarial_qa:dbert --model $(BASE_MODEL_PATH) --output_dir "$(MODEL_TRAINING_PATH)/eval-electra_$(ELECTRA_TYPE)-squad-ON-DBERT/"

eval-bidaf-on-baseline:
	python3 run.py --do_eval --task qa --dataset adversarial_qa:dbidaf --model $(BASE_MODEL_PATH) --output_dir "$(MODEL_TRAINING_PATH)/eval-electra_$(ELECTRA_TYPE)-squad-ON-DBIDAF/"

eval-roberta-on-baseline:
	python3 run.py --do_eval --task qa --dataset adversarial_qa:droberta --model $(BASE_MODEL_PATH) --output_dir "$(MODEL_TRAINING_PATH)/eval-electra_$(ELECTRA_TYPE)-squad-ON-DROBERTA/"

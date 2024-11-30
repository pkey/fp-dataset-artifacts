
ifneq (,$(wildcard .env))
    include .env
    export $(shell sed 's/=.*//' .env)
endif


.PHONY: install
install:
	pip3 install -r requirements.txt

.PHONY: env
env:
	python3 -m venv venv

.PHONE: activate
activate:
	source ./venv/bin/activate

.PHONY: initialise/colab
initialise/colab:
	apt install python3.10-venv
	python -m venv tf-venv
	. tf-venv/bin/activate

	pip install --upgrade pip
	pip install -r requirements.txt

.PHONY: initialise/local
initialise/local:
	pre-commit install

train-squad:
	python3 run.py --do_train --task qa --dataset squad --output_dir "$(MODEL_TRAINING_PATH)/trained_model_squad/" --per_device_train_batch_size $(PER_DEVICE_BATCH)

train-squad-v2:
	python3 run.py --do_train --task qa --dataset squad_v2 --output_dir "$(MODEL_TRAINING_PATH)/trained_model_squad_v2/" --per_device_train_batch_size $(PER_DEVICE_BATCH)

eval-squad:
	python3 run.py --do_eval --task qa --dataset squad --model "$(MODEL_TRAINING_PATH)/trained_model_squad/" --output_dir "$(MODEL_TRAINING_PATH)/eval_output_squad/"

# make evall-squad EXPERIMENT_NAME=when_experiment -- {'eval_exact_match': 40.0, 'eval_f1': 80.88857057974704}
# make evall-squad EXPERIMENT_NAME=when_experiment_eval_ai (AI) -- {'eval_exact_match': 20.253164556962027, 'eval_f1': 42.212620060397576}
# make evall-squad EXPERIMENT_NAME=when_experiment_eval_good -- {'eval_exact_match': 87.0, 'eval_f1': 95.21391591724408}
# make evall-squad EXPERIMENT_NAME=when_experiment_eval_brother -- {'eval_exact_match': 7.826086956521739, 'eval_f1': 39.18075661059444}
evall-squad:
ifeq ($(strip $(EXPERIMENT_NAME)),)
	$(error EXPERIMENT_NAME is required. Please provide it using 'make evall-squad EXPERIMENT_NAME=your_experiment_name')
endif
	python3 ./scripts/convert_to_dataset_format.py $(EXPERIMENT_NAME)
	python3 run.py --do_eval --task qa --dataset "$(LOCAL_DATASET_PATH)/$(EXPERIMENT_NAME).json" --model "$(MODEL_TRAINING_PATH)/trained_model_electra_small_squad/" --output_dir "$(MODEL_TRAINING_PATH)/eval_output_trained_model_electra_small_squad_$(EXPERIMENT_NAME)/"

# NOTE: can be only run on squad_v2
eval-squad-v2:
	python3 run.py --do_eval --task qa --dataset squad_v2 --model "$(MODEL_TRAINING_PATH)/trained_model_squad_v2/" --output_dir "$(MODEL_TRAINING_PATH)/eval_output_squad_v2/"

# --- PROBLEMATIC datasets with electra due to weird results ---
eval-hotpot:
	python3 run.py --do_eval --task qa --dataset hotpot_qa:distractor --model "$(MODEL_TRAINING_PATH)/trained_model_squad/" --output_dir "$(MODEL_TRAINING_PATH)/eval_output_hotpot/"

# NOTE: format is of squad_v1
eval-adversarial-qa:
	python3 run.py --do_eval --task qa --dataset adversarial_qa:adversarialQA --model "$(MODEL_TRAINING_PATH)/trained_model_squad/" --output_dir "$(MODEL_TRAINING_PATH)/eval_output_adversarial/"

ifneq (,$(wildcard .env))
    include .env
    export $(shell sed 's/=.*//' .env)
endif

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

train-squad/colab:
	python3 run.py --do_train --task qa --dataset squad --output_dir "$(TRAIN_PATH)/trained_model_squad/" --per_device_train_batch_size 60

eval-hotpot:
	python3 run.py --do_eval --task qa --dataset hotpot_qa:distractor --model "$(TRAIN_PATH)/trained_model_squad/" --output_dir "$(TRAIN_PATH)/eval_output_hotpot/"

eval-adversarial_qa:
	python3 run.py --do_eval --task qa --dataset adversarial_qa:adversarialQA --model "$(TRAIN_PATH)/trained_model_squad/" --output_dir "$(TRAIN_PATH)/eval_output_adversarial/"

eval-squad:
	python3 run.py --do_eval --task qa --dataset squad --model "$(TRAIN_PATH)/trained_model_squad/" --output_dir "$(TRAIN_PATH)/eval_output_squad/"

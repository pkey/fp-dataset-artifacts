
ifneq (,$(wildcard .env))
    include .env
    export $(shell sed 's/=.*//' .env)
endif

initialise/colab:
	apt install python3.10-venv
	python -m venv tf-venv
	. tf-venv/bin/activate

	pip install --upgrade pip
	pip install -r requirements.txt

train-nli/colab:
	python3 run.py --do_train --task nli --dataset snli --output_dir "$(TRAIN_PATH)/trained_model_nli/" --save_steps 25000 --per_device_train_batch_size 400

train-squad/colab:
	python3 run.py --do_train --task qa --dataset squad --output_dir "$(TRAIN_PATH)/trained_model_squad/" --per_device_train_batch_size 60

train-hotpot/colab:
	python3 run.py --do_train --task qa --dataset hotpot_qa:fullwiki --output_dir "$(TRAIN_PATH)/trained_model_hotpot_qa/" --per_device_train_batch_size 100

eval-nli/colab:
	python3 run.py --do_eval --task nli --dataset snli --model "$(TRAIN_PATH)/trained_model_nli/" --output_dir "$(TRAIN_PATH)/eval_output_nli/"

eval-squad/colab:
	python3 run.py --do_eval --task qa --dataset squad --model "$(TRAIN_PATH)/trained_model_squad/" --output_dir "$(TRAIN_PATH)/eval_output_squad/"

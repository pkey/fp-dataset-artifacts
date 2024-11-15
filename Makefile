
DRIVE_PATH = /content/drive/My Drive/model_training

download:
	git clone git@github.com:pkey/fp-dataset-artifacts.git

initialise/colab:
	apt install python3.10-venv
	python -m venv tf-venv
	source tf-venv/bin/activate

	pip install --upgrade pip
	pip install -r requirements.txt

train-nli/colab:
	python3 run.py --do_train --task nli --dataset snli --output_dir "$(DRIVE_PATH)/trained_model_nli/" --save_steps 25000 --per_device_train_batch_size 400

train-squad/colab:
	python3 run.py --do_train --task qa --dataset squad --output_dir "$(DRIVE_PATH)/trained_model_squad/" --per_device_train_batch_size 60

train-hotpot/colab:
	python3 run.py --do_train --task qa --dataset hotpot_qa:fullwiki --output_dir "$(DRIVE_PATH)/trained_model_hotpot_qa/" --per_device_train_batch_size 100

eval-nli/colab:
	python3 run.py --do_eval --task nli --dataset snli --model "$(DRIVE_PATH)/trained_model_nli/" --output_dir "$(DRIVE_PATH)/eval_output_nli/"

eval-squad/colab:
	python3 run.py --do_eval --task qa --dataset squad --model "$(DRIVE_PATH)/trained_model_squad/" --output_dir "$(DRIVE_PATH)/eval_output_squad/"

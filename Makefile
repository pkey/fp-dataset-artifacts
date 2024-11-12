
download:
	git clone git@github.com:pkey/fp-dataset-artifacts.git

initialise/colab:
	apt install python3.10-venv
	python -m venv tf-venv
	source tf-venv/bin/activate

	pip install --upgrade pip
	pip install -r requirements.txt

train-nli/colab:
	python3 run.py --do_train --task nli --dataset snli --output_dir "/content/drive/My Drive/Model Training/trained_model_nli/" --save_steps 25000 --per_device_train_batch_size 400

train-squad/colab:
	python3 run.py --do_train --task qa --dataset squad --output_dir "/content/drive/My Drive/Model Training/trained_model_squad/" --save_steps 25000 --per_device_train_batch_size 60

train-hotpot/colab:
	python3 run.py --do_train --task qa --dataset hotpot_qa:fullwiki --output_dir "/content/drive/My Drive/Model Training/trained_model_hotpot_qa/" --save_steps 25000 --per_device_train_batch_size 100

eval-nli/colab:
	python3 run.py --do_eval --task nli --dataset snli --model "/content/drive/MyDrive/Model Training/trained_model_nli/" --output_dir "/content/drive/My Drive/Model Training/eval_output/"

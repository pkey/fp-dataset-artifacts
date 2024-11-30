

### To run an experiment

```markdown
- create a file in format "${EXPERIMENT_NAME}_qa_format.csv" in /scripts

- if you don't have yet data, put it into "adversarial_date_excel.csv"
- the data must have columns: "context,question,answers"
- add your OPEN_API key: "export OPENAI_API_KEY=..."
- run in /scripts: python3 complicate_questions.py

to train and eval your model: "make train-eval-squad EXPERIMENT_NAME=${EXPERIMENT NAME}"

to eval your model on SQUAD and your model: "make eval-squad EXPERIMENT_NAME=${EXPERIMENT NAME}"

```

Some example runs:
```markdown
# GENERATE AI DATA of X SAMPLES
python3 scripts/complicate_questions.py 100

# MOVE YOUR GENERATED DATA TO THE EXPERIMENT CSV file
# cp scripts/adversarial_date_excel_openai.csv scripts/when_experiment_eval_ai_simple_qa_format.csv

# ON BASE: {'eval_exact_match': 39.21568627450981, 'eval_f1': 70.32850783715834}
make eval-squad-exp EXPERIMENT_NAME=when_experiment_eval_ai_simple

# ON TRAINED: {'eval_exact_match': 70.58823529411765, 'eval_f1': 85.57854762458341}
make train-eval-squad EXPERIMENT_NAME=when_experiment_eval_ai_simple

# SQUAD ON TRAINED MODEL: {'eval_exact_match': 71.8543046357616, 'eval_f1': 81.54690127731999}
make eval-squad EXPERIMENT_NAME=when_experiment_eval_ai_simple
```



## Getting Started

You'll need Python >= 3.6 to run the code in this repo.

First, clone the repository:

`git clone git@github.com:gregdurrett/fp-dataset-artifacts.git`

Then install the dependencies:

`pip install --upgrade pip`

`pip install -r requirements.txt`

### Sharing model with GDrive

1. Make sure you have access to the correct folder on GDRive and you have GDrive installed on your machine.
2. If you are one who the folder was shared with, make sure to create a shortcut to your own drive in GDrive.
3. Finally, symlink folder to GDrive (make sure you are in the root of this project OR replace `$PWD` with the path to the root of this project):

```
ln -s  ~/Google\ Drive/My\ Drive/Model\ Training $PWD
```

## Training and evaluating a model

To train an ELECTRA-small model on the SNLI natural language inference dataset, you can run the following command:

`python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model/`

Checkpoints will be written to sub-folders of the `trained_model` output directory.
To evaluate the final trained model on the SNLI dev set, you can use

`python3 run.py --do_eval --task nli --dataset snli --model ./trained_model/ --output_dir ./eval_output/`

To prevent `run.py` from trying to use a GPU for training, pass the argument `--no_cuda`.

To train/evaluate a question answering model on SQuAD instead, change `--task nli` and `--dataset snli` to `--task qa` and `--dataset squad`.

**Descriptions of other important arguments are available in the comments in `run.py`.**

Data and models will be automatically downloaded and cached in `~/.cache/huggingface/`.
To change the caching directory, you can modify the shell environment variable `HF_HOME` or `TRANSFORMERS_CACHE`.
For more details, see [this doc](https://huggingface.co/transformers/v4.0.1/installation.html#caching-models).

An ELECTRA-small based NLI model trained on SNLI for 3 epochs (e.g. with the command above) should achieve an accuracy of around 89%, depending on batch size.
An ELECTRA-small based QA model trained on SQuAD for 3 epochs should achieve around 78 exact match score and 86 F1 score.

## Working with datasets

This repo uses [Huggingface Datasets](https://huggingface.co/docs/datasets/) to load data.
The Dataset objects loaded by this module can be filtered and updated easily using the `Dataset.filter` and `Dataset.map` methods.
For more information on working with datasets loaded as HF Dataset objects, see [this page](https://huggingface.co/docs/datasets/process.html).

## Virtual environments

Python 3 supports virtual environments with the `venv` module. These will let you select a particular Python interpreter
to be the default (so that you can run it with `python`) and install libraries only for a particular project.
To set up a virtual environment, use the following command:

`python3 -m venv path/to/my_venv_dir`

This will set up a virtual environment in the target directory.
WARNING: This command overwrites the target directory, so choose a path that doesn't exist yet!

To activate your virtual environment (so that `python` redirects to the right version, and your virtual environment packages are active),
use this command:

`source my_venv_dir/bin/activate`

This command looks slightly different if you're not using `bash` on Linux. The [venv docs](https://docs.python.org/3/library/venv.html) have a list of alternate commands for different systems.

Once you've activated your virtual environment, you can use `pip` to install packages the way you normally would, but the installed
packages will stay in the virtual environment instead of your global Python installation. Only the virtual environment's Python
executable will be able to see these packages.

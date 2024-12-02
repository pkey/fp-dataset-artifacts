import csv
import json

import pandas as pd
from openai import ChatCompletion, OpenAI
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from helpers import QuestionAnsweringTrainer, prepare_validation_dataset_qa


def create_complexity_prompt(context, question, answers):
    return f"""
        You have:
            context: {context}
            question: {question}
            answers: {answers}

        You have change the context for the answer.

        Produce the output in ONLY THIS FORMAT in JSON - RETURN ONLY JSON:
        {{
            "new_context": "NEWLY_GENERATED_CONTEXT",
            "new_answer": "NEWLY_GENERATED_ANSWER",
        }}
    """


def main():
    done = False
    available_commands = ["exit", "no", "yes"]
    client = OpenAI()

    # initialise model

    trainer_class = Trainer
    eval_kwargs = {}
    compute_metrics = None
    # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
    # to enable the question-answering specific evaluation metrics
    trainer_class = QuestionAnsweringTrainer

    model_class = AutoModelForQuestionAnswering
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained("google/electra-small-discriminator")
    # Make tensor contiguous if needed https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, "electra"):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

    tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator", use_fast=True)

    json_data = []
    with open("./scripts/adversarial_date_excel.csv", "r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            json_data.append(row)

    index_of_example = 0
    generated_answers = []

    while not done:
        original_example = json_data[index_of_example]
        context, question, answers = original_example["context"], original_example["question"], original_example["answers"]
        prompt = create_complexity_prompt(context, question, answers)

        # response: ChatCompletion = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": """
        #                 You are an AI system which receives some context a question and an answer.
        #                 And for the same question, will produce a bit different context with a new answer which doesn't have specific dates.

        #                 You will use many different ways in not clearly saying the data, or giving similar events nearby with a date where
        #                 the answer should happen at a similar time.
        #             """,
        #         },
        #         {"role": "user", "content": prompt},
        #     ],
        # )

        try:
            # stripped_response = response.choices[0].message.content.strip().replace("`", "").replace("json", "")
            # structured_response = json.loads(stripped_response)
            structured_response = {"context": "mock context", "answer": "mock answer"}

            response_with_past_data = {
                **structured_response,
                "title": str(index_of_example),
                "context": context,
                "question": question,
                "answers": answers,
            }

            new_data = [
                {
                    "title": str(index_of_example),
                    "context": structured_response['new_context'],
                    "question": ,
                    "answers": answers,
                }
            ]

            df = pd.DataFrame(new_data)

            print(f"{index_of_example + 1} AI answer generated! ✅ ")
            print(structured_response)
            print("Does example make sense? Answer yes or no.")

        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}")

        single_example = [structured_response]

        def prepare_eval_dataset(exs):
            return prepare_validation_dataset_qa(exs, tokenizer)

        eval_dataset_featurized = single_example.map(
            prepare_eval_dataset,
            batched=False,
            num_proc=1,
            # should define columns here
            remove_columns=single_example.column_names,
        )

        trainer = trainer_class(
            model=model,
            # train_dataset=train_dataset_featurized,
            eval_dataset=eval_dataset_featurized,
            tokenizer=tokenizer,
        )

        result = trainer.evaluate()

        command = input()

        if command not in available_commands:
            print(f"Command {command} not available. Did you mean one of these: {available_commands}?")
            continue

        if command == "exit":
            done = True
            print("Script finished!")

        if command == "yes":
            generated_answers.append(response_with_past_data)

        if command == "no":
            pass


if __name__ == "__main__":
    # Ensures the script runs only when executed directly
    main()

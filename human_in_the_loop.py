import json
import subprocess
import tempfile
import uuid
from pathlib import Path

import pandas as pd
from openai import ChatCompletion, OpenAI


def _convert_and_save_to_json(input_data: list, output_path: str):
    output_data = []

    for item in input_data:
        answer_start = item["context"].lower().find(item["answer"].lower())

        converted_item = {
            "id": str(uuid.uuid4()),
            "title": item["title"],
            "context": item["context"],
            "question": item["question"],
            "answers": {"text": [item["answer"]], "answer_start": [answer_start]},
        }
        output_data.append(converted_item)

    with open(output_path, "w") as file:
        file.write(json.dumps(output_data, indent=2, ensure_ascii=False))

    return output_data


def _create_prompt():
    return """
        Produce a randomly generated passage or a paragraph on any topic, generate a question from it
        and provide an answer. It should be possible to locate the answer in the passage, spelled out exactly the same.

        Constraints:

        - The context should have at least 5 sentences and involve a narrative, progression of events, or interconnected facts.
        - The question should test the ability to extract specific details, infer connections between sentences, or grasp cause-effect relationships.
        The topic should involve advanced knowledge in history, science, literature, or a technical field.

        Produce the output in ONLY THIS FORMAT in JSON - RETURN ONLY JSON:
        {
            "title": "NEWLY_GENERATED_CONTEXT",
            "context": "NEWLY_GENERATED_ANSWER",
            "question": "NEWLY_GENERATED_QUESTION",
            "context": "NEWLY_GENERATED_ANSWER"
        }
    """


def generateExample(prompt):
    client = OpenAI()
    response: ChatCompletion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                    You are helpful assistant that creates examples for training QA
                    """,
            },
            {"role": "user", "content": prompt},
        ],
    )
    try:
        stripped_response = response.choices[0].message.content.strip().replace("`", "").replace("json", "")
        structured_response = json.loads(stripped_response)

        return structured_response

    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")


def main():
    path_to_data = "datasets/data_human_in_the_loop_main.json"
    done = False
    while not done:
        prompt = _create_prompt()

        example = generateExample(prompt=prompt)

        temp_dir = tempfile.mkdtemp()

        path_to_json = Path(f"{temp_dir}/example.json")

        new_data = _convert_and_save_to_json([example], path_to_json)

        path_to_results = Path(f"{temp_dir}/results")

        # Define the arguments
        args = [
            "python",
            "run.py",
            "--do_eval",
            "--task",
            "qa",
            "--dataset",
            f"{path_to_json}",
            "--model",
            "./model_training/trained_model_electra_small_squad",
            "--output_dir",
            f"{path_to_results}",
        ]

        # Run the command
        try:
            result = subprocess.run(args, check=True, capture_output=True, text=True)
            print("Command executed successfully!")
            print("Output:\n", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error occurred while running the command!")
            print("Error:\n", e.stderr)

        path_to_eval_metrics = path_to_results / "eval_metrics.json"
        # Open and read the JSON file

        df = pd.read_json(path_to_data)

        with open(path_to_eval_metrics, "r") as json_file:
            result = json.load(json_file)

            # exact_match = result["eval_exact_match"]
            f1_score = result["eval_f1"]

            if f1_score < 50:
                print("Added example!")
                df_new = pd.DataFrame(new_data)
                df = pd.concat([df, df_new])
            else:
                print("Example too easy, moving on.")

        df.to_json(path_to_data, orient="records")


if __name__ == "__main__":
    # Ensures the script runs only when executed directly
    main()

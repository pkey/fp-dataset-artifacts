import csv
import json
import sys
import uuid


def _convert_format(input_data):
    output_data = []

    for idx, item in enumerate(input_data):
        answer_start = item["context"].lower().find(item["answer"].lower())

        converted_item = {
            "id": str(uuid.uuid4()),
            "title": str(idx),
            "context": item["context"],
            "question": item["question"],
            "answers": {
                "text": [item["answer"]],
                "answer_start": [answer_start]
            }
        }
        output_data.append(converted_item)

    output_path = f"./experiments/complex/existing_questions.json"
    with open(output_path, "w") as file:
        file.write(json.dumps(output_data, indent=2, ensure_ascii=False))

    return output_data


def _read_csv_to_json(csv_path):
    input_data = []
    with open(csv_path, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            input_data.append(
                {
                    "context": row["new_context"],
                    "question": row["question"],
                    "answer": row["new_answer"]
                }
            )
    return input_data


if __name__ == "__main__":
    csv_input_data_path = f"./experiments/complex/existing_questions.csv"

    input_data = _read_csv_to_json(csv_input_data_path)

    _convert_format(input_data)
    print(f"Data successfully written to ./experiments/complex/existing_questions.json")
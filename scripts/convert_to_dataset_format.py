import csv
import json
import sys
import uuid


def _convert_format(experiment_name: str, input_data):
    output_data = []

    for item in input_data:
        answer_start = item["context"].lower().find(item["answer"].lower())

        converted_item = {
            "id": str(uuid.uuid4()),
            "title": item["title"],
            "context": item["context"],
            "question": item["question"],
            "answers": {
                "text": [item["answer"]],
                "answer_start": [answer_start]
            }
        }
        output_data.append(converted_item)

    output_path = f"./datasets/{experiment_name}.json"
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
                    "title": row["title"],
                    "context": row["context"],
                    "question": row["question"],
                    "answer": row["answer"]
                }
            )
    return input_data


if __name__ == "__main__":
    experiment_name = sys.argv[1]
    csv_input_data_path = f"./scripts/{experiment_name}_qa_format.csv"

    input_data = _read_csv_to_json(csv_input_data_path)

    _convert_format(experiment_name, input_data)
    print(f"Data successfully written to ./datasets/{experiment_name}.json")

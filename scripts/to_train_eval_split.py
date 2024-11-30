import csv
import json
import sys
import uuid


def _convert_format(output_path: str, input_data):
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

    print(f"Total experiment size = {len(input_data)}")
    train_split, validation_split = input_data[:len(input_data)//2], input_data[len(input_data)//2:]

    _convert_format(f"./datasets/train_with_eval/train_{experiment_name}.json", train_split)
    _convert_format(f"./datasets/train_with_eval/validation_{experiment_name}.json", validation_split)

    print(f"Data successfully for train and validation sets in ./datasets/train_with_eval")

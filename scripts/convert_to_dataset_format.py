import json
import sys
import uuid


def convert_format(experiment_name: str, input_data):
    output_data = []

    for item in input_data:
        answer_start = item["context"].find(item["answer"])

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


if __name__ == "__main__":
    experiment_name = sys.argv[1]
    input_data_path = f"./scripts/{experiment_name}_qa_dataset.json"

    with open(input_data_path, "r") as f:
        input_data = json.load(f)

    convert_format(experiment_name, input_data)
    print(f"Data successfully written to ./datasets/{experiment_name}.json")

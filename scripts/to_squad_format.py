import json
import sys

if __name__ == "__main__":
    args = sys.argv[1].split("=")
    if len(args) != 2:
        sys.exit("Please specify an experiment name")

    experiment_name = args[1]

    base_path = f"./datasets/{experiment_name}"
    with open(f"{base_path}/convert_test.json", "r") as file:
        data = json.load(file)

    converted_data = []
    for entry in data:
        context = entry["context"]
        question = entry["question"]
        answer = entry["answer"]

        answer_start = context.lower().find(answer.lower())
        if answer_start == -1:
            print(f"Answer not found for ${question}")
            continue

        paragraph = {
            "id": f"{len(converted_data) + 1}",
            "title": f"title-{len(converted_data) + 1}",
            "context": context,
            "question": question,
            "answers": {"text": [answer], "answer_start": [answer_start]}
        }
        converted_data.append(paragraph)

    print(f"Total file record size = {len(data)}")
    with open(f"{base_path}/test.json", "w") as test_file:
        json.dump(converted_data, test_file, indent=2)

    print(f"Data successfully converted in /datasets/{experiment_name}/test.json!")

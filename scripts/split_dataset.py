import json
import sys

if __name__ == "__main__":
    args = sys.argv[1].split("=")
    if len(args) != 2:
        sys.exit("Please specify an experiment name")

    experiment_name = args[1]

    base_path = f"./datasets/{experiment_name}"
    with open(f"{base_path}/test.json", "r") as file:
        data = json.load(file)

    split_index = int(len(data) * 0.7)
    train_data, validation_data = data[:split_index], data[split_index:]

    print(f"Total file record size = {len(data)}")
    with open(f"{base_path}/training.json", "w") as train_file:
        json.dump(train_data, train_file, indent=2)

    with open(f"{base_path}/validation.json", "w") as validation_file:
        json.dump(validation_data, validation_file, indent=2)

    print(f"Data successfully split to training and validation in /datasets/{experiment_name}/{{training|validation}}.json!")

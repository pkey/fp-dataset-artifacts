import json
import csv

with open("./adversarial_date_excel_openai.json", 'r') as json_file:
    data = json.load(json_file)

    with open("./adversarial_date_excel_openai.csv", 'w', newline='') as csv_file:
        fieldnames = ['title', 'context', 'question', 'answer']

        # renaming new_context to context + new_answer to answer
        for answer in data:
            answer.pop("answers")

            answer['context'] = answer.pop('new_context')
            answer['answer'] = answer.pop('new_answer')

        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        csv_writer.writeheader()
        csv_writer.writerows(data)
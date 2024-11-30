import pandas as pd
import json

input_file_path = 'adversarial_all_excel_file.csv'  # Replace with your file name
output_file_path = 'adversarial_all_excel_file.csv'  # Desired output file name

data = pd.read_csv(input_file_path)

cleaned_data = data.rename(columns={"new_answer": "answer"})
cleaned_data = cleaned_data[["new_context", "question", "answer"]]  # Retain only these columns

cleaned_data = cleaned_data.rename(columns={"new_context": "context"})
cleaned_data.insert(0, "title", cleaned_data.index)
cleaned_data = cleaned_data.dropna(subset=["context", "answer"]).reset_index(drop=True)

cleaned_data.to_csv(output_file_path, index=False)
print(f"Cleaned data saved to {output_file_path}")


input_file_path = 'adversarial_dates_generated.json'
output_file_path_clean = 'adversarial_dates_generated_final_good.csv'

# Load JSON from the file
with open(input_file_path, 'r') as file:
    data = json.load(file)  # Expecting a list of dictionaries

# Prepare list for storing CSV rows
csv_rows = []

# Process each JSON row
for row in data:
    # print(json.loads(row["answers"].replace("'", '"')))
    csv_row = {
        "title": row["title"],  # Map "title"
        "context": row["context"],  # Map "new_context" to "context"
        "question": row["question"],  # Map "question"
        "answer": json.loads(row["answers"].replace("'", '"'))['text'][0]  # Map "new_answer" to "answer"
    }
    csv_rows.append(csv_row)

# Convert to DataFrame
df = pd.DataFrame(csv_rows)

# Save to CSV
df.to_csv(output_file_path_clean, index=False)

print(f"CSV saved to: {output_file_path_clean}")
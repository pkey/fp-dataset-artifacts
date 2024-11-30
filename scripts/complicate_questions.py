
import csv
import sys
import json

from openai import OpenAI, ChatCompletion


client = OpenAI()

def create_complexity_prompt(context, question, answers):
    return f"""
        You have:
            context: {context}
            question: {question}
            answers: {answers} 
        
        You have to create THE SAME, BUT MORE complex and unique with more adversarial data context FOR THE SAME QUESTION to deduce an ANSWER which will be the same as the PREVIOUS, just PARAPHRASED -
        for example referencing other events, centuries, not specifically mentioned the concrete date, etc.. (how many different ways you think to make the answer in other words but be correct)
         
        some examples for inspiration: "in the year before other big stadium was built" / "when Lithuania rolled out Euro" / "two seasons before that" / "late 1920s." / 
        "on the same time when Battle of Verdun happened," / "on the third day of the full moon" 
         
        The answer MUST be a part of the NEW CONTEXT and be quite short. I MUST be able to do NEW_CONTEXT.find(NEW_ANSWER) to find the answer. 
        The new context can have as much random data or extra details. BUT I MUST find the answer in the new_context.
        
        Produce the output in ONLY THIS FORMAT in JSON - RETURN ONLY JSON:
        {{
            "new_context": "NEWLY_GENERATED_CONTEXT",
            "new_answer": "NEWLY_GENERATED_ANSWER",
        }}
    """

csv_file_path = 'data.csv'  # Replace with your CSV file

json_data = []
with open("./adversarial_dates.csv", 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        json_data.append(row)


with open("./adversarial_dates_generated.json", 'w') as json_file:
    generated_answers = []

    print("generating your data 📈..")
    for idx, data in enumerate(json_data[100:200]): # take only first 50 examples
        context, question, answers = data["context"], data["question"], data["answers"]
        complexity_prompt = create_complexity_prompt(context, question, answers)

        response: ChatCompletion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """
                        You are an AI system which receives some context a question and an answer.
                        And for the same question, will produce a bit different context with a new answer which doesn't have specific dates.
                        
                        You will use many different ways in not clearly saying the data, or giving similar events nearby with a date where
                        the answer should happen at a similar time.
                    """
                 },
                {"role": "user", "content": complexity_prompt},
            ]
        )

        try:
            stripped_response = response.choices[0].message.content.strip().replace("`", "").replace("json", "")
            structured_response = json.loads(stripped_response)

            response_with_past_data = {
                **structured_response,
                "title": str(idx),
                "context": context,
                "question": question,
                "answers": answers
            }

            print(f"{idx + 1} AI answer generated! ✅ ")
            generated_answers.append(response_with_past_data)
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}")

    json.dump(generated_answers, json_file, indent=2)

    with open("./adversarial_dates_generated.csv", 'w', newline='') as csv_file:
        # fieldnames = ['context', 'question', 'answers', "new_context", "new_answer"]
        fieldnames = ['title', 'new_context', 'question', 'new_answer']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        csv_writer.writeheader()
        csv_writer.writerows(generated_answers)

# latest example where no paraphrased data exists
# snippet >>> On 15 June 1520, the Pope warned Luther with the papal bull


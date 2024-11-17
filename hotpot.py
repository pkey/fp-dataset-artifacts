def create_example_dict(context, answer_start, answer, id, is_impossible, question):
    return {
        "context": context,
        "answers": [{"answer_start": answer_start, "text": answer}],
        "id": id,
        "is_impossible": is_impossible,
        "question": question,
    }


def add_yes_no(string):
    # Allow model to explicitly select yes/no from text (location front, avoid truncation)
    return " ".join(["yes", "no", string])

count = 0

def convert_hotpot_to_squad_format(example):
    global count

    raw_contexts = example["context"]

    context = " ".join([" ".join(sentence_list) for sentence_list in raw_contexts["sentences"]])

    answer = example["answer"]
    context = add_yes_no(context)
    answer_start = context.index(answer) if answer in context else -1

    updated_example = create_example_dict(
        context=context,
        answer_start=answer_start,
        answer=answer,
        id=str(count),
        is_impossible=(answer_start == -1),
        question=example["question"],
    )

    count += 1

    return updated_example

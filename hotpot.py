# for hotpot_qa we squash context sentences to one sentence (multi-hop to one single-hop)
def flatten_hotpot_context_sentence(example):
    context = example["context"]
    sentences_2d = context["sentences"]

    if sentences_2d:
        flattened_sentences = " ".join([" ".join(sentence_1d) for sentence_1d in sentences_2d])
        # overriding sentences to context metadata, probably can use more existing metadata here
        example["context"] = flattened_sentences

    # making the structure same as in squad for answers
    example["answers"] = [example["answers"]]
    return example
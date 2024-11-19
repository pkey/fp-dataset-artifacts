import collections
from typing import Tuple

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm
from transformers import EvalPrediction, Trainer

# Download NLTK tokenizer data (if not already done)
nltk.download("punkt")

QA_MAX_ANSWER_LENGTH = 30


# This function preprocesses an NLI dataset, tokenizing premises and hypotheses.
def prepare_dataset_nli(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    tokenized_examples = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )

    tokenized_examples["label"] = examples["label"]
    return tokenized_examples


# This function computes sentence-classification accuracy.
# Functions with signatures like this one work as the "compute_metrics" argument of transformers.Trainer.
def compute_accuracy(eval_preds: EvalPrediction):
    return {"accuracy": (np.argmax(eval_preds.predictions, axis=1) == eval_preds.label_ids).astype(np.float32).mean().item()}


# This function preprocesses a question answering dataset, tokenizing the question and context text
# and finding the right offsets for the answer spans in the tokenized context (to use as labels).
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def prepare_train_dataset_qa(examples, tokenizer, max_seq_length=None):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    # tokenize both questions and the corresponding context
    # if the context length is longer than max_length, we split it to several
    # chunks of max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context,
    # we need a map from a feature to its corresponding example.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position
    # in the original context. This will help us compute the start_positions
    # and end_positions to get the final answer string.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        # We will label features not containing the answer the index of the CLS token.
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        # from the feature idx to sample idx
        sample_index = sample_mapping[i]
        # get the answer for a feature
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_dataset_qa(examples, tokenizer):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None) for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    null_score_diff_threshold: float = 0.0,
    n_best_size: int = 20,
):
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > QA_MAX_ANSWER_LENGTH:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative and min_null_prediction is not None:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and min_null_prediction is not None and not any(p["offsets"] == (0, 0) for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

    return all_predictions


# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples

    def evaluate(
        self,
        eval_dataset=None,  # denotes the dataset after mapping
        eval_examples=None,  # denotes the raw dataset
        ignore_keys=None,  # keys to be ignored in dataset
        metric_key_prefix: str = "eval",
        version_2_with_negative: bool = False,
    ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            # compute the raw predictions (start_logits and end_logits)
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            # post process the raw predictions to get the final prediction
            # (from start_logits, end_logits to an answer string)

            eval_preds = postprocess_qa_predictions(eval_examples, eval_dataset, output.predictions, version_2_with_negative)
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in eval_preds.items()]
            if version_2_with_negative:
                formatted_predictions = [{**prediction, "no_answer_probability": 0.0} for prediction in formatted_predictions]

            references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

            # compute the metrics according to the predictions and references
            metrics = self.compute_metrics(EvalPrediction(predictions=formatted_predictions, label_ids=references))

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


# Function to compute letter overlap score
def calculate_overlap_score(predicted: str, answers: list[str]):
    """
    Compute the maximum letter overlap score between the predicted answer
    and the ground-truth answers.
    """

    def calculate_overlap(predicted, actual):
        # Tokenize both the predicted and actual answers
        predicted_tokens = set(word_tokenize(predicted.lower()))
        actual_tokens = set(word_tokenize(actual.lower()))

        # Compute common and total unique tokens
        common_tokens = predicted_tokens.intersection(actual_tokens)
        total_tokens = predicted_tokens.union(actual_tokens)

        # Compute overlap ratio
        return len(common_tokens) / len(total_tokens) if total_tokens else 0

    # Compute overlap for each ground-truth answer
    scores = [calculate_overlap(predicted, answer) for answer in answers]
    return max(scores) if scores else 0

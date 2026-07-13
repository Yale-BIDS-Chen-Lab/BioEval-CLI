import argparse
from src.evaluation.utils import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Compute exact metric for a single model output.")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset for which the metric is computed (e.g., plos, chemprot).")
    parser.add_argument('--task', type=str, required=True,
                                                 help="Task for which the metric is computed (e.g., generation).")
    parser.add_argument('--input', type=str, required=True,
                        help="Path to the JSON file containing model outputs and gold labels.")
    parser.add_argument('--metric', type=str, required=True,
                        help="Metric to compute (e.g., rouge1, accuracy).")
    parser.add_argument('--postprocessing', type=str, default=None,
                        help="Explicit postprocessing function name (e.g., load_normalized_data, process_mcq_custom).")
    parser.add_argument('--label_string', type=str, default=None,
                        help="Comma-separated string of labels for process_mlc_custom or option type for process_mcq_custom (e.g., A-E, Yes/No/Maybe).")

    args = parser.parse_args()

    # Evaluate the model using evaluate_model from utils
    if args.postprocessing == "process_mlc_custom" and args.label_string:
        from src.evaluation.data_processing import process_mlc_custom
        process_json = lambda file_path: process_mlc_custom(file_path, args.label_string)
        result_text = evaluate_model(dataset=args.dataset,
                                     task=args.task,
                                     json_file=args.input,
                                     metric=args.metric,
                                     postprocessing=None,
                                     process_json_override=process_json)
    elif args.postprocessing == "process_mcq_custom" and args.label_string:
        from src.evaluation.data_processing import process_mcq_custom
        process_json = lambda file_path: process_mcq_custom(file_path, args.label_string)
        result_text = evaluate_model(dataset=args.dataset,
                                     task=args.task,
                                     json_file=args.input,
                                     metric=args.metric,
                                     postprocessing=None,
                                     process_json_override=process_json)
    elif (args.postprocessing in ["process_ner_custom", "process_ner_token_indices"]) and args.label_string:
        from src.evaluation.data_processing import process_ner_token_indices
        process_json = lambda file_path: process_ner_token_indices(file_path, args.label_string)
        result_text = evaluate_model(dataset=args.dataset,
                                     task=args.task,
                                     json_file=args.input,
                                     metric=args.metric,
                                     postprocessing=None,
                                     process_json_override=process_json)
    elif args.postprocessing == "process_ner_char_offsets" and args.label_string:
        from src.evaluation.data_processing import process_ner_char_offsets
        process_json = lambda file_path: process_ner_char_offsets(file_path, args.label_string)
        result_text = evaluate_model(dataset=args.dataset,
                                     task=args.task,
                                     json_file=args.input,
                                     metric=args.metric,
                                     postprocessing=None,
                                     process_json_override=process_json)
    else:
        result_text = evaluate_model(dataset=args.dataset,
                                     task=args.task,
                                     json_file=args.input,
                                     metric=args.metric,
                                     postprocessing=args.postprocessing)

    # Simply print the result
    print(result_text)


if __name__ == "__main__":
    main()
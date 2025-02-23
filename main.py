import argparse
import json
import logging
import os

from lm_eval import tasks, evaluator

logging.getLogger("openai").setLevel(logging.WARNING)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_args', default="")
    parser.add_argument('--tasks', default="all_tasks")
    parser.add_argument('--provide_description', action="store_true")
    parser.add_argument('--num_fewshot', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--no_cache', action="store_true")
    parser.add_argument('--description_dict_path', default=None)
    parser.add_argument('--check_integrity', action="store_true")
    parser.add_argument('--return_vals', action="store_true")
    parser.add_argument('--output_base_path', default='./results')
    return parser.parse_args()


def main():
    args = parse_args()
    assert not args.provide_description  # not implemented

    if args.limit:
        print("WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, 'r') as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        check_integrity=args.check_integrity,
        return_vals=args.return_vals,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)
    # create a file for each task and put the output:
    for task_name in task_names:
        all_output = {"samples": results["samples"][task_name]}
        for metric_name, output_list in results["vals"][task_name].items():
            all_output[metric_name] = output_list
        with open(os.path.join(args.output_base_path, f'{task_name}.json'), "w") as f:
            json.dump(all_output, f)


    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    )
    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()

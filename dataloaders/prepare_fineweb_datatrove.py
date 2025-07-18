import argparse
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import LocalPipelineExecutor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder", type=str, required=True, help="Path to input .jsonl files"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to save processed .jsonl files",
    )
    parser.add_argument("--tasks", type=int, default=1, help="Number of parallel tasks")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers"
    )
    args = parser.parse_args()

    pipeline = [
        JsonlReader(data_folder=args.input_folder),
        LambdaFilter(lambda doc: doc.text and doc.text.strip() != ""),
        JsonlWriter(output_folder=args.output_folder),
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=f"{args.output_folder}/logs",
        tasks=args.tasks,
        workers=args.workers,
    )
    executor.run()

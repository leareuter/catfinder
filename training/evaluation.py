from cat_finder.evaluation.gnn_evaluation import Evaluation
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument(
    "--trainings_dir", type=str, required=True, help="Path to the training directory"
)
ap.add_argument(
    "--model",
    type=str,
    required=False,
    default="best_model",
    help="Name of the model to evaluate",
)
ap.add_argument(
    "--dataset",
    type=str,
    required=False,
    default=None,
    help="Path to the dataset, if None the path in the config is used",
)
ap.add_argument(
    "--output_dir",
    type=str,
    required=False,
    default=None,
    help="Path where to safe the evaluation results",
)
ap.add_argument(
    "--filename",
    type=str,
    required=False,
    default="evaluation_results",
    help="Filename of the evaluation results",
)
args = ap.parse_args()


eval = Evaluation(
    outputdir=args.trainings_dir,
    model=args.model,
)
eval.get_dataset(
    input_dir=args.dataset,
)
eval.evaluate_minimal_dataset(
    radius=0.7,
    hit_radius=0.3,
    threshold=0.3,
    min_hit_num=7,
)
eval.save_results(
    outputdir=args.output_dir,
    filename=args.filename,
)

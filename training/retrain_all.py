import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DISEASES = ("melasma", "port_wine_stain", "vitiligo")
BASELINE_RUN = {
    "run_name": "baseline",
    "created_at": "prefilled",
    "hyperparameters": {
        "learning_rate": 5e-5,
        "epochs": 100,
        "batch_size": 4,
        "weight_decay": 0.01,
    },
    "results": {
        "melasma": {"best_val_iou": 0.5597608264898718, "test_iou": 0.4826439215677261},
        "port_wine_stain": {"best_val_iou": 0.5841015669166646, "test_iou": 0.5921451932097909},
        "vitiligo": {"best_val_iou": 0.49330217319333486, "test_iou": None},
    },
}


def run_command(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def ensure_baseline(experiments_path: Path) -> list[dict[str, Any]]:
    experiments = load_json(experiments_path, [])
    if not isinstance(experiments, list):
        raise ValueError(f"{experiments_path} must contain a JSON array")
    if not any(row.get("run_name") == "baseline" for row in experiments):
        experiments.insert(0, BASELINE_RUN)
        write_json(experiments_path, experiments)
    return experiments


def best_val_from_history(history_path: Path, metric_name: str) -> float:
    history = load_json(history_path, [])
    if not history:
        return 0.0
    values = [float(row.get(metric_name, 0.0)) for row in history]
    return max(values) if values else 0.0


def test_iou_from_metrics(metrics_path: Path, metric_name: str) -> float:
    metrics = load_json(metrics_path, {})
    return float(metrics.get(metric_name, 0.0))


def format_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) < 0.001:
            return f"{value:.0e}"
        return f"{value:.4f}"
    return str(value)


def format_delta(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) < 0.00005:
        return "0.0000"
    return f"{value:+.4f}"


def generate_report(experiments_path: Path, report_path: Path) -> None:
    experiments = ensure_baseline(experiments_path)
    baseline = next((row for row in experiments if row.get("run_name") == "baseline"), None)

    lines = [
        "# Hyperparameter Experiment Report",
        "",
        f"Generated from `{experiments_path.as_posix()}`.",
        "",
    ]

    for disease in DISEASES:
        title = disease.replace("_", " ").title()
        baseline_test = None
        if baseline:
            baseline_test = baseline.get("results", {}).get(disease, {}).get("test_iou")

        lines.extend(
            [
                f"## {title}",
                "",
                "| Run | Learning Rate | Epochs | Batch Size | Weight Decay | Val IoU | Test IoU | Delta Test IoU |",
                "|-----|---------------|--------|------------|--------------|---------|----------|----------------|",
            ]
        )

        best_run = None
        best_test = None
        for row in experiments:
            params = row.get("hyperparameters", {})
            result = row.get("results", {}).get(disease, {})
            test_iou = result.get("test_iou")
            delta = None
            if isinstance(test_iou, (int, float)) and isinstance(baseline_test, (int, float)):
                delta = float(test_iou) - float(baseline_test)
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("run_name", "")),
                        format_number(params.get("learning_rate")),
                        format_number(params.get("epochs")),
                        format_number(params.get("batch_size")),
                        format_number(params.get("weight_decay")),
                        format_number(result.get("best_val_iou")),
                        format_number(test_iou),
                        format_delta(delta),
                    ]
                )
                + " |"
            )
            if isinstance(test_iou, (int, float)) and (best_test is None or float(test_iou) > best_test):
                best_test = float(test_iou)
                best_run = row.get("run_name")

        lines.append("")
        if best_run is not None:
            if isinstance(baseline_test, (int, float)):
                delta = best_test - float(baseline_test)
                lines.append(f"**Best run:** {best_run} ({format_delta(delta)} test IoU vs baseline)")
            else:
                lines.append(f"**Best run:** {best_run}")
        else:
            lines.append("**Best run:** not available")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def build_step_commands(args) -> list[dict[str, Any]]:
    checkpoint_root = Path(args.checkpoint_root)
    melasma_dir = checkpoint_root / "melasma"
    port_dir = checkpoint_root / "port_wine_stain"
    vitiligo_dir = checkpoint_root / "vitiligo"

    common = [
        "--image-size",
        str(args.image_size),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]

    return [
        {
            "name": "melasma",
            "train": [
                sys.executable,
                "training/SegFormer/melasma/train_melasma.py",
                "--dataset-dir",
                args.melasma_dataset,
                "--checkpoint",
                args.initial_checkpoint,
                "--output-dir",
                str(melasma_dir),
                *common,
            ],
            "evaluate": [
                sys.executable,
                "training/SegFormer/melasma/evaluate_melasma.py",
                "--dataset-dir",
                args.melasma_dataset,
                "--checkpoint",
                str(melasma_dir / "best"),
                "--output-dir",
                str(melasma_dir / "evaluation"),
                "--split",
                "test",
                "--image-size",
                str(args.image_size),
                "--max-overlays",
                str(args.max_overlays),
                "--device",
                args.device,
            ],
            "history": melasma_dir / "training_history.json",
            "history_metric": "val_melasma_iou",
            "metrics": melasma_dir / "evaluation" / "test_metrics.json",
            "test_metric": "mean_melasma_iou",
        },
        {
            "name": "port_wine_stain",
            "train": [
                sys.executable,
                "training/SegFormer/port_wine_stain/train_port_wine_stain.py",
                "--dataset-dir",
                args.port_wine_stain_dataset,
                "--checkpoint",
                str(melasma_dir / "best"),
                "--output-dir",
                str(port_dir),
                *common,
            ],
            "evaluate": [
                sys.executable,
                "training/SegFormer/port_wine_stain/evaluate_port_wine_stain.py",
                "--dataset-dir",
                args.port_wine_stain_dataset,
                "--checkpoint",
                str(port_dir / "best"),
                "--output-dir",
                str(port_dir / "evaluation"),
                "--split",
                "test",
                "--image-size",
                str(args.image_size),
                "--max-overlays",
                str(args.max_overlays),
                "--device",
                args.device,
            ],
            "history": port_dir / "training_history.json",
            "history_metric": "val_port_wine_stain_iou",
            "metrics": port_dir / "evaluation" / "test_metrics.json",
            "test_metric": "mean_port_wine_stain_iou",
        },
        {
            "name": "vitiligo",
            "train": [
                sys.executable,
                "training/SegFormer/vitiligo/train_vitiligo.py",
                "--dataset-dir",
                args.vitiligo_dataset,
                "--checkpoint",
                str(port_dir / "best"),
                "--output-dir",
                str(vitiligo_dir),
                *common,
            ],
            "evaluate": [
                sys.executable,
                "training/SegFormer/vitiligo/evaluate_vitiligo.py",
                "--dataset-dir",
                args.vitiligo_dataset,
                "--checkpoint",
                str(vitiligo_dir / "best"),
                "--output-dir",
                str(vitiligo_dir / "evaluation"),
                "--split",
                "test",
                "--image-size",
                str(args.image_size),
                "--max-overlays",
                str(args.max_overlays),
                "--device",
                args.device,
            ],
            "history": vitiligo_dir / "training_history.json",
            "history_metric": "val_vitiligo_iou",
            "metrics": vitiligo_dir / "evaluation" / "test_metrics.json",
            "test_metric": "mean_vitiligo_iou",
        },
    ]


def append_experiment(args, steps: list[dict[str, Any]]) -> None:
    experiments_path = Path(args.experiments_path)
    report_path = Path(args.report_path)
    experiments = ensure_baseline(experiments_path)

    results = {}
    for step in steps:
        results[step["name"]] = {
            "best_val_iou": best_val_from_history(step["history"], step["history_metric"]),
            "test_iou": test_iou_from_metrics(step["metrics"], step["test_metric"]),
        }

    entry = {
        "run_name": args.run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
        },
        "results": results,
    }
    experiments.append(entry)
    write_json(experiments_path, experiments)
    generate_report(experiments_path, report_path)

    print(f"appended_experiment={experiments_path}")
    print(f"updated_report={report_path}")


def export_final_model(args) -> None:
    checkpoint_root = Path(args.checkpoint_root)
    source = checkpoint_root / "vitiligo" / "best"
    target = Path(args.final_model_dir)

    if not source.exists():
        raise FileNotFoundError(f"Final source checkpoint not found: {source}")

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)

    metadata = {
        "model_name": "lumiere_segformer_unified",
        "source_checkpoint": source.as_posix(),
        "training_order": ["melasma", "port_wine_stain", "vitiligo"],
        "class_mapping": {
            "0": "background",
            "1": "vitiligo",
            "2": "melasma_like_hyperpigmentation",
            "3": "port_wine_stain",
        },
        "run_name": args.run_name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    (target / "lumiere_unified_model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"exported_final_model={target}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Lumiere SegFormer retraining in order: Melasma -> Port Wine Stain -> Vitiligo."
    )
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--max-overlays", type=int, default=12)
    parser.add_argument("--initial-checkpoint", default="training/checkpoints/SegFormer/segformer_b2_melasma_colab_export")
    parser.add_argument("--checkpoint-root", default="training/checkpoints/SegFormer/unified/tmp")
    parser.add_argument("--melasma-dataset", default="data-collection/melasma")
    parser.add_argument("--port-wine-stain-dataset", default="data-collection/port_wine_stain/processed")
    parser.add_argument("--vitiligo-dataset", default="data-collection/vitiligo")
    parser.add_argument("--experiments-path", default="training/experiments.json")
    parser.add_argument("--report-path", default="training/experiment_report.md")
    parser.add_argument("--final-model-dir", default="training/checkpoints/SegFormer/unified/best")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only refresh experiments.json/report from existing output artifacts.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip test evaluation and read existing evaluation metrics.",
    )
    return parser.parse_args()


def validate_device(device_name: str) -> None:
    if device_name == "auto":
        return
    try:
        import torch
    except ImportError:
        return

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch environment does not have CUDA available. "
            "Use '--device auto' or '--device mps' on Apple Silicon, or run with '--device cuda' in Colab/NVIDIA GPU."
        )
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested, but this PyTorch environment does not have Apple MPS available. "
            "Use '--device auto' or '--device cpu'."
        )


def main() -> None:
    args = parse_args()
    validate_device(args.device)
    steps = build_step_commands(args)
    ensure_baseline(Path(args.experiments_path))

    for step in steps:
        if not args.skip_training:
            run_command(step["train"])
        if not args.skip_evaluation:
            run_command(step["evaluate"])

    append_experiment(args, steps)
    export_final_model(args)


if __name__ == "__main__":
    main()

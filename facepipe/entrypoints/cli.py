"""
CLI administration tool for FacePipe.

Commands:
  - migrate: Migrate v1 data to v2
  - evaluate: Run benchmark evaluation
  - evaluate-failures: Analyze failed benchmark pairs
  - inspect: View identity database stats
  - models: Check model download status
  - demo: Launch interactive Gradio demo
"""

from __future__ import annotations

import argparse
import sys

from facepipe.observability.logging import setup_logging


def do_migrate(args: argparse.Namespace) -> None:
    """Run data migration."""
    from facepipe.storage.migrations.migrate_v1_to_v2 import migrate_v1_to_v2
    result = migrate_v1_to_v2()
    print(f"Migration completed: {result}")


def do_evaluate(args: argparse.Namespace) -> None:
    """Run benchmark evaluation."""
    from facepipe.config.settings import get_settings
    from facepipe.evaluation.benchmark import BenchmarkHarness

    get_settings()
    harness = BenchmarkHarness()

    pairs = harness.parse_lfw_pairs(args.pairs, args.lfw_dir)
    print(f"Parsed {len(pairs)} pairs from {args.pairs}")

    result = harness.evaluate_pairs(pairs, dataset_name="LFW")

    report = harness.format_report([result])
    print(report)

    if args.output:
        harness.save_results([result], args.output)
        print(f"Results saved to {args.output}")


def do_evaluate_failures(args: argparse.Namespace) -> None:
    """Run failure analysis on benchmark pairs."""
    from facepipe.evaluation.benchmark import BenchmarkHarness
    from facepipe.evaluation.failure_analysis import FailureAnalyzer

    harness = BenchmarkHarness()
    pairs = harness.parse_lfw_pairs(args.pairs, args.lfw_dir)
    print(f"Parsed {len(pairs)} pairs from {args.pairs}")
    print("Running failure analysis (detection-only pass)...")

    analyzer = FailureAnalyzer()
    report = analyzer.analyze_pairs(pairs)

    print(analyzer.format_report(report))

    if args.output:
        analyzer.save_report(report, args.output)
        print(f"Failure analysis saved to {args.output}")


def do_inspect(args: argparse.Namespace) -> None:
    """Inspect identity database."""
    from facepipe.core.pipeline import RecognitionPipeline
    from facepipe.storage.identity_manager import IdentityManager

    mgr = IdentityManager()
    pipeline = RecognitionPipeline()
    pipeline.initialize()

    print(f"Total Identities (Active): {mgr.count(active_only=True)}")
    print(f"Total Identities (All):    {mgr.count(active_only=False)}")
    print(f"Vector Store Index Size:   {pipeline.vector_store.size}")

    if args.list:
        identities = mgr.list_all()
        print("\nActive Identities:")
        print(f"{'ID':<30} | {'Name':<25} | {'Embeddings':<10} | {'Clusters':<8}")
        print("-" * 80)
        for ident in identities:
            print(f"{ident.identity_id:<30} | {ident.name:<25} | {ident.embedding_count:<10} | {ident.cluster_count:<8}")


def do_models(args: argparse.Namespace) -> None:
    """Check model download status."""
    from facepipe.core.models import print_model_status
    print_model_status()


def do_demo(args: argparse.Namespace) -> None:
    """Launch interactive Gradio demo."""
    try:
        import gradio  # noqa: F401
    except ImportError:
        print("Gradio not installed. Run: pip install facepipe[demo]")
        sys.exit(1)

    from demo.app import demo
    print("Launching FacePipe demo at http://localhost:7860")
    demo.launch(server_name=args.host, server_port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FacePipe — Production-ready face recognition pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Migrate
    subparsers.add_parser("migrate", help="Migrate v1 database to v2")

    # Evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Run benchmark evaluation")
    parser_eval.add_argument("--pairs", required=True, help="Path to pairs.txt file")
    parser_eval.add_argument("--lfw-dir", required=True, help="Path to LFW dataset directory")
    parser_eval.add_argument("--output", help="Path to save JSON results")

    # Inspect
    parser_inspect = subparsers.add_parser("inspect", help="Inspect database status")
    parser_inspect.add_argument("--list", action="store_true", help="List all identities")

    # Evaluate Failures
    parser_fail = subparsers.add_parser("evaluate-failures", help="Analyze failed benchmark pairs")
    parser_fail.add_argument("--pairs", required=True, help="Path to pairs.txt file")
    parser_fail.add_argument("--lfw-dir", required=True, help="Path to LFW dataset directory")
    parser_fail.add_argument("--output", help="Path to save JSON failure report")

    # Models
    subparsers.add_parser("models", help="Check model download status")

    # Demo
    parser_demo = subparsers.add_parser("demo", help="Launch interactive Gradio demo")
    parser_demo.add_argument("--host", default="0.0.0.0", help="Demo server host")
    parser_demo.add_argument("--port", type=int, default=7860, help="Demo server port")

    args = parser.parse_args()

    setup_logging(level="DEBUG" if args.debug else "INFO", json_output=False)

    commands = {
        "migrate": do_migrate,
        "evaluate": do_evaluate,
        "inspect": do_inspect,
        "evaluate-failures": do_evaluate_failures,
        "models": do_models,
        "demo": do_demo,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


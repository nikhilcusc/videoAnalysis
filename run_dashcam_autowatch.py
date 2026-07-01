from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dashcam_change_analysis import (
    AnalysisConfig,
    analyze_top_changed_frames,
    save_analysis_results,
    setup_logging,
    summarize_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run dashcam frame-change and object-transition analysis."
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("12Feb2022/VID_004.MOV"),
        help="Path to the dashcam video file.",
    )
    parser.add_argument(
        "--model",
        dest="model_name_or_path",
        default="yolov8n.pt",
        help="Ultralytics model name or local weights path.",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=5,
        help="Sample every Nth frame.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of high-change frame pairs to keep.",
    )
    parser.add_argument(
        "--diff-threshold",
        type=int,
        default=25,
        help="Threshold for pixel-change masking.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for matching detections between frames.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=960,
        help="Optional width used to resize extracted frames.",
    )
    parser.add_argument(
        "--object-limit",
        type=int,
        default=25,
        help="Optional cap on the number of detections per frame.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/dashcam_change_analysis"),
        help="Directory for annotated image outputs.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Where to write the compact CSV summary.",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Save annotated frames for the top changed results.",
    )
    return parser


def write_summary_csv(summary, csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in summary.results:
        compact = summarize_result(result)
        rows.append(
            {
                "timestamp_seconds": round(compact["timestamp_seconds"], 2),
                "frame_index": compact["frame_index"],
                "previous_frame_index": compact["previous_frame_index"],
                "change_score": round(compact["change_score"], 2),
                "changed_pixels": compact["changed_pixels"],
                "changed_ratio": round(compact["changed_ratio"], 4),
                "detections": ", ".join(
                    detection["class_name"] for detection in compact["detected_objects"]
                ),
                "entered_objects": ", ".join(
                    detection["class_name"] for detection in compact["entered_objects"]
                ),
                "exited_objects": ", ".join(
                    detection["class_name"] for detection in compact["exited_objects"]
                ),
                "matched_objects": ", ".join(
                    f"{match['previous']['class_name']}->{match['current']['class_name']}({match['iou']:.2f})"
                    for match in compact["matched_objects"]
                ),
            }
        )

    fieldnames = [
        "timestamp_seconds",
        "frame_index",
        "previous_frame_index",
        "change_score",
        "changed_pixels",
        "changed_ratio",
        "detections",
        "entered_objects",
        "exited_objects",
        "matched_objects",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging()

    config = AnalysisConfig(
        resize_width=args.resize_width,
        diff_threshold=args.diff_threshold,
        top_n=args.top_n,
        sample_interval=args.sample_interval,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold,
        object_limit=args.object_limit,
    )

    summary = analyze_top_changed_frames(
        video_path=args.video_path,
        model_name_or_path=args.model_name_or_path,
        config=config,
    )

    print(f"Video: {summary.video_path}")
    print(f"Frames analyzed: {summary.metadata.frame_count}")
    print(f"Results kept: {len(summary.results)}")

    for result in summary.results:
        compact = summarize_result(result)
        print(
            f"Frame {compact['frame_index']} | score={compact['change_score']:.0f} | "
            f"entered={len(compact['entered_objects'])} | "
            f"exited={len(compact['exited_objects'])} | "
            f"matched={len(compact['matched_objects'])}"
        )

    csv_path = args.csv_path or Path("output") / f"{args.video_path.stem}_top_changed_frames_detections.csv"
    saved_csv = write_summary_csv(summary, csv_path)
    print(f"CSV saved to: {saved_csv}")

    if args.save_annotated:
        saved_images = save_analysis_results(summary, output_dir=args.output_dir)
        print(f"Annotated frames saved: {len(saved_images)}")
        if saved_images:
            print(f"Annotated output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
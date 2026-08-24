from __future__ import annotations

import csv
import heapq
from pathlib import Path
from typing import Iterable
import cv2

ResultTuple = tuple[int, int, float]

def find_top_changed_frames(
    video_path: str,
    top_n: int = 20,
    diff_threshold: int = 25,
    time_relaxation_seconds: float = 2.0,
) -> list[ResultTuple]:
    """Find the top N frames with the most pixel changes vs previous frames.

    Each result tuple is:
        (changed_pixels, frame_number, timestamp_seconds)
    """
    if top_n <= 0:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # Fallback for videos where FPS metadata is unavailable.
        fps = 30.0

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return []

    prev_gray_blurred = cv2.GaussianBlur(
        cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY),
        (5, 5),
        0,
    )

    frame_number = 1
    candidate_heap: list[ResultTuple] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_gray_blurred = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (5, 5),
            0,
        )

        abs_diff = cv2.absdiff(current_gray_blurred, prev_gray_blurred)
        _, diff_mask = cv2.threshold(abs_diff, diff_threshold, 255, cv2.THRESH_BINARY)
        changed_pixels = int(cv2.countNonZero(diff_mask))

        timestamp_seconds = frame_number / fps
        result = (changed_pixels, frame_number, float(timestamp_seconds))

        if len(candidate_heap) < top_n:
            heapq.heappush(candidate_heap, result)
        elif changed_pixels > candidate_heap[0][0]:
            heapq.heapreplace(candidate_heap, result)

        prev_gray_blurred = current_gray_blurred
        frame_number += 1

    cap.release()

    selected_frames: list[ResultTuple] = []
    for candidate in sorted(candidate_heap, key=lambda item: item[0], reverse=True):
        if all(
            abs(candidate[2] - selected[2]) >= time_relaxation_seconds
            for selected in selected_frames
        ):
            selected_frames.append(candidate)

        if len(selected_frames) >= top_n:
            break

    return selected_frames


def save_frame_change_counts_to_csv(
    video_path: str,
    csv_path: str = "output/frame_change_counts.csv",
    diff_threshold: int = 25,
) -> Path:
    """Save the changed-pixel count for every frame in the video to CSV.

    The first frame is recorded with 0 changed pixels because there is no
    previous frame to compare against.
    """
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Frame Number", "Timestamp (Seconds)", "Changed Pixels"])
        return output_path

    prev_gray_blurred = cv2.GaussianBlur(
        cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY),
        (5, 5),
        0,
    )

    frame_number = 0

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Frame Number", "Timestamp (Seconds)", "Changed Pixels"])

        writer.writerow([frame_number, frame_number / fps, 0])
        frame_number += 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_gray_blurred = cv2.GaussianBlur(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (5, 5),
                0,
            )

            abs_diff = cv2.absdiff(current_gray_blurred, prev_gray_blurred)
            _, diff_mask = cv2.threshold(abs_diff, diff_threshold, 255, cv2.THRESH_BINARY)
            changed_pixels = int(cv2.countNonZero(diff_mask))

            writer.writerow([frame_number, frame_number / fps, changed_pixels])

            prev_gray_blurred = current_gray_blurred
            frame_number += 1

    cap.release()
    return output_path


def export_results_to_csv(
    results: Iterable[ResultTuple],
    csv_path: str = "output/top_changed_frames.csv",
) -> Path:
    """Export analysis results to a CSV file and return the saved path."""
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Changed Pixels", "Timestamp (Seconds)", "Frame Number"])
        for changed_pixels, frame_number, timestamp_seconds in results:
            writer.writerow([changed_pixels, timestamp_seconds, frame_number])

    return output_path


def save_frames_as_images(
    video_path: str,
    results: Iterable[ResultTuple],
    output_dir: str = "output",
) -> list[Path]:
    """Save exact frames listed in results as JPEG images and return file paths."""
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    saved_files: list[Path] = []

    # Save in ranked order, highest changed pixel count first.
    ranked_results = sorted(results, key=lambda item: item[0], reverse=True)

    for rank, (changed_pixels, frame_number, _timestamp_seconds) in enumerate(ranked_results, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            continue

        file_name = f"rank_{rank:02d}_frame_{frame_number:06d}_changes_{changed_pixels}.jpg"
        file_path = output_folder / file_name

        if cv2.imwrite(str(file_path), frame):
            saved_files.append(file_path)

    cap.release()
    return saved_files

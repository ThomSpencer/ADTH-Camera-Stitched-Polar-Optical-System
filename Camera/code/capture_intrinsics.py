import os
from datetime import datetime
from typing import List

import cv2  # type: ignore
import numpy as np # type: ignore

# Update these indices based on `v4l2-ctl --list-devices`
CAMERA_INDICES = ["/dev/video2", "/dev/video4", "/dev/video0"]
FRAME_SIZE = (640, 480)
TARGET_FPS = 30
FOURCC = "MJPG"

OUTPUT_ROOT = "calibration/intrinsics"
CAMERA_NAMES = ["cam0", "cam1", "cam2"]


def read_frame_with_retries(cap, retries=1):
	for _ in range(retries):
		ret, frame = cap.read()
		if ret:
			return True, frame
	return False, None


def open_cameras(indices: List[str]):
	opened = []
	for idx in indices:
		print(f"Attempting to open camera at index {idx}...")
		cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
		if not cap.isOpened():
			print(f"Failed to open camera index {idx}")
			cap.release()
			continue
		cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
		cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
		cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
		read_frame_with_retries(cap, retries=3)
		opened.append((idx, cap))
		print(f"Successfully opened camera index {idx}")
	return opened


def _ensure_output_dirs() -> List[str]:
	if len(CAMERA_NAMES) != len(CAMERA_INDICES):
		raise RuntimeError("CAMERA_NAMES must match CAMERA_INDICES length.")
	paths = []
	for name in CAMERA_NAMES:
		path = os.path.join(OUTPUT_ROOT, name)
		os.makedirs(path, exist_ok=True)
		paths.append(path)
	return paths


def _make_filename() -> str:
	return datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"


def main() -> None:
	output_dirs = _ensure_output_dirs()
	cameras = open_cameras(CAMERA_INDICES)
	if len(cameras) != len(CAMERA_INDICES):
		print("Not all cameras could be opened.")
		for _, cap in cameras:
			cap.release()
		return

	cv2.namedWindow("Capture Intrinsics", cv2.WINDOW_NORMAL)

	try:
		while True:
			frames: List[np.ndarray] = []
			for idx, cap in cameras:
				ret, frame = read_frame_with_retries(cap, retries=1)
				if not ret:
					print(f"Failed to grab frame from camera {idx}")
					frames = []
					break
				frames.append(cv2.resize(frame, FRAME_SIZE))

			if frames:
				preview = cv2.hconcat(frames)
				cv2.putText(
					preview,
					"SPACE=save  Q=quit",
					(10, 30),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.8,
					(255, 255, 255),
					2,
				)
				cv2.imshow("Capture Intrinsics", preview)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
			if key == ord(" ") and frames:
				filename = _make_filename()
				for frame, out_dir in zip(frames, output_dirs):
					path = os.path.join(out_dir, filename)
					cv2.imwrite(path, frame)
				print(f"Saved {filename}")
	finally:
		for _, cap in cameras:
			cap.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

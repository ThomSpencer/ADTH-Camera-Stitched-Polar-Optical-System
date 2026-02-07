import glob
import os
from typing import Dict, List, Tuple

import cv2  # type: ignore
import numpy as np

# AprilTag board config
APRILTAG_DICT = cv2.aruco.DICT_APRILTAG_36h11
BOARD_ROWS = 4
BOARD_COLS = 6
TAG_SIZE_MM = 25.0
TAG_SPACING_MM = 5.0

IMAGE_GLOB = "calibration/intrinsics/*/*.jpg"
OUTPUT_FILE = "calibration/intrinsics_params.npz"


def _collect_images() -> List[str]:
	return sorted(glob.glob(IMAGE_GLOB))


def _build_board():
	dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT)
	board = cv2.aruco.GridBoard(
		size=(BOARD_COLS, BOARD_ROWS),
		markerLength=TAG_SIZE_MM,
		markerSeparation=TAG_SPACING_MM,
		dictionary=dictionary,
	)
	return dictionary, board


def _group_by_camera(paths: List[str]) -> Dict[str, List[str]]:
	cameras: Dict[str, List[str]] = {}
	for path in paths:
		parts = os.path.normpath(path).split(os.sep)
		if len(parts) < 2:
			continue
		camera_id = parts[-2]
		cameras.setdefault(camera_id, []).append(path)
	return cameras


def _detect_markers(
	image_paths: List[str],
	dictionary,
	detector,
) -> Tuple[List[np.ndarray], np.ndarray, List[int], Tuple[int, int] | None]:
	all_corners: List[np.ndarray] = []
	all_ids: List[np.ndarray] = []
	counter: List[int] = []
	image_size = None
	for path in image_paths:
		img = cv2.imread(path)
		if img is None:
			continue
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		image_size = gray.shape[::-1]
		corners, ids, _rejected = detector.detectMarkers(gray)
		if ids is None or len(ids) == 0:
			continue
		all_corners.extend(corners)
		all_ids.append(ids)
		counter.append(len(corners))
	if all_ids:
		ids_flat = np.vstack(all_ids)
	else:
		ids_flat = np.empty((0, 1), dtype=np.int32)
	return all_corners, ids_flat, counter, image_size


def main() -> None:
	paths = _collect_images()
	if not paths:
		raise RuntimeError("No calibration images found.")

	dictionary, board = _build_board()
	detector = cv2.aruco.ArucoDetector(dictionary)

	cameras = _group_by_camera(paths)
	params = {}
	for camera_id, image_paths in cameras.items():
		print(f"Calibrating {camera_id} with {len(image_paths)} images...")
		all_corners, all_ids, counter, image_size = _detect_markers(image_paths, dictionary, detector)
		if image_size is None or not all_corners or all_ids.size == 0:
			raise RuntimeError(f"No valid detections for camera {camera_id}.")

		counter = np.array(counter, dtype=np.int32)

		ret, mtx, dist, _rvecs, _tvecs = cv2.aruco.calibrateCameraAruco(
			corners=all_corners,
			ids=all_ids,
			counter=counter,
			board=board,
			imageSize=image_size,
			cameraMatrix=None,
			distCoeffs=None,
		)
		if not ret:
			raise RuntimeError(f"Calibration failed for {camera_id}.")
		params[camera_id] = {"mtx": mtx, "dist": dist}

	os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
	flat = {
		**{f"{cid}_mtx": p["mtx"] for cid, p in params.items()},
		**{f"{cid}_dist": p["dist"] for cid, p in params.items()},
	}
	np.savez(OUTPUT_FILE, **flat)
	print(f"Saved intrinsics to {OUTPUT_FILE}")


if __name__ == "__main__":
	main()

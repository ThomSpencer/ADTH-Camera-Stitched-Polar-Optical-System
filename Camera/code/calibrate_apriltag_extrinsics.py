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

IMAGE_GLOB = "calibration/extrinsics/*/*.jpg"
INTRINSICS_FILE = "calibration/intrinsics_params.npz"
OUTPUT_FILE = "calibration/extrinsics_params.npz"

# Tuning thresholds
MIN_SHARED_IMAGES = 5
MIN_SHARED_MARKERS = 1
MIN_VALID_PAIRS = 5


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


def _load_intrinsics(path: str) -> Dict[str, Dict[str, np.ndarray]]:
	if not os.path.isfile(path):
		raise RuntimeError(f"Intrinsics file not found: {path}")
	data = np.load(path)
	params: Dict[str, Dict[str, np.ndarray]] = {}
	for key in data.files:
		if key.endswith("_mtx"):
			camera_id = key[: -len("_mtx")]
			params.setdefault(camera_id, {})["mtx"] = data[key]
		if key.endswith("_dist"):
			camera_id = key[: -len("_dist")]
			params.setdefault(camera_id, {})["dist"] = data[key]
	return params


def _build_marker_map(board) -> Dict[int, np.ndarray]:
	marker_map: Dict[int, np.ndarray] = {}
	ids = board.getIds()
	obj_points = board.getObjPoints()
	for marker_id, corners in zip(ids, obj_points):
		marker_map[int(marker_id)] = corners.reshape(4, 3)
	return marker_map


def _detect_markers(path: str, detector) -> Tuple[List[np.ndarray], np.ndarray | None, Tuple[int, int] | None]:
	img = cv2.imread(path)
	if img is None:
		return [], None, None
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	image_size = gray.shape[::-1]
	corners, ids, _rejected = detector.detectMarkers(gray)
	return corners, ids, image_size


def main() -> None:
	paths = _collect_images()
	if not paths:
		raise RuntimeError("No calibration images found.")

	dictionary, board = _build_board()
	detector = cv2.aruco.ArucoDetector(dictionary)
	marker_map = _build_marker_map(board)

	cameras = _group_by_camera(paths)
	params = _load_intrinsics(INTRINSICS_FILE)

	image_sizes: Dict[str, Tuple[int, int]] = {}
	per_camera: Dict[str, Dict[str, Tuple[List[np.ndarray], np.ndarray]]] = {}
	for camera_id, image_paths in cameras.items():
		per_camera[camera_id] = {}
		for path in image_paths:
			corners, ids, image_size = _detect_markers(path, detector)
			if image_size is not None:
				image_sizes[camera_id] = image_size
			if ids is None or len(ids) == 0:
				continue
			per_camera[camera_id][os.path.basename(path)] = (corners, ids)

	results: Dict[str, Dict[str, np.ndarray]] = {}
	camera_ids = sorted(cameras.keys())
	for i in range(len(camera_ids)):
		for j in range(i + 1, len(camera_ids)):
			cam_a = camera_ids[i]
			cam_b = camera_ids[j]
			if cam_a not in params or cam_b not in params:
				print(f"Missing intrinsics for {cam_a} or {cam_b}")
				continue
			common = sorted(set(per_camera[cam_a].keys()) & set(per_camera[cam_b].keys()))
			if len(common) < MIN_SHARED_IMAGES:
				print(
					f"Not enough shared images for {cam_a} & {cam_b} "
					f"(have {len(common)}, need >= {MIN_SHARED_IMAGES})."
				)
				continue

			objpoints = []
			imgpoints_a = []
			imgpoints_b = []
			for name in common:
				corners_a, ids_a = per_camera[cam_a][name]
				corners_b, ids_b = per_camera[cam_b][name]
				ids_a = ids_a.flatten()
				ids_b = ids_b.flatten()
				shared = sorted(set(ids_a.tolist()) & set(ids_b.tolist()))
				if len(shared) < MIN_SHARED_MARKERS:
					continue

				obj = []
				img_a = []
				img_b = []
				for marker_id in shared:
					obj_corners = marker_map.get(marker_id)
					if obj_corners is None:
						continue
					idx_a = int(np.where(ids_a == marker_id)[0][0])
					idx_b = int(np.where(ids_b == marker_id)[0][0])
					img_corners_a = corners_a[idx_a].reshape(4, 2)
					img_corners_b = corners_b[idx_b].reshape(4, 2)
					obj.extend(obj_corners)
					img_a.extend(img_corners_a)
					img_b.extend(img_corners_b)

				if len(obj) < 4:
					continue
				objpoints.append(np.array(obj, dtype=np.float32))
				imgpoints_a.append(np.array(img_a, dtype=np.float32))
				imgpoints_b.append(np.array(img_b, dtype=np.float32))

			if len(objpoints) < MIN_VALID_PAIRS:
				print(
					f"Not enough valid pairs for {cam_a} & {cam_b} "
					f"(have {len(objpoints)}, need >= {MIN_VALID_PAIRS})."
				)
				continue

			mtx_a = params[cam_a]["mtx"]
			dist_a = params[cam_a]["dist"]
			mtx_b = params[cam_b]["mtx"]
			dist_b = params[cam_b]["dist"]
			image_size = image_sizes.get(cam_a) or image_sizes.get(cam_b)
			if image_size is None:
				print(f"Missing image size for {cam_a} & {cam_b}.")
				continue

			flags = cv2.CALIB_FIX_INTRINSIC
			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
			ret, _mtx_a, _dist_a, _mtx_b, _dist_b, R, T, E, F = cv2.stereoCalibrate(
				objpoints,
				imgpoints_a,
				imgpoints_b,
				mtx_a,
				dist_a,
				mtx_b,
				dist_b,
				image_size,
				criteria=criteria,
				flags=flags,
			)
			if not ret:
				print(f"Stereo calibration failed for {cam_a} & {cam_b}.")
				continue

			key = f"{cam_a}__{cam_b}"
			results[key] = {"R": R, "T": T, "E": E, "F": F}

	os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
	flat = {}
	for key, values in results.items():
		for name, mat in values.items():
			flat[f"{key}_{name}"] = mat
	if not flat:
		raise RuntimeError("No extrinsics computed. Check shared images.")

	np.savez(OUTPUT_FILE, **flat)
	print(f"Saved extrinsics to {OUTPUT_FILE}")


if __name__ == "__main__":
	main()

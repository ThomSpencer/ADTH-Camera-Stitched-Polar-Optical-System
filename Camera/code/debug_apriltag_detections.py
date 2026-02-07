import glob
import itertools
import os
from typing import List

import cv2  # type: ignore

# AprilTag board config (must match your printed board)
APRILTAG_DICT = cv2.aruco.DICT_APRILTAG_36h11
IMAGE_GLOB = "calibration/extrinsics/*/*.jpg"


def _collect_images() -> List[str]:
	return sorted(glob.glob(IMAGE_GLOB))


def main() -> None:
	paths = _collect_images()
	if not paths:
		raise RuntimeError("No images found.")

	dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT)
	detector = cv2.aruco.ArucoDetector(dictionary)

	total = 0
	with_tags = 0
	per_cam = {}
	files_by_cam = {}
	files_with_tags = {}
	for path in paths:
		total += 1
		img = cv2.imread(path)
		if img is None:
			continue
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		corners, ids, _rejected = detector.detectMarkers(gray)
		camera_id = os.path.basename(os.path.dirname(path))
		filename = os.path.basename(path)
		per_cam.setdefault(camera_id, {"total": 0, "with_tags": 0})
		files_by_cam.setdefault(camera_id, set()).add(filename)
		files_with_tags.setdefault(camera_id, set())
		per_cam[camera_id]["total"] += 1
		if ids is not None and len(ids) > 0:
			with_tags += 1
			per_cam[camera_id]["with_tags"] += 1
			files_with_tags[camera_id].add(filename)

	print(f"Total images: {total}")
	print(f"Images with tags: {with_tags}")
	for cam, stats in sorted(per_cam.items()):
		print(f"{cam}: {stats['with_tags']} / {stats['total']} with tags")

	cameras = sorted(files_by_cam.keys())
	if len(cameras) >= 2:
		print("Shared filenames per pair:")
		for cam_a, cam_b in itertools.combinations(cameras, 2):
			shared = files_by_cam[cam_a] & files_by_cam[cam_b]
			shared_tags = files_with_tags[cam_a] & files_with_tags[cam_b]
			print(
				f"{cam_a} & {cam_b}: {len(shared)} shared, "
				f"{len(shared_tags)} shared with tags"
			)


if __name__ == "__main__":
	main()

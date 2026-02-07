import os

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

import cv2  # type: ignore
import numpy as np


# Update these indices based on `v4l2-ctl --list-devices`
CAMERA_INDICES = ["/dev/video2", "/dev/video4", "/dev/video0"]
TARGET_FPS = 30
FRAME_SIZE = (1280, 720)
FOURCC = "MJPG"

# Per-camera names matching calibration folders
CAMERA_NAMES = ["cam0", "cam1", "cam2"]
REFERENCE_CAMERA_NAME = CAMERA_NAMES[1]

INTRINSICS_FILE = "calibration/intrinsics_params_720p.npz"
EXTRINSICS_FILE = "calibration/extrinsics_params.npz"
CANVAS_SIZE = (FRAME_SIZE[0] * len(CAMERA_INDICES), FRAME_SIZE[1])

# Approximate scene plane depth in mm for translation-aware homography.
# Increase if your scene is far; decrease if closer.
SCENE_DEPTH_MM = 10000.0

# Manual trim offsets (pixels) applied after homography, per camera name.
# Positive x moves left, positive y moves down.
CAMERA_TRIM_OFFSETS_PX = {
	"cam0": (0, 0),
	"cam1": (0, 0),
	"cam2": (0, 0),
}

# Manual rotation per camera in degrees (positive = counterclockwise).
CAMERA_ROTATE_DEG = {
	"cam0": 4.75,
	"cam1": 0.0,
	"cam2": 0.0,
}

# Optional crop margins on the final output: (left, right, top, bottom)
OUTPUT_CROP = (0, 0, 0, 0)

# Performance toggles
ENABLE_EXPOSURE_COMP = False
ENABLE_BLENDING = True

# AprilTag detection settings
APRILTAG_DICT = cv2.aruco.DICT_APRILTAG_36h11
TAG_SIZE_MM = 160.0
TEXT_COLOR = (20, 220, 20)
OUTLINE_COLOR = (0, 200, 255)


def read_frame_with_retries(cap, retries=1):
	for _ in range(retries):
		ret, frame = cap.read()
		if ret:
			return True, frame
	return False, None


def open_cameras(indices):
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


def _load_intrinsics(path: str):
	if not os.path.isfile(path):
		return {}
	data = np.load(path)
	params = {}
	for key in data.files:
		if key.endswith("_mtx"):
			camera_id = key[: -len("_mtx")]
			params.setdefault(camera_id, {})["mtx"] = data[key]
		if key.endswith("_dist"):
			camera_id = key[: -len("_dist")]
			params.setdefault(camera_id, {})["dist"] = data[key]
	return params


def _load_extrinsics(path: str):
	if not os.path.isfile(path):
		return {}
	data = np.load(path)
	params = {}
	for key in data.files:
		params[key] = data[key]
	return params


def _get_extrinsics_to_ref(extrinsics, cam_name: str, ref_name: str):
	if cam_name == ref_name:
		return np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)

	key_forward_r = f"{cam_name}__{ref_name}_R"
	key_forward_t = f"{cam_name}__{ref_name}_T"
	if key_forward_r in extrinsics and key_forward_t in extrinsics:
		return extrinsics[key_forward_r], extrinsics[key_forward_t]

	key_inverse_r = f"{ref_name}__{cam_name}_R"
	key_inverse_t = f"{ref_name}__{cam_name}_T"
	if key_inverse_r in extrinsics and key_inverse_t in extrinsics:
		R = extrinsics[key_inverse_r]
		T = extrinsics[key_inverse_t]
		return R.T, -R.T @ T

	return None, None


def _create_apriltag_detector():
	dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT)
	parameters = cv2.aruco.DetectorParameters()
	if hasattr(cv2.aruco, "ArucoDetector"):
		return cv2.aruco.ArucoDetector(dictionary, parameters), dictionary, parameters
	return None, dictionary, parameters


def _warp_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
	points = points.astype(np.float32)
	return cv2.perspectiveTransform(points, H)


def main() -> None:
	cameras = open_cameras(CAMERA_INDICES)
	if len(cameras) != len(CAMERA_INDICES):
		print("Not all cameras could be opened.")
		for _, cap in cameras:
			cap.release()
		return

	if len(CAMERA_NAMES) != len(CAMERA_INDICES):
		print("CAMERA_NAMES must match CAMERA_INDICES length.")
		for _, cap in cameras:
			cap.release()
		return

	intrinsics = _load_intrinsics(INTRINSICS_FILE)
	extrinsics = _load_extrinsics(EXTRINSICS_FILE)
	if not intrinsics:
		print("Missing intrinsics. Run calibration first.")
		for _, cap in cameras:
			cap.release()
		return

	ref_params = intrinsics.get(REFERENCE_CAMERA_NAME)
	if not ref_params:
		print(f"Missing intrinsics for reference {REFERENCE_CAMERA_NAME}.")
		for _, cap in cameras:
			cap.release()
		return
	K_ref = ref_params["mtx"]

	offset_x = (CANVAS_SIZE[0] - FRAME_SIZE[0]) // 2
	translate = np.array([[1.0, 0.0, float(offset_x)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

	homographies = {}
	plane_normal = np.array([[0.0], [0.0], [1.0]], dtype=np.float32)
	undistort_maps = {}
	extrinsics_to_ref = {}
	for cam_name in CAMERA_NAMES:
		params = intrinsics.get(cam_name)
		if not params:
			print(f"Missing intrinsics for {cam_name}.")
			continue
		K_cam = params["mtx"]
		dist = params["dist"]
		map1, map2 = cv2.initUndistortRectifyMap(
			K_cam,
			dist,
			None,
			K_cam,
			FRAME_SIZE,
			cv2.CV_16SC2,
		)
		undistort_maps[cam_name] = (map1, map2)
		R, T = _get_extrinsics_to_ref(extrinsics, cam_name, REFERENCE_CAMERA_NAME)
		if R is None or T is None:
			print(f"Missing rotation for {cam_name} -> {REFERENCE_CAMERA_NAME}.")
			continue
		extrinsics_to_ref[cam_name] = (R, T)
		H = K_ref @ (R + (T @ plane_normal.T) / SCENE_DEPTH_MM) @ np.linalg.inv(K_cam)
		rot_deg = CAMERA_ROTATE_DEG.get(cam_name, 0.0)
		if rot_deg:
			rot2 = cv2.getRotationMatrix2D(
				(FRAME_SIZE[0] / 2.0, FRAME_SIZE[1] / 2.0),
				rot_deg,
				1.0,
			)
			rot3 = np.vstack([rot2, [0.0, 0.0, 1.0]]).astype(np.float32)
		else:
			rot3 = np.eye(3, dtype=np.float32)
		offset_x_px, offset_y_px = CAMERA_TRIM_OFFSETS_PX.get(cam_name, (0, 0))
		shift = np.array(
			[[1.0, 0.0, float(offset_x_px)], [0.0, 1.0, float(offset_y_px)], [0.0, 0.0, 1.0]],
			dtype=np.float32,
		)
		homographies[cam_name] = shift @ translate @ H @ rot3

	detector, dictionary, parameters = _create_apriltag_detector()

	compensator = None
	if ENABLE_EXPOSURE_COMP:
		compensator = cv2.detail.ExposureCompensator_createDefault(
			cv2.detail.ExposureCompensator_GAIN_BLOCKS
		)
	blender = None
	if ENABLE_BLENDING:
		blender = cv2.detail_FeatherBlender()
		blender.setSharpness(0.02)
		blender.prepare((0, 0, CANVAS_SIZE[0], CANVAS_SIZE[1]))

	zero_dist = np.zeros((5, 1), dtype=np.float32)

	cv2.namedWindow("Stitched", cv2.WINDOW_NORMAL)

	try:
		while True:
			frames = []
			for idx, cap in cameras:
				ret, frame = read_frame_with_retries(cap, retries=1)
				if not ret:
					print(f"Failed to grab frame from camera {idx}")
					frames = []
					break
				frames.append(frame)

			if not frames:
				continue

			resized = [cv2.resize(frame, FRAME_SIZE) for frame in frames]

			warped_images = []
			warped_masks = []
			corners = []
			tracked_tags = []
			for (idx, _), frame in zip(cameras, resized):
				cam_index = CAMERA_INDICES.index(idx)
				cam_name = CAMERA_NAMES[cam_index]
				params = intrinsics.get(cam_name)
				if not params:
					continue
				maps = undistort_maps.get(cam_name)
				if maps is None:
					continue
				H = homographies.get(cam_name)
				if H is None:
					continue
				R_to_ref, T_to_ref = extrinsics_to_ref.get(cam_name, (None, None))
				if R_to_ref is None or T_to_ref is None:
					continue
				frame_undist = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)

				if detector is not None:
					marker_corners, marker_ids, _ = detector.detectMarkers(frame_undist)
				else:
					marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
						frame_undist,
						dictionary,
						parameters=parameters,
					)

				if marker_ids is not None and len(marker_ids) > 0:
					rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
						marker_corners,
						TAG_SIZE_MM,
						params["mtx"],
						zero_dist,
					)
					for i, marker_id in enumerate(marker_ids.flatten()):
						corners_np = marker_corners[i].reshape(4, 2)
						center = corners_np.mean(axis=0)
						center_h = np.array([[center]], dtype=np.float32)
						center_warped = _warp_points(center_h, H)[0, 0]

						tvec = tvecs[i].reshape(3, 1)
						tvec_ref = R_to_ref @ tvec + T_to_ref
						tracked_tags.append(
							{
								"id": int(marker_id),
								"pos_ref": tvec_ref.flatten(),
								"center": center_warped,
								"corners": marker_corners[i],
								"H": H,
							}
						)

				warped = cv2.warpPerspective(frame_undist, H, CANVAS_SIZE)
				mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
				_, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
				warped_images.append(warped)
				warped_masks.append(mask)
				corners.append((0, 0))

			if not warped_images:
				continue

			if compensator is not None:
				compensator.feed(corners=corners, images=warped_images, masks=warped_masks)
				for i in range(len(warped_images)):
					compensator.apply(i, corners[i], warped_images[i], warped_masks[i])

			if blender is not None:
				for image, mask in zip(warped_images, warped_masks):
					blender.feed(image.astype(np.int16), mask, (0, 0))
				result, _result_mask = blender.blend(None, None)
				result = cv2.convertScaleAbs(result)
				blender.prepare((0, 0, CANVAS_SIZE[0], CANVAS_SIZE[1]))
			else:
				result = warped_images[0]

			left, right, top, bottom = OUTPUT_CROP
			offset_crop = (0, 0)
			if left or right or top or bottom:
				h, w = result.shape[:2]
				x0 = min(max(left, 0), w)
				x1 = max(min(w - right, w), x0)
				y0 = min(max(top, 0), h)
				y1 = max(min(h - bottom, h), y0)
				result = result[y0:y1, x0:x1]
				offset_crop = (x0, y0)

			for tag in tracked_tags:
				center = tag["center"]
				cx, cy = int(center[0] - offset_crop[0]), int(center[1] - offset_crop[1])
				pos_ref = tag["pos_ref"]
				label = f"ID {tag['id']} x={pos_ref[0]:.1f} y={pos_ref[1]:.1f} z={pos_ref[2]:.1f} mm"
				cv2.circle(result, (cx, cy), 4, OUTLINE_COLOR, -1)
				cv2.putText(
					result,
					label,
					(cx + 6, cy + 18),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5,
					TEXT_COLOR,
					1,
					cv2.LINE_AA,
				)

				corners_warped = _warp_points(tag["corners"], tag["H"]).reshape(4, 2)
				corners_warped[:, 0] -= offset_crop[0]
				corners_warped[:, 1] -= offset_crop[1]
				poly = corners_warped.astype(int).reshape((-1, 1, 2))
				cv2.polylines(result, [poly], True, OUTLINE_COLOR, 2)

			cv2.imshow("Stitched", result)

			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
	finally:
		for _, cap in cameras:
			cap.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

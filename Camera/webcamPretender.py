import os

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

import cv2  # type: ignore
import numpy as np
import math
try:
	import pyvirtualcam
except Exception:  # pragma: no cover
	pyvirtualcam = None


# Update these indices based on `v4l2-ctl --list-devices`
CAMERA_INDICES = ["/dev/video2", "/dev/video0","/dev/video4"]
TARGET_FPS = 30
FRAME_SIZE = (1280, 720)
FOURCC = "MJPG"

# Per-camera names matching calibration folders
CAMERA_NAMES = ["cam0", "cam1", "cam2"]
REFERENCE_CAMERA_NAME = CAMERA_NAMES[1]

INTRINSICS_FILE = "code/calibration/intrinsics_params_720p.npz"
EXTRINSICS_FILE = "code/calibration/extrinsics_params.npz"
CANVAS_SIZE = (FRAME_SIZE[0] * len(CAMERA_INDICES), FRAME_SIZE[1])

# Approximate scene plane depth in mm for translation-aware homography.
SCENE_DEPTH_MM = 10000.0

# Positive x moves left, positive y moves down.
CAMERA_TRIM_OFFSETS_PX = {
	"cam0": (0, -30),
	"cam1": (0, 0),
	"cam2": (8, -15),
}

# Manual rotation per camera in degrees (positive = counterclockwise).
CAMERA_ROTATE_DEG = {
	"cam0": 0.0,
	"cam1": 0.0,
	"cam2": 0.0,
}

# Manual tilt per camera in degrees (positive = pitch up, negative = pitch down).
# Use this if a camera is physically leaning forward/back.
CAMERA_TILT_DEG = {
	"cam0": -7.0,
	"cam1": 0.0,
	"cam2": 0.0,
}

# Optional per-camera crop margins (left, right, top, bottom) in pixels.
# Use this to trim distorted edges before warping.
CAMERA_CROP_PX = {
	"cam0": (0, 200, 0, 0),
	"cam1": (0, 0, 0, 0),
	"cam2": (0, 0, 0, 0),
}

# Optional crop margins on the final output: (left, right, top, bottom)
OUTPUT_CROP = (0, 0, 0, 0)

# Performance toggles
ENABLE_EXPOSURE_COMP = False
ENABLE_BLENDING = True

# Virtual camera output (v4l2loopback)
OUTPUT_DEVICE = "/dev/video10"
OUTPUT_FPS = TARGET_FPS
OUTPUT_SIZE = FRAME_SIZE  # set to consumer size, e.g. (640, 480)
OUTPUT_FOURCC = "YUYV"
OUTPUT_FOURCC_FALLBACKS = ["MJPG", "YUYV"]
USE_GSTREAMER_OUTPUT = True
ENABLE_PYVIRTUALCAM = True


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


def _open_output_writer():
	if USE_GSTREAMER_OUTPUT:
		pipeline = (
			"appsrc ! videoconvert ! video/x-raw,format=YUY2 "
			f"! v4l2sink device={OUTPUT_DEVICE}"
		)
		writer = cv2.VideoWriter(
			pipeline,
			cv2.CAP_GSTREAMER,
			0,
			OUTPUT_FPS,
			OUTPUT_SIZE,
		)
		if writer.isOpened():
			print(f"Opened virtual device {OUTPUT_DEVICE} with GStreamer")
			return writer
		writer.release()

	attempts = []
	if OUTPUT_FOURCC not in OUTPUT_FOURCC_FALLBACKS:
		attempts.append(OUTPUT_FOURCC)
	attempts.extend(OUTPUT_FOURCC_FALLBACKS)

	for fourcc in attempts:
		writer = cv2.VideoWriter(
			OUTPUT_DEVICE,
			cv2.CAP_V4L2,
			cv2.VideoWriter_fourcc(*fourcc),
			OUTPUT_FPS,
			OUTPUT_SIZE,
		)
		if writer.isOpened():
			print(f"Opened virtual device {OUTPUT_DEVICE} with {fourcc}")
			return writer
		writer.release()

	writer = cv2.VideoWriter(
		OUTPUT_DEVICE,
		cv2.VideoWriter_fourcc(*OUTPUT_FOURCC),
		OUTPUT_FPS,
		OUTPUT_SIZE,
	)
	if writer.isOpened():
		print(f"Opened virtual device {OUTPUT_DEVICE} with default backend")
		return writer
	writer.release()
	return None


def _open_virtualcam():
	if not ENABLE_PYVIRTUALCAM or pyvirtualcam is None:
		return None
	try:
		cam = pyvirtualcam.Camera(
			width=OUTPUT_SIZE[0],
			height=OUTPUT_SIZE[1],
			fps=OUTPUT_FPS,
			device=OUTPUT_DEVICE,
		)
		print(f"Opened virtual device {OUTPUT_DEVICE} with pyvirtualcam")
		return cam
	except Exception as exc:
		print(f"pyvirtualcam failed: {exc}")
		return None


def _resize_letterbox(frame: np.ndarray) -> np.ndarray:
	if frame.shape[:2] == (OUTPUT_SIZE[1], OUTPUT_SIZE[0]):
		return frame
	h, w = frame.shape[:2]
	target_w, target_h = OUTPUT_SIZE
	scale = min(target_w / w, target_h / h)
	new_w = max(1, int(w * scale))
	new_h = max(1, int(h * scale))
	resized = cv2.resize(frame, (new_w, new_h))
	canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
	pad_x = (target_w - new_w) // 2
	pad_y = (target_h - new_h) // 2
	canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
	return canvas


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
		tilt_deg = CAMERA_TILT_DEG.get(cam_name, 0.0)
		if tilt_deg:
			rad = math.radians(tilt_deg)
			c = math.cos(rad)
			s = math.sin(rad)
			R_tilt = np.array(
				[[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
				dtype=np.float32,
			)
			R_use = R @ R_tilt
		else:
			R_use = R
		H = K_ref @ (R_use + (T @ plane_normal.T) / SCENE_DEPTH_MM) @ np.linalg.inv(K_cam)
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

	virtual_cam = _open_virtualcam()
	output_writer = None
	if virtual_cam is None:
		output_writer = _open_output_writer()
	if virtual_cam is None and output_writer is None:
		print(
			f"Failed to open virtual device {OUTPUT_DEVICE}. "
			"Check v4l2loopback and container device permissions."
		)
		for _, cap in cameras:
			cap.release()
		return

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
			for (idx, _), frame in zip(cameras, resized):
				cam_index = CAMERA_INDICES.index(idx)
				cam_name = CAMERA_NAMES[cam_index]
				params = intrinsics.get(cam_name)
				if not params:
					continue
				maps = undistort_maps.get(cam_name)
				if maps is None:
					continue
				frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
				H = homographies.get(cam_name)
				if H is None:
					continue
				crop_left, crop_right, crop_top, crop_bottom = CAMERA_CROP_PX.get(
					cam_name, (0, 0, 0, 0)
				)
				h, w = frame.shape[:2]
				x0 = min(max(crop_left, 0), w)
				x1 = max(min(w - crop_right, w), x0)
				y0 = min(max(crop_top, 0), h)
				y1 = max(min(h - crop_bottom, h), y0)
				if x0 != 0 or y0 != 0 or x1 != w or y1 != h:
					frame = frame[y0:y1, x0:x1]
					T_crop = np.array(
						[[1.0, 0.0, float(x0)], [0.0, 1.0, float(y0)], [0.0, 0.0, 1.0]],
						dtype=np.float32,
					)
					H_use = H @ T_crop
				else:
					H_use = H
				warped = cv2.warpPerspective(frame, H_use, CANVAS_SIZE)
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
			if left or right or top or bottom:
				h, w = result.shape[:2]
				x0 = min(max(left, 0), w)
				x1 = max(min(w - right, w), x0)
				y0 = min(max(top, 0), h)
				y1 = max(min(h - bottom, h), y0)
				result = result[y0:y1, x0:x1]

			out_frame = _resize_letterbox(result)
			if virtual_cam is not None:
				rgb_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
				virtual_cam.send(rgb_frame)
				virtual_cam.sleep_until_next_frame()
			elif output_writer is not None:
				output_writer.write(out_frame)

			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
	finally:
		for _, cap in cameras:
			cap.release()
		if output_writer is not None:
			output_writer.release()
		if virtual_cam is not None:
			virtual_cam.close()


if __name__ == "__main__":
	main()

import cv2  # type: ignore

# Update these indices based on `v4l2-ctl --list-devices`
CAMERA_INDICES = ["/dev/video2", "/dev/video4", "/dev/video0"]
TARGET_FPS = 30
FRAME_SIZE = (640, 480)
FOURCC = "MJPG"


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
		# Warm up the camera by grabbing a few frames
		read_frame_with_retries(cap, retries=3)
		opened.append((idx, cap))
		print(f"Successfully opened camera index {idx}")
	return opened

def draw_center_dot(frame, color=(0, 0, 255), radius=6, thickness=-1):
	h, w = frame.shape[:2]
	center = (w // 2, h // 2)
	cv2.circle(frame, center, radius, color, thickness)


def main() -> None:
	cameras = open_cameras(CAMERA_INDICES)
	if len(cameras) != len(CAMERA_INDICES):
		print("Not all cameras could be opened.")
		for _, cap in cameras:
			cap.release()
		return

	cv2.namedWindow("Stitched", cv2.WINDOW_NORMAL)

	stitcher = cv2.Stitcher_create(mode=cv2.Stitcher_PANORAMA)

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
			for frame in resized:
				draw_center_dot(frame)

			try:
				status, stitched = stitcher.stitch(resized)
			except cv2.error as exc:
				print(f"Stitcher error: {exc}")
				status, stitched = None, None

			if status == cv2.Stitcher_OK and stitched is not None:
				cv2.imshow("Stitched", stitched)
			else:
				fallback = cv2.hconcat(resized)
				cv2.imshow("Stitched", fallback)

			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
	finally:
		for _, cap in cameras:
			cap.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

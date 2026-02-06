import cv2  # type: ignore

# Update these indices based on `v4l2-ctl --list-devices`
CAMERA_INDICES = ["/dev/video0", "/dev/video5", "/dev/video3"]
TARGET_FPS = 30
FRAME_SIZE = (1080, 720)
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


def main():
    cameras = open_cameras(CAMERA_INDICES)
    if not cameras:
        print("No cameras could be opened.")
        return

    for idx, _ in cameras:
        cv2.namedWindow(f"Camera {idx}", cv2.WINDOW_NORMAL)

    try:
        while True:
            for idx, cap in cameras:
                ret, frame = read_frame_with_retries(cap, retries=1)
                if not ret:
                    print(f"Failed to grab frame from camera {idx}")
                    continue
                cv2.imshow(f"Camera {idx}", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for _, cap in cameras:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
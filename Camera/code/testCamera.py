import cv2  # type: ignore
import numpy as np # type: ignore

# Update these indices based on `v4l2-ctl --list-devices`
CAMERA_INDICES = ["/dev/video2", "/dev/video4", "/dev/video0"]
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
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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

    window_name = "Cameras"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    blank_frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)

    try:
        while True:
            frames = []
            for idx, cap in cameras:
                ret, frame = read_frame_with_retries(cap, retries=1)
                if not ret:
                    print(f"Failed to grab frame from camera {idx}")
                    frame = blank_frame.copy()
                else:
                    if (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
                        frame = cv2.resize(frame, FRAME_SIZE)
                frames.append(frame)

            if frames:
                stitched = cv2.hconcat(frames)
                cv2.imshow(window_name, stitched)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for _, cap in cameras:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
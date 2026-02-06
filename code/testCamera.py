import cv2  # type: ignore

# Update these indices based on `v4l2-ctl --list-devices`
CAMERA_INDICES = ["/dev/video0", "/dev/video3", "/dev/video5"]


def read_frame_with_retries(cap, retries=3):
    for _ in range(retries):
        ret, frame = cap.read()
        if ret:
            return True, frame
    return False, None


def open_cameras(indices):
    opened = []
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"Failed to open camera index {idx}")
            cap.release()
            input("Press Enter to continue...")
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Warm up the camera by grabbing a few frames
        read_frame_with_retries(cap, retries=5)
        opened.append((idx, cap))
    return opened


def main():
    cameras = open_cameras(CAMERA_INDICES)
    if not cameras:
        print("No cameras could be opened.")
        return

    try:
        while True:
            for idx, cap in cameras:
                ret, frame = read_frame_with_retries(cap, retries=3)
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
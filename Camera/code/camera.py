import cv2  # type: ignore

CAMERA_PATH = "/dev/video0"


def main() -> None:
    camera = cv2.VideoCapture(CAMERA_PATH, cv2.CAP_V4L2)
    if not camera.isOpened():
        print(f"Failed to open camera {CAMERA_PATH}")
        return

    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

    try:
        while True:
            # Flush a couple frames to avoid showing stale images
            camera.grab()
            camera.grab()

            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame")
                break

            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
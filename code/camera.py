import cv2  # type: ignore

camera  = cv2.VideoCapture("/dev/video3", cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
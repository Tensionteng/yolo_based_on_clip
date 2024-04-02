import cv2

from ultralytics import YOLO

model = YOLO("runs/obb/train2/weights/best.pt")

# Open the video file
video_path = "videos/bilibili_demo.mp4"
cap = cv2.VideoCapture(video_path)
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)
video_writer = cv2.VideoWriter(
    "results/obb_track2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot(conf=False, line_width=2, font_size=12)

        video_writer.write(annotated_frame)

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
video_writer.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
from pupil_apriltags import Detector
from ultralytics import YOLO


def get_roi(frame):
    h, w = frame.shape[:2]
    y1 = int(h * 0.55)
    y2 = h
    x1 = 0
    x2 = w
    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


def get_masks(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([15, 80, 80])
    yellow_upper = np.array([40, 255, 255])

    white_lower = np.array([0, 0, 180])
    white_upper = np.array([180, 70, 255])

    red_lower1 = np.array([0, 120, 120])
    red_upper1 = np.array([10, 255, 255])

    red_lower2 = np.array([170, 120, 120])
    red_upper2 = np.array([180, 255, 255])

    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    return yellow_mask, white_mask, red_mask


def get_centroid(mask):
    M = cv2.moments(mask)
    if M["m00"] < 800:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def get_lane_command(yellow_center, white_center, red_center, roi_width):
    if red_center is not None:
        return "STOP"

    if yellow_center is not None and white_center is not None:
        lane_center = (yellow_center[0] + white_center[0]) // 2
    elif yellow_center is not None:
        lane_center = yellow_center[0] + 120
    elif white_center is not None:
        lane_center = white_center[0] - 120
    else:
        return "SEARCH"

    image_center = roi_width // 2
    error = lane_center - image_center

    if abs(error) <= 40:
        return "FORWARD"
    elif error < 0:
        return "LEFT"
    else:
        return "RIGHT"


def draw_tag(frame, tag):
    corners = tag.corners.astype(int)
    for i in range(4):
        p1 = tuple(corners[i])
        p2 = tuple(corners[(i + 1) % 4])
        cv2.line(frame, p1, p2, (255, 0, 255), 2)

    center = tuple(tag.center.astype(int))
    cv2.circle(frame, center, 6, (255, 0, 255), -1)

    cv2.putText(
        frame,
        f"AprilTag id={tag.tag_id}",
        (center[0] + 10, center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
        cv2.LINE_AA
    )


def draw_objects(frame, results):
    object_names = []

    if not results:
        return object_names

    boxes = results[0].boxes
    names = results[0].names

    if boxes is None:
        return object_names

    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = names.get(cls_id, str(cls_id))

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        object_names.append(label)

    return object_names


def draw_output(frame, roi_box, yellow_center, white_center, red_center, command, tags, object_names):
    x1, y1, x2, y2 = roi_box
    roi_width = x2 - x1

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    image_center_x = x1 + roi_width // 2
    cv2.line(frame, (image_center_x, y1), (image_center_x, y2), (255, 0, 0), 2)

    if yellow_center is not None:
        cx, cy = yellow_center
        cv2.circle(frame, (x1 + cx, y1 + cy), 8, (0, 255, 255), -1)

    if white_center is not None:
        cx, cy = white_center
        cv2.circle(frame, (x1 + cx, y1 + cy), 8, (255, 255, 255), -1)

    if red_center is not None:
        cx, cy = red_center
        cv2.circle(frame, (x1 + cx, y1 + cy), 8, (0, 0, 255), -1)

    for tag in tags:
        draw_tag(frame, tag)

    cv2.putText(
        frame,
        f"COMMAND: {command}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    if tags:
        cv2.putText(
            frame,
            f"APRILTAGS: {len(tags)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )

    if object_names:
        text = "OBJECTS: " + ", ".join(object_names[:4])
        cv2.putText(
            frame,
            text,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return frame


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    april_detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    # small YOLO model
    yolo_model = YOLO("yolov8n.pt")

    print("Press q to quit.")

    frame_count = 0
    object_names = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        frame = cv2.resize(frame, (960, 540))

        roi, roi_box = get_roi(frame)
        yellow_mask, white_mask, red_mask = get_masks(roi)

        yellow_center = get_centroid(yellow_mask)
        white_center = get_centroid(white_mask)
        red_center = get_centroid(red_mask)

        lane_command = get_lane_command(
            yellow_center, white_center, red_center, roi.shape[1]
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = april_detector.detect(gray)

        # run object detection every 5 frames for smoother laptop performance
        if frame_count % 5 == 0:
            results = yolo_model(frame, verbose=False)
            object_names = draw_objects(frame, results)

        frame_count += 1

        if tags:
            command = f"APRILTAG_{tags[0].tag_id}"
        else:
            command = lane_command

        print(command)

        output = draw_output(
            frame,
            roi_box,
            yellow_center,
            white_center,
            red_center,
            command,
            tags,
            object_names
        )

        # only one main window now
        cv2.imshow("Camera Output", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
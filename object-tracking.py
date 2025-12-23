import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

def get_bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

def get_sam_mask(frame, predictor):
    r = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=False)
    x, y, w, h = r
    if w == 0 or h == 0:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    box = np.array([x, y, x + w, y + h])
    masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
    best_idx = np.argmax(scores)
    return masks[best_idx].astype(np.uint8)

def is_grayscale(frame):
    if len(frame.shape) == 2:
        return True
    if frame.shape[2] == 3:
        b, g, r = cv2.split(frame)
        if np.array_equal(b, g) and np.array_equal(g, r):
            return True
    return False

def run_meanshift(cap, track_window, roi_hist, use_grayscale=False):
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if use_grayscale:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            hsv[:, :, 2] = gray
            backproj = cv2.calcBackProject([hsv], [2], roi_hist, [0, 256], 1)
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            backproj = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

        _, backproj = cv2.threshold(backproj, 50, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        backproj = cv2.morphologyEx(backproj, cv2.MORPH_OPEN, kernel)

        _, track_window = cv2.meanShift(backproj, track_window, term_crit)

        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)
        cv2.imshow("Back Projection", backproj)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def track_blue_ball(video_path, predictor):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Cannot read blue-ball video.")
        cap.release()
        return

    mask = get_sam_mask(frame, predictor)
    if mask is None:
        print("No mask for blue ball.")
        cap.release()
        return

    bbox = get_bbox_from_mask(mask)
    if bbox is None:
        print("No bbox for blue ball.")
        cap.release()
        return

    x, y, w, h = bbox
    track_window = (x, y, w, h)

    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_roi = (mask[y:y + h, x:x + w] * 255).astype(np.uint8)

    roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask_roi, [180, 256], [0, 180, 0, 256])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    run_meanshift(cap, track_window, roi_hist, use_grayscale=False)

def track_person(video_path, predictor):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Cannot read person video.")
        cap.release()
        return

    gray_mode = is_grayscale(frame)
    if gray_mode and len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    mask = get_sam_mask(frame, predictor)
    if mask is None:
        print("No mask for person.")
        cap.release()
        return

    bbox = get_bbox_from_mask(mask)
    if bbox is None:
        print("No bbox for person.")
        cap.release()
        return

    x, y, w, h = bbox
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame.shape[1] - x, w + 2 * padding)
    h = min(frame.shape[0] - y, h + 2 * padding)
    track_window = (x, y, w, h)

    if gray_mode:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray_frame[y:y + h, x:x + w]
        mask_roi = (mask[y:y + h, x:x + w] * 255).astype(np.uint8)
        roi_hist = cv2.calcHist([roi], [0], mask_roi, [256], [0, 256])
    else:
        roi = frame[y:y + h, x:x + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_roi = (mask[y:y + h, x:x + w] * 255).astype(np.uint8)
        roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask_roi, [180, 256], [0, 180, 0, 256])

    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    run_meanshift(cap, track_window, roi_hist, use_grayscale=gray_mode)

if __name__ == "__main__":
    sam_checkpoint = r"sam_vit_b_01ec64.pth"
    model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    predictor = SamPredictor(model)

    track_blue_ball("Sample 1.mp4", predictor)
    track_person("Sample 2.mp4", predictor)

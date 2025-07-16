import os
import cv2
import math
import numpy as np
import torch
import mediapipe as mp

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "tmp", "comfyui_debug_crop")
os.makedirs(DEBUG_DIR, exist_ok=True)

TARGET_RATIO = 11 / 14

def debug_save(name, img):
    path = os.path.join(DEBUG_DIR, name)
    try:
        cv2.imwrite(path, img)
        print(f"🖼️  Saved debug image: {path}")
    except Exception as e:
        print(f"❌ Failed to save debug image: {e}")

def tensor2np(image):
    if not isinstance(image, torch.Tensor):
        raise ValueError("Input is not a torch.Tensor")
    shape = image.shape
    print(f"[DEBUG] tensor shape: {shape}")
    if len(shape) == 4 and shape[1] == 3:
        arr = image[0].cpu().numpy().transpose(1, 2, 0) * 255.0
    elif len(shape) == 4 and shape[3] == 3:
        arr = image[0].cpu().numpy() * 255.0
    elif len(shape) == 3 and shape[0] == 3:
        arr = image.cpu().numpy().transpose(1, 2, 0) * 255.0
    else:
        raise ValueError(f"[❌] Unknown image tensor shape: {shape}")
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def np2tensor(image_np):
    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    if image_np.shape[2] != 3:
        raise ValueError("np2tensor: Invalid channel count")
    # BGR → RGB 변환
    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # 채널-마지막 텐서 생성 (H,W,3) -> (1,H,W,3)
    tensor = torch.from_numpy(img_rgb)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # 배치 차원 추가
    return tensor

def crop_to_ratio(img, ratio):
    h, w = img.shape[:2]
    current_ratio = w / h
    if current_ratio > ratio:
        new_w = int(h * ratio)
        offset = (w - new_w) // 2
        return img[:, offset:offset+new_w]
    elif current_ratio < ratio:
        new_h = int(w / ratio)
        offset = (h - new_h) // 2
        return img[offset:offset+new_h, :]
    return img

def auto_crop_document(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 30, 150)
    debug_save("step_1a_gray.png", gray)
    debug_save("step_1b_blur.png", blur)
    debug_save("step_1c_edge.png", edged)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        if cv2.contourArea(contour) < 50000:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            break
    else:
        print("⚠️ No valid document contour found.")
        return img
    def order_points(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        return np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], dtype="float32")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    maxWidth = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    maxHeight = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0,0], [maxWidth-1,0], [maxWidth-1,maxHeight-1], [0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

def rotate_preserve_canvas(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

def get_correct_rotation(image):
    best_angle = 0
    best_score = -float("inf")
    final_rotated = image
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for angle in [0, 90, 180, 270]:
            rotated = rotate_preserve_canvas(image, angle)
            rotated_small = cv2.resize(rotated, (min(640, rotated.shape[1]), min(640, rotated.shape[0])))
            image_rgb = cv2.cvtColor(rotated_small, cv2.COLOR_BGR2RGB)
            debug_save(f"step_2a_input_angle_{angle}.png", rotated_small)
            results = face_mesh.process(image_rgb)
            if not results.multi_face_landmarks:
                continue
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            nose = landmarks[1]
            mouth = landmarks[13]
            eye_avg_y = (left_eye.y + right_eye.y) / 2
            if eye_avg_y < nose.y < mouth.y:
                vertical_score = (nose.y - eye_avg_y) + (mouth.y - nose.y)
                if vertical_score > best_score:
                    best_score = vertical_score
                    best_angle = angle
                    final_rotated = rotated
    return best_angle, final_rotated

class CropRotateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "process"
    CATEGORY = "image/transform"

    def process(self, image):
        image_np = tensor2np(image)
        debug_save("step_0_input.png", image_np)
        doc_cropped = auto_crop_document(image_np)
        debug_save("step_1_doc_crop.png", doc_cropped)
        angle, rotated = get_correct_rotation(doc_cropped)
        debug_save("step_2_rotated.png", rotated)
        final_cropped = crop_to_ratio(rotated, TARGET_RATIO)
        debug_save("step_4_final_cropped.png", final_cropped)
        tensor = np2tensor(final_cropped)
        return (tensor,)

NODE_CLASS_MAPPINGS = {"CropRotateNode": CropRotateNode}

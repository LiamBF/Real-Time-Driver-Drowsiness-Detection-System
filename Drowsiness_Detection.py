"""

--- DDD LIVE INFERENCE DASHBOARD: OPERATIONAL GUIDELINES ---

1. HARDWARE REQUIREMENTS:
   - CAMERA: A functional webcam (internal or USB) must be connected.
   - AVAILABILITY: The camera must not be in use by other applications (e.g., Zoom, Teams).
   - LIGHTING: The environment should be well-lit; however, the model is trained 
     to handle varied lighting thanks to ImageNet normalization.

2. DIRECTORY & MODEL SETUP:
   - ROOT FOLDER: This script looks for a directory named 'DDD_Tests'.
   - SUBFOLDERS: Each experiment folder (e.g., 'DDD_Test_2epochs') must contain a '.pth' file.
   - MODEL LOADING: The system dynamically searches these subfolders to allow 
     the user to compare different training versions of the MobileNetV3 model.

3. USER POSITIONING (FOR OPTIMAL ACCURACY):
   - VIEW: The driver must have a clear, unobstructed, full-face view within the frame.
   - DISTANCE: Ensure the head and shoulders are visible to allow the model 
     to accurately extract facial landmarks.
   - OBSTRUCTIONS: Large sunglasses or hair covering the eyes may increase 
     latency or reduce the classification accuracy.

4. REAL-TIME PERFORMANCE:
   - The dashboard provides a live Latency reading.
   - Target Performance: 15ms - 25ms.
   - Alert System: A red border and 'DROWSY' label will trigger instantly upon 
     fatigue detection, simulating an integrated vehicle safety alert.

"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox
import os
import time

# --- 1. SETTINGS & MODEL SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory where your test folders are stored
PARENT_DIR = "DDD_Tests"

# Mapping for the binary folders: 'drowsy' (0), 'non drowsy' (1)
class_labels = {0: "DROWSY", 1: "ALERT"}

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def launch_camera(folder_path):
    # Starts the OpenCV camera and handles inference for the selected folder
    # Find .pth file within the specific subfolder
    model_files = [f for f in os.listdir(folder_path) if f.endswith(".pth")]
    if not model_files:
        messagebox.showerror("Error", f"No .pth file found in {folder_path}")
        return
    
    pth_path = os.path.join(folder_path, model_files[0])
    
    try:
        model = models.mobilenet_v3_large(weights=None)
        # Set to 2 classes for the DDD binary dataset
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
        model.load_state_dict(torch.load(pth_path, map_location=device))
        model.to(device).eval()
    except Exception as e:
        messagebox.showerror("Model Error", f"Failed to load model: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Could not access the webcam.")
        return

    print(f"Running Inference with: {pth_path}")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Model Prediction
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            latency = (time.time() - start_time) * 1000

        label = class_labels.get(predicted.item(), "Unknown")
        color = (0, 0, 255) if label == "DROWSY" else (0, 255, 0)

        # UI Overlay
        cv2.putText(frame, f"Model Path: {folder_path}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"STATE: {label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, f"Latency: {latency:.1f}ms", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        if label == "DROWSY":
            cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 15)

        cv2.imshow("DDD Live Inference (Press 'q' to go back)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- 2. GUI CONSTRUCTION ---
root = tk.Tk()
root.title("DDD System Evaluator")
root.geometry("1000x900")
root.configure(bg="#0e1117")

header = tk.Label(root, text="DDD (Drowsiness Detection) Dashboard", 
                 font=("Segoe UI", 24, "bold"), fg="#ff4b4b", bg="#0e1117", pady=25)
header.pack()


canvas = tk.Canvas(root, bg="#0e1117", highlightthickness=0)
scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="#0e1117")

scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True, padx=20)
scrollbar.pack(side="right", fill="y")

# Look inside "DDD_Tests"
if not os.path.exists(PARENT_DIR):
    os.makedirs(PARENT_DIR)

# Get all subfolders inside DDD_Tests
test_folders = sorted([
    os.path.join(PARENT_DIR, f) 
    for f in os.listdir(PARENT_DIR) 
    if os.path.isdir(os.path.join(PARENT_DIR, f))
])

image_refs = []

for folder in test_folders:
    card = tk.Frame(scrollable_frame, bg="#1e2130", pady=15, padx=20, highlightbackground="#3d446d", highlightthickness=1)
    card.pack(fill="x", pady=15, padx=10)

    side_panel = tk.Frame(card, bg="#1e2130")
    side_panel.pack(side="left", fill="y")

    # Display folder name
    display_name = os.path.basename(folder)
    tk.Label(side_panel, text=f"Experiment: {display_name}", font=("Segoe UI", 12, "bold"), fg="#ffffff", bg="#1e2130").pack(anchor="w")
    
    tk.Button(side_panel, text="▶ LIVE TEST", bg="#ff4b4b", fg="white", font=("Segoe UI", 11, "bold"),
              width=18, pady=10, relief="flat", cursor="hand2", 
              command=lambda f=folder: launch_camera(f)).pack(pady=20)

    # Load Graph if it exists in that folder
    graph_path = os.path.join(folder, "results_graph.png")
    if os.path.exists(graph_path):
        try:
            img = Image.open(graph_path)
            img.thumbnail((500, 300))
            photo = ImageTk.PhotoImage(img)
            image_refs.append(photo)
            tk.Label(card, image=photo, bg="#1e2130").pack(side="right")
        except:
            tk.Label(card, text="Graph Not Found", fg="#ff4b4b", bg="#1e2130").pack(side="right")

root.mainloop()
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
from ultralytics import YOLO
import os
import glob
import cv2

# ---- Customization Variables ---- #
CUSTOM_NAMES = ['FireExtinguisher', 'ToolBox', 'OxygenTank']
PROJECT_COLOR = "#00FFC6"
BG_DARK = "#181D27"
HEADER_BG = "#232947"
FOOTER_BG = "#232947"
BTN_COLOR = "#0CB877"
RESULT_BG = "#232947"
RESULT_FG = "#FFD600"
FONT_HEADER = ("Montserrat", 22, "bold")
FONT_BODY = ("Montserrat", 13)
FONT_BOLD = ("Montserrat", 15, "bold")

# ---- Model Load ---- #
model = YOLO("best.pt")

# ---- GUI Setup ---- #
root = tk.Tk()
root.title("🚀 ElectroBeasts - Space Object Detector")
root.geometry("970x780")
root.configure(bg=BG_DARK)

header = tk.Frame(root, bg=HEADER_BG, height=60)
header.pack(fill="x")
header_label = tk.Label(header, text="🛰 ElectroBeasts Space Station Object Detector", font=FONT_HEADER, bg=HEADER_BG, fg=PROJECT_COLOR, pady=12)
header_label.pack(side="left", padx=45)

img_frame = tk.Frame(root, bg=BG_DARK)
img_frame.pack(pady=18)

img_label = tk.Label(img_frame, bg=BG_DARK, borderwidth=3, relief=tk.RIDGE)
img_label.pack(ipadx=3, ipady=3)

result_pane = tk.Frame(root, bg=RESULT_BG)
result_pane.pack(pady=16, padx=14, fill="x")
result_title = tk.Label(result_pane, text="DETECTION RESULTS", font=FONT_BOLD, bg=RESULT_BG, fg=PROJECT_COLOR, anchor="w", pady=6)
result_title.pack(anchor="w")
output_text = tk.Text(result_pane, height=7, width=69, bg=BG_DARK, fg=RESULT_FG, font=FONT_BODY, bd=2, relief=tk.FLAT, wrap=tk.WORD)
output_text.pack(padx=3, pady=(0, 7))
output_text.tag_configure("header", font=FONT_BOLD, foreground=PROJECT_COLOR)
output_text.tag_configure("footer", font=FONT_BOLD, foreground=RESULT_FG)

# ---- Utility ---- #
def clear_previous_outputs(folder="runs/detect/predict"):
    if os.path.exists(folder):
        files = glob.glob(os.path.join(folder, "*"))
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Could not delete {f}: {e}")

# ---- IMAGE DETECTION ---- #
def detect_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return
    output_text.delete(1.0, tk.END)
    img_label.config(image=None)
    clear_previous_outputs()

    results = model.predict(source=file_path, save=True, conf=0.38, project="runs/detect", name="predict", exist_ok=True)

    save_dir = results[0].save_dir
    detected_images = [f for f in os.listdir(save_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not detected_images:
        output_text.insert(tk.END, "❌ No detection image found.\n")
        return

    result_path = os.path.join(save_dir, detected_images[0])
    try:
        img = Image.open(result_path)
        img = ImageOps.pad(img, (684, 512), color=BG_DARK)
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk
    except Exception:
        output_text.insert(tk.END, "❌ Error loading image.")
        return

    if results[0].boxes is not None and len(results[0].boxes.cls) > 0:
        class_indices = [int(x) for x in results[0].boxes.cls]
        detected = [CUSTOM_NAMES[i] for i in class_indices]
        summary = {cls: detected.count(cls) for cls in CUSTOM_NAMES}
        output_text.insert(tk.END, f"🟢 Objects detected:\n", "header")
        for key in summary:
            if summary[key]:
                output_text.insert(tk.END, f"   • {key}: {summary[key]}\n")
        output_text.insert(tk.END, "\nDetection complete! 🚀", "footer")
    else:
        output_text.insert(tk.END, "⚠ No objects detected in the selected image.\n")

# ---- LIVE CAMERA DETECTION ---- #
cap = None
running = False

def stop_camera():
    global cap, running
    running = False
    if cap:
        cap.release()
    img_label.config(image="")
    output_text.insert(tk.END, "\n🛑 Camera stopped.\n", "footer")

def start_camera():
    global cap, running
    output_text.delete(1.0, tk.END)
    stop_camera()
    cap = cv2.VideoCapture(0)
    running = True
    show_frame()

def show_frame():
    global cap, running
    if not running:
        return
    ret, frame = cap.read()
    if not ret:
        output_text.insert(tk.END, "⚠ Error accessing webcam.\n")
        return

    # Run YOLO on frame
    results = model.predict(source=frame, conf=0.4, verbose=False)[0]
    annotated_frame = results.plot()

    # Show frame
    img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageOps.pad(img, (684, 512), color=BG_DARK)
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

    # Detection Output
    output_text.delete(1.0, tk.END)
    if results.boxes and results.boxes.cls is not None:
        classes = [CUSTOM_NAMES[int(i)] for i in results.boxes.cls]
        summary = {cls: classes.count(cls) for cls in CUSTOM_NAMES}
        output_text.insert(tk.END, f"🟢 Live Detection:\n", "header")
        for k in summary:
            if summary[k]:
                output_text.insert(tk.END, f"   • {k}: {summary[k]}\n")
    else:
        output_text.insert(tk.END, "⚠ No objects detected.\n")

    root.after(60, show_frame)

# ---- Footer ---- #
footer = tk.Frame(root, bg=FOOTER_BG, height=40)
footer.pack(side="bottom", fill="x")
lbl_footer = tk.Label(footer, text="ElectroBeasts | AI x Space Safety | Hackathon 2025", font=("Montserrat", 10), bg=FOOTER_BG, fg="#888")
lbl_footer.pack(fill="both", pady=5)

# ---- Buttons ---- #
btn_frame = tk.Frame(root, bg=BG_DARK)
btn_frame.pack(pady=12)

btn_image = tk.Button(btn_frame, text="📂 SELECT IMAGE AND DETECT", command=detect_image, font=("Montserrat", 14, "bold"), bg=BTN_COLOR, fg="white", padx=18, pady=10, bd=0, cursor="hand2")
btn_image.grid(row=0, column=0, padx=10)

btn_cam = tk.Button(btn_frame, text="📸 START LIVE CAMERA", command=start_camera, font=("Montserrat", 14, "bold"), bg=BTN_COLOR, fg="white", padx=18, pady=10, bd=0, cursor="hand2")
btn_cam.grid(row=0, column=1, padx=10)

btn_stop = tk.Button(btn_frame, text="🛑 STOP CAMERA", command=stop_camera, font=("Montserrat", 14, "bold"), bg="#D23B3B", fg="white", padx=18, pady=10, bd=0, cursor="hand2")
btn_stop.grid(row=0, column=2, padx=10)

# ---- Start GUI ---- #
root.mainloop()

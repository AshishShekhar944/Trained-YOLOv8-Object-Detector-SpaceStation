import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import glob

# Load YOLOv8 model
model = YOLO("best.pt")

# GUI setup
root = tk.Tk()
root.title("YOLOv8 Space Object Detector - Team ElectroBeasts")
root.geometry("900x700")
root.configure(bg="#1e1e1e")

# Title
title_label = tk.Label(root, text="🚀 ElectroBeasts - YOLOv8 Object Detection",
                       font=("Helvetica", 20, "bold"), bg="#1e1e1e", fg="#00FFAA")
title_label.pack(pady=15)

# Image display area
img_frame = tk.Frame(root, bg="#1e1e1e")
img_frame.pack()

img_label = tk.Label(img_frame, bg="#1e1e1e")
img_label.pack(pady=10)

# Output area
output_frame = tk.Frame(root, bg="#1e1e1e")
output_frame.pack(pady=10)

output_label = tk.Label(output_frame, text="📝 Detection Output:",
                        font=("Helvetica", 14, "bold"), bg="#1e1e1e", fg="#FFCC00")
output_label.pack(anchor="w")

output_text = tk.Text(output_frame, height=6, width=80,
                      bg="#282828", fg="white", font=("Courier", 12))
output_text.pack()


def clear_previous_outputs(folder="runs/detect/predict"):
    if os.path.exists(folder):
        files = glob.glob(os.path.join(folder, "*"))
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Could not delete {f}: {e}")


def detect_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    output_text.delete(1.0, tk.END)

    # Delete old outputs
    clear_previous_outputs()

    # Run detection
    results = model.predict(
        source=file_path,
        save=True,
        conf=0.4,
        project="runs/detect",
        name="predict",
        exist_ok=True
    )

    # Get output folder
    save_dir = results[0].save_dir

    # Find image
    detected_images = [
        f for f in os.listdir(save_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not detected_images:
        output_text.insert(tk.END, "❌ No result image found.\n")
        return

    result_path = os.path.join(save_dir, detected_images[0])

    # Show image
    try:
        img = Image.open(result_path)
        img = img.resize((640, 480))
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk
    except FileNotFoundError:
        output_text.insert(tk.END, "❌ Error loading image.\n")
        return

    # Show detected objects
    if results[0].boxes is not None and len(results[0].boxes.cls) > 0:
        class_names = results[0].names
        for cls_id in results[0].boxes.cls:
            cls_name = class_names[int(cls_id)]
            output_text.insert(tk.END, f"🟢 Detected: {cls_name}\n")
    else:
        output_text.insert(tk.END, "⚠️ No objects detected.\n")


# Button
btn = tk.Button(
    root,
    text="📂 Select Image and Detect",
    command=detect_image,
    font=("Arial", 16),
    bg="#4CAF50",
    fg="white",
    padx=20,
    pady=10,
    bd=0
)
btn.pack(pady=15)

# Run app
root.mainloop()


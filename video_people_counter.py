import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import torch
from PIL import Image, ImageTk

class PeopleCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video People Counter")
        self.root.geometry("1200x800")
        
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Set model to detect only people (class 0 in COCO dataset)
        self.model.classes = [0]  # 0 is the class ID for people
        
        # Create UI elements
        self.create_widgets()
        
        # Video related variables
        self.cap = None
        self.playing = False
        self.frame_count = 0
        self.current_frame = None
        self.people_count = 0

    def create_widgets(self):
        # Top frame for buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Upload button
        self.upload_btn = tk.Button(btn_frame, text="Upload Video", command=self.upload_video)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Play/Pause button
        self.play_btn = tk.Button(btn_frame, text="Play", command=self.toggle_play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        # People count display
        self.count_label = tk.Label(btn_frame, text="People count: 0", font=("Arial", 14))
        self.count_label.pack(side=tk.RIGHT, padx=20)
        
        # Main frame for video display
        self.video_frame = tk.Label(self.root, bg="black")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready. Upload a video to begin.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def upload_video(self):
        """Open a file dialog to select a video file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if file_path:
            # Release any previously opened video
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            
            # Open the new video
            self.cap = cv2.VideoCapture(file_path)
            
            if not self.cap.isOpened():
                self.status_bar.config(text=f"Error: Could not open video file")
                return
                
            # Get video properties
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Update status
            video_name = file_path.split("/")[-1]
            self.status_bar.config(text=f"Loaded: {video_name} ({width}x{height}, {fps:.2f} fps, {self.frame_count} frames)")
            
            # Enable play button
            self.play_btn.config(state=tk.NORMAL)
            
            # Display the first frame
            ret, self.current_frame = self.cap.read()
            if ret:
                self.detect_and_display(self.current_frame)

    def toggle_play(self):
        """Toggle between play and pause states"""
        if self.playing:
            self.playing = False
            self.play_btn.config(text="Play")
        else:
            self.playing = True
            self.play_btn.config(text="Pause")
            self.play_video()

    def play_video(self):
        """Process and display video frames"""
        if self.cap and self.playing:
            ret, frame = self.cap.read()
            
            if ret:
                self.current_frame = frame
                self.detect_and_display(frame)
                
                # Continue playing at appropriate frame rate
                # Using a simple approach here for simplicity
                self.root.after(30, self.play_video)
            else:
                # End of video reached
                self.playing = False
                self.play_btn.config(text="Play")
                
                # Reset to the beginning of the video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def detect_and_display(self, frame):
        """Detect people in frame and display the result"""
        if frame is None:
            return
            
        # Convert from BGR to RGB for YOLOv5
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        results = self.model(rgb_frame)
        
        # Get detections dataframe
        df = results.pandas().xyxy[0]
        
        # Filter for person class (class 0)
        people_df = df[df['class'] == 0]
        
        # Update people count
        self.people_count = len(people_df)
        self.count_label.config(text=f"People count: {self.people_count}")
        
        # Draw bounding boxes
        result_frame = frame.copy()
        for _, row in people_df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            conf = row['confidence']
            
            # Draw rectangle
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with confidence
            label = f"Person: {conf:.2f}"
            cv2.putText(result_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert to format suitable for tkinter
        img = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        # Resize if needed to fit the window
        window_width = self.video_frame.winfo_width()
        window_height = self.video_frame.winfo_height()
        
        if window_width > 1 and window_height > 1:  # Ensure valid dimensions
            img = self.resize_image(img, window_width, window_height)
        
        # Convert to PhotoImage
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update the display
        self.video_frame.config(image=img_tk)
        self.video_frame.image = img_tk  # Keep a reference to prevent garbage collection

    def resize_image(self, img, target_width, target_height):
        """Resize image while maintaining aspect ratio"""
        width, height = img.size
        ratio = min(target_width/width, target_height/height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return img.resize((new_width, new_height), Image.LANCZOS)

    def on_closing(self):
        """Handle window closing"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = PeopleCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
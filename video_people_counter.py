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
        
        # Set fixed window size
        self.window_width = 1200
        self.window_height = 800
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        
        # Prevent automatic window resizing
        self.root.minsize(self.window_width, self.window_height)
        self.root.maxsize(self.window_width, self.window_height)
        
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Set model to detect only people (class 0) and cars (class 2) in COCO dataset
        self.model.classes = [0, 2]  # 0 is people, 2 is cars
        
        # Configure layout
        self.setup_layout()
        
        # Video related variables
        self.cap = None
        self.playing = False
        self.frame_count = 0
        self.current_frame = None
        self.people_count = 0
        self.car_count = 0
        self.slider_updating = False  # Flag to prevent recursive slider updates
    
    def setup_layout(self):
        """Setup the UI layout using place for absolute positioning"""
        # Create a main frame to contain everything
        self.main_frame = tk.Frame(self.root, width=self.window_width, height=self.window_height)
        self.main_frame.pack_propagate(False)  # Prevent propagation of size changes
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Button frame at top - fixed height
        btn_frame_height = 50
        self.btn_frame = tk.Frame(self.main_frame, height=btn_frame_height)
        self.btn_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Upload button
        self.upload_btn = tk.Button(self.btn_frame, text="Upload Video", command=self.upload_video)
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Play/Pause button
        self.play_btn = tk.Button(self.btn_frame, text="Play", command=self.toggle_play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        # Count display frame
        count_frame = tk.Frame(self.btn_frame)
        count_frame.pack(side=tk.RIGHT, padx=20)
        
        # People count display
        self.people_count_label = tk.Label(count_frame, text="People: 0", font=("Arial", 14))
        self.people_count_label.pack(side=tk.TOP, anchor=tk.E)
        
        # Car count display
        self.car_count_label = tk.Label(count_frame, text="Cars: 0", font=("Arial", 14))
        self.car_count_label.pack(side=tk.TOP, anchor=tk.E)
        
        # Bottom area for slider and status - fixed height
        bottom_area_height = 80
        self.bottom_area = tk.Frame(self.main_frame, height=bottom_area_height)
        self.bottom_area.pack(side=tk.BOTTOM, fill=tk.X)
        self.bottom_area.pack_propagate(False)  # Prevent propagation of size changes
        
        # Slider frame
        self.slider_frame = tk.Frame(self.bottom_area)
        self.slider_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Current frame position label
        self.frame_pos_label = tk.Label(self.slider_frame, text="Frame: 0 / 0", width=15, anchor=tk.W)
        self.frame_pos_label.pack(side=tk.LEFT, padx=5)
        
        # Timeline slider
        self.timeline_slider = tk.Scale(
            self.slider_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            showvalue=0,
            command=self.on_slider_change,
            state=tk.DISABLED
        )
        self.timeline_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Status bar
        self.status_bar = tk.Label(self.bottom_area, text="Ready. Upload a video to begin.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Video frame area - fills the rest of the space
        video_frame_height = self.window_height - btn_frame_height - bottom_area_height
        self.video_container = tk.Frame(self.main_frame, width=self.window_width, height=video_frame_height, bg="black")
        self.video_container.pack_propagate(False)  # Prevent propagation of size changes
        self.video_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Actual video display label
        self.video_frame = tk.Label(self.video_container, bg="black")
        self.video_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Center in container

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
            
            # Update slider range
            self.timeline_slider.config(
                from_=0,
                to=self.frame_count - 1,
                state=tk.NORMAL
            )
            
            # Update status
            video_name = file_path.split("/")[-1]
            self.status_bar.config(text=f"Loaded: {video_name} ({width}x{height}, {fps:.2f} fps, {self.frame_count} frames)")
            
            # Enable play button
            self.play_btn.config(state=tk.NORMAL)
            
            # Display the first frame
            ret, self.current_frame = self.cap.read()
            if ret:
                self.detect_and_display(self.current_frame)
                self.update_frame_position(0)

    def toggle_play(self):
        """Toggle between play and pause states"""
        if self.playing:
            self.playing = False
            self.play_btn.config(text="Play")
        else:
            self.playing = True
            self.play_btn.config(text="Pause")
            self.play_video()

    def on_slider_change(self, value):
        """Handle slider position changes"""
        if self.slider_updating or not self.cap:
            return
            
        # Convert to integer
        frame_pos = int(float(value))
        
        # Seek to the selected frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        
        # Read and display the frame
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.detect_and_display(frame)
            self.update_frame_position(frame_pos)

    def update_frame_position(self, frame_pos):
        """Update frame position label and slider"""
        # Update frame position label
        self.frame_pos_label.config(text=f"Frame: {frame_pos} / {self.frame_count-1}")
        
        # Update slider position without triggering on_slider_change
        self.slider_updating = True
        self.timeline_slider.set(frame_pos)
        self.slider_updating = False

    def play_video(self):
        """Process and display video frames"""
        if self.cap and self.playing:
            # Get current position
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Check if we're at the end
            if current_pos >= self.frame_count - 1:
                # Reset to the beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_pos = 0
            
            # Read the next frame
            ret, frame = self.cap.read()
            
            if ret:
                self.current_frame = frame
                self.detect_and_display(frame)
                
                # Update the frame position
                self.update_frame_position(current_pos)
                
                # Continue playing at appropriate frame rate
                # Using a simple approach here for simplicity
                self.root.after(30, self.play_video)
            else:
                # End of video reached or error
                self.playing = False
                self.play_btn.config(text="Play")

    def detect_and_display(self, frame):
        """Detect people and cars in frame and display the result"""
        if frame is None:
            return
            
        # Convert from BGR to RGB for YOLOv5
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        results = self.model(rgb_frame)
        
        # Get detections dataframe
        df = results.pandas().xyxy[0]
        
        # Filter for person class (class 0) and car class (class 2)
        people_df = df[df['class'] == 0]
        car_df = df[df['class'] == 2]
        
        # Update counts
        self.people_count = len(people_df)
        self.car_count = len(car_df)
        self.people_count_label.config(text=f"People: {self.people_count}")
        self.car_count_label.config(text=f"Cars: {self.car_count}")
        
        # Draw bounding boxes
        result_frame = frame.copy()
        
        # Draw people bounding boxes (green)
        for _, row in people_df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            conf = row['confidence']
            
            # Draw rectangle
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with confidence
            label = f"Person: {conf:.2f}"
            cv2.putText(result_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Draw car bounding boxes (blue)
        for _, row in car_df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            conf = row['confidence']
            
            # Draw rectangle
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Add label with confidence
            label = f"Car: {conf:.2f}"
            cv2.putText(result_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Convert to format suitable for tkinter
        img = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        # Get container dimensions
        container_width = self.video_container.winfo_width()
        container_height = self.video_container.winfo_height()
        
        # Resize image to fit container while maintaining aspect ratio
        if container_width > 1 and container_height > 1:
            img = self.resize_image(img, container_width, container_height)
        
        # Convert to PhotoImage
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update the display and configure dimensions
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
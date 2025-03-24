import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from PIL import Image, ImageTk
import math

class AutoTableDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Restaurant Table Usage Detector")
        
        # Set fixed window size
        self.window_width = 1200
        self.window_height = 800
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        
        # Prevent automatic window resizing
        self.root.minsize(self.window_width, self.window_height)
        self.root.maxsize(self.window_width, self.window_height)
        
        # Load YOLOv5 model for people detection
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Set model to detect people (class 0) and dining table (class 60) in COCO dataset
        self.model.classes = [0, 60]  # 0 is person, 60 is dining table
        
        # Configure layout
        self.setup_layout()
        
        # Video related variables
        self.cap = None
        self.playing = False
        self.frame_count = 0
        self.current_frame = None
        self.people_count = 0
        self.slider_updating = False  # Flag to prevent recursive slider updates
        
        # Table related variables
        self.tables = []  # Will store table positions as [x, y, width, height, id]
        self.total_tables = 0
        self.tables_in_use = 0
        self.tables_empty = 0
        self.proximity_threshold = 100  # Distance in pixels to consider a person at a table
        
        # Table detection parameters
        self.tables_detected = False
        self.detection_confidence = 0.6  # Increased from 0.25 to be more selective
        self.table_overlap_threshold = 0.4  # Increased from 0.2 to require more separation between tables
    
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
        
        # Detect Tables button
        self.detect_tables_btn = tk.Button(self.btn_frame, text="Detect Tables", command=self.detect_tables, state=tk.DISABLED)
        self.detect_tables_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear Tables button
        self.clear_tables_btn = tk.Button(self.btn_frame, text="Clear Tables", command=self.clear_tables, state=tk.DISABLED)
        self.clear_tables_btn.pack(side=tk.LEFT, padx=5)
        
        # People count display
        self.count_label = tk.Label(self.btn_frame, text="People count: 0", font=("Arial", 14))
        self.count_label.pack(side=tk.RIGHT, padx=20)
        
        # Table count display
        self.table_count_label = tk.Label(self.btn_frame, text="Tables: 0 total | 0 in use | 0 empty", font=("Arial", 14))
        self.table_count_label.pack(side=tk.RIGHT, padx=20)
        
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
            
            # Enable buttons
            self.play_btn.config(state=tk.NORMAL)
            self.detect_tables_btn.config(state=tk.NORMAL)
            self.clear_tables_btn.config(state=tk.NORMAL)
            
            # Reset tables
            self.tables = []
            self.tables_detected = False
            
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

    def detect_tables(self):
        """Detect tables in the current frame using multiple approaches"""
        if self.current_frame is None:
            return
            
        # Pause video if playing
        was_playing = self.playing
        if self.playing:
            self.toggle_play()
            
        self.status_bar.config(text="Detecting tables automatically... Please wait.")
        self.root.update()
        
        # Clear any existing tables
        self.tables = []
        
        # Method 1: Use YOLO to detect dining tables
        detected_tables = self.detect_tables_yolo(self.current_frame)
        
        # Method 2: If YOLO didn't find enough tables, use computer vision techniques
        if len(detected_tables) < 3:
            tables_cv = self.detect_tables_cv(self.current_frame)
            # Add only non-overlapping tables
            for table_cv in tables_cv:
                should_add = True
                for existing_table in detected_tables:
                    # Check for overlap (using IoU)
                    if self.calculate_iou(table_cv, existing_table) > self.table_overlap_threshold:
                        should_add = False
                        break
                if should_add:
                    detected_tables.append(table_cv)
        
        # Add all detected tables
        for i, (x, y, width, height) in enumerate(detected_tables):
            table_id = i + 1
            self.tables.append([x, y, width, height, table_id])
        
        self.tables_detected = True
        self.total_tables = len(self.tables)
        
        # Update status
        if self.total_tables > 0:
            self.status_bar.config(text=f"Successfully detected {self.total_tables} tables")
        else:
            self.status_bar.config(text="No tables could be detected automatically")
            messagebox.showinfo("No Tables Detected", "No tables could be detected automatically. The video may not contain visible tables, or they might be hard to detect.")
        
        # Update display
        self.detect_and_display(self.current_frame)
        
        # Resume playing if it was playing before
        if was_playing:
            self.toggle_play()

    def detect_tables_yolo(self, frame):
        """Detect tables using YOLOv5"""
        # Convert from BGR to RGB for YOLOv5
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        results = self.model(rgb_frame)
        
        # Get detections dataframe
        df = results.pandas().xyxy[0]
        
        # Filter for tables (class 60 is dining table in COCO)
        tables_df = df[(df['class'] == 60) & (df['confidence'] > self.detection_confidence)]
        
        # Extract table positions
        tables = []
        for _, row in tables_df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            width = x2 - x1
            height = y2 - y1
            tables.append([x1, y1, width, height])
        
        return tables

    def detect_tables_cv(self, frame):
        """Detect potential tables using computer vision techniques"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Apply Canny edge detector
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect broken lines
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        min_area = 2000  # Increased from 500 to require larger tables
        max_area = frame.shape[0] * frame.shape[1] * 0.4  # Reduced from 70% to 40% of frame
        
        for contour in contours:
            # Approximate the contour to simplify shape
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            
            # Filter by size and shape (with stricter aspect ratio)
            if (min_area < area < max_area and 
                0.7 < (w / h) < 1.5):  # Narrower aspect ratio constraint
                tables.append([x, y, w, h])
        
        return tables

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x, y, width, height]"""
        # Convert to [x1, y1, x2, y2] format
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Calculate intersection area
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area

    def detect_and_display(self, frame):
        """Detect people in frame, check table usage, and display the result"""
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
        
        # Create a working copy of the frame
        result_frame = frame.copy()
        
        # Extract people positions (center points of bounding boxes)
        people_positions = []
        for _, row in people_df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            conf = row['confidence']
            
            # Calculate center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            people_positions.append((center_x, center_y))
            
            # Draw rectangle around person
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with confidence
            label = f"Person: {conf:.2f}"
            cv2.putText(result_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Check table usage and draw tables
        self.tables_in_use = 0
        self.tables_empty = 0
        
        for table in self.tables:
            x, y, width, height, table_id = table
            table_center_x = x + width // 2
            table_center_y = y + height // 2
            
            # Check if any person is near this table
            table_in_use = False
            for person_x, person_y in people_positions:
                # Calculate distance between person and table center
                distance = math.sqrt((person_x - table_center_x)**2 + (person_y - table_center_y)**2)
                
                # Check if the person's center is within the table boundaries (plus some margin)
                margin = self.proximity_threshold
                if (x - margin <= person_x <= x + width + margin and 
                    y - margin <= person_y <= y + height + margin):
                    table_in_use = True
                    break
            
            # Update counter and draw table with appropriate color
            if table_in_use:
                self.tables_in_use += 1
                color = (0, 0, 255)  # Red for tables in use
                status = "In Use"
            else:
                self.tables_empty += 1
                color = (255, 0, 0)  # Blue for empty tables
                status = "Empty"
                
            # Draw table rectangle
            cv2.rectangle(result_frame, (x, y), (x + width, y + height), color, 2)
            
            # Add table label with ID and status
            table_label = f"Table {table_id}: {status}"
            cv2.putText(result_frame, table_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Update table counts label
        self.total_tables = len(self.tables)
        self.table_count_label.config(
            text=f"Tables: {self.total_tables} total | {self.tables_in_use} in use | {self.tables_empty} empty"
        )
        
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

    def clear_tables(self):
        """Clear all tables"""
        self.tables = []
        self.total_tables = 0
        self.tables_in_use = 0
        self.tables_empty = 0
        self.tables_detected = False
        self.table_count_label.config(text="Tables: 0 total | 0 in use | 0 empty")
        self.status_bar.config(text="All tables cleared")
        
        # Update the display
        if self.current_frame is not None:
            self.detect_and_display(self.current_frame)

    def on_closing(self):
        """Handle window closing"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = AutoTableDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
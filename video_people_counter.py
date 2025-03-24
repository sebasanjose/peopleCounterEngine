import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import torch
from PIL import Image, ImageTk
import pandas as pd

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
        
        # Vehicle tracking system
        self.vehicle_tracking = {}  # Will store {track_id: {'type': 'truck', 'last_seen': frame_num, 'bbox': (x1,y1,x2,y2)}}
        self.track_id_counter = 0
        self.tracking_threshold = 0.5  # IoU threshold for considering it the same vehicle
        
        # Check for available device
        self.device = 'cpu'
        try:
            if torch.backends.mps.is_available():
                self.device = 'mps'
                print("Using MPS (Apple Silicon GPU) for acceleration")
            elif torch.cuda.is_available():
                self.device = 'cuda'
                print("Using CUDA GPU for acceleration")
            else:
                print("Using CPU for computation (no GPU acceleration available)")
                
            # Load YOLOv5 model
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            
            # Set model to detect people and various vehicles in COCO dataset
            # 0: person, 2: car, 3: motorcycle, 7: truck
            self.model.classes = [0, 2, 3, 7]
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to CPU
            self.device = 'cpu'
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device='cpu')
            self.model.classes = [0, 2, 3, 7]
        
        # Configure layout
        self.setup_layout()
        
        # Video related variables
        self.cap = None
        self.playing = False
        self.frame_count = 0
        self.current_frame = None
        self.people_count = 0
        self.car_count = 0
        self.motorcycle_count = 0
        self.truck_count = 0
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
        
        # Motorcycle count display
        self.motorcycle_count_label = tk.Label(count_frame, text="Motorcycles: 0", font=("Arial", 14))
        self.motorcycle_count_label.pack(side=tk.TOP, anchor=tk.E)
        
        # Truck count display
        self.truck_count_label = tk.Label(count_frame, text="Trucks: 0", font=("Arial", 14))
        self.truck_count_label.pack(side=tk.TOP, anchor=tk.E)
        
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
            # Reset tracking when loading a new video
            self.vehicle_tracking = {}
            self.track_id_counter = 0
            
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
        """Detect people and vehicles in frame and display the result"""
        if frame is None:
            return
            
        # Convert from BGR to RGB for YOLOv5
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection - let YOLOv5 handle the tensor conversion
        with torch.no_grad():
            results = self.model(rgb_frame)
        
        # Get detections dataframe
        df = results.pandas().xyxy[0]
        
        # Filter for different classes
        people_df = df[df['class'] == 0]  # person
        car_df = df[df['class'] == 2]     # car
        motorcycle_df = df[df['class'] == 3]  # motorcycle
        truck_df = df[df['class'] == 7]   # truck
        
        # Current frame number
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Combine all vehicle detections for tracking
        all_vehicles_df = pd.concat([car_df, truck_df])
        vehicles_to_remove_from_car = []
        vehicles_to_add_to_truck = []
        
        # First pass: Check if any detection matches previously tracked trucks
        for idx, vehicle in all_vehicles_df.iterrows():
            vehicle_bbox = (vehicle['xmin'], vehicle['ymin'], vehicle['xmax'], vehicle['ymax'])
            
            # Check against tracked trucks
            for track_id, track_info in self.vehicle_tracking.items():
                if track_info['type'] == 'truck' and self.calculate_iou(vehicle_bbox, track_info['bbox']) > self.tracking_threshold:
                    # This is likely the same truck we saw before
                    if vehicle['class'] == 2:  # It's currently classified as a car
                        # Reclassify as truck with higher confidence
                        vehicles_to_remove_from_car.append(idx)
                        
                        # Boost the confidence but cap at 0.95
                        boosted_conf = min(0.95, vehicle['confidence'] * 1.5)
                        
                        # Create a new truck entry
                        truck_entry = vehicle.copy()
                        truck_entry['class'] = 7  # Change to truck class
                        truck_entry['confidence'] = boosted_conf
                        truck_entry['name'] = 'truck'  # Update the name
                        vehicles_to_add_to_truck.append(truck_entry)
                    
                    # Update tracking
                    self.vehicle_tracking[track_id]['last_seen'] = current_frame
                    self.vehicle_tracking[track_id]['bbox'] = vehicle_bbox
                    break
        
        # Apply the changes to car and truck dataframes
        if vehicles_to_remove_from_car:
            car_df = car_df.drop(vehicles_to_remove_from_car)
        
        if vehicles_to_add_to_truck:
            truck_add_df = pd.DataFrame(vehicles_to_add_to_truck)
            truck_df = pd.concat([truck_df, truck_add_df], ignore_index=True)
        
        # Update tracking with new detections
        for _, truck in truck_df.iterrows():
            truck_bbox = (truck['xmin'], truck['ymin'], truck['xmax'], truck['ymax'])
            tracked = False
            
            # Check if this truck is already being tracked
            for track_id, track_info in self.vehicle_tracking.items():
                if self.calculate_iou(truck_bbox, track_info['bbox']) > self.tracking_threshold:
                    # Update existing track
                    self.vehicle_tracking[track_id]['last_seen'] = current_frame
                    self.vehicle_tracking[track_id]['bbox'] = truck_bbox
                    tracked = True
                    break
            
            # If not tracked, create new track
            if not tracked:
                self.track_id_counter += 1
                self.vehicle_tracking[self.track_id_counter] = {
                    'type': 'truck',
                    'last_seen': current_frame,
                    'bbox': truck_bbox,
                    'confidence_history': [truck['confidence']]
                }
        
        # Clean up old tracks (not seen for more than 30 frames)
        self.vehicle_tracking = {k: v for k, v in self.vehicle_tracking.items() 
                                if current_frame - v['last_seen'] < 30}
        
        # Update counts
        self.people_count = len(people_df)
        self.car_count = len(car_df)
        self.motorcycle_count = len(motorcycle_df)
        self.truck_count = len(truck_df)
        
        # Update count labels
        self.people_count_label.config(text=f"People: {self.people_count}")
        self.car_count_label.config(text=f"Cars: {self.car_count}")
        self.motorcycle_count_label.config(text=f"Motorcycles: {self.motorcycle_count}")
        self.truck_count_label.config(text=f"Trucks: {self.truck_count}")
        
        # Draw bounding boxes
        result_frame = frame.copy()
        
        # Define colors for different object types (BGR format for OpenCV)
        colors = {
            'person': (0, 255, 0),      # Green
            'car': (255, 0, 0),         # Blue
            'motorcycle': (255, 0, 255), # Magenta
            'truck': (0, 255, 255)       # Yellow
        }
        
        # Draw boxes for each object type
        for _, row in people_df.iterrows():
            self.draw_box(result_frame, row, "Person", colors['person'])
            
        for _, row in car_df.iterrows():
            self.draw_box(result_frame, row, "Car", colors['car'])
            
        for _, row in motorcycle_df.iterrows():
            self.draw_box(result_frame, row, "Motorcycle", colors['motorcycle'])
            
        for _, row in truck_df.iterrows():
            self.draw_box(result_frame, row, "Truck", colors['truck'])
        
        # Convert to format suitable for tkinter
        img = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        # Resize if needed to fit the window
        container_width = self.video_container.winfo_width()
        container_height = self.video_container.winfo_height()
        
        if container_width > 1 and container_height > 1:  # Ensure valid dimensions
            img = self.resize_image(img, container_width, container_height)
        
        # Convert to PhotoImage
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update the display and configure dimensions
        self.video_frame.config(image=img_tk)
        self.video_frame.image = img_tk  # Keep a reference to prevent garbage collection
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        # Convert bboxes to (x, y, w, h) format
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        # Calculate area of intersection
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both bboxes
        bbox1_area = (x2 - x1) * (y2 - y1)
        bbox2_area = (x4 - x3) * (y4 - y3)
        
        # Calculate IoU
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        return iou
    
    def draw_box(self, frame, detection, label_prefix, color):
        """Draw a single bounding box with label"""
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add label with confidence
        label = f"{label_prefix}: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
import cv2
import numpy as np
import os
import math
import json
import onnxruntime as ort

class OpticalFlowProcessor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Use onnxruntime session instead of OpenCV DNN to support quantized ONNX models
        providers = None
        try:
            # Prefer CUDAExecutionProvider if available
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception:
            # Fallback to default CPU provider
            self.session = ort.InferenceSession(self.model_path)
        
        self.input_width = 480
        self.input_height = 360
        
        # Drawing parameters
        self.draw_step = 30
        self.min_motion_magnitude = 0.40
        self.vector_color = (255, 255, 40) # BGR for (40, 255, 255)
        self.dot_radius = 4
        self.vector_thickness = 4
        self.vector_length_multiplier = 3.6
        self.min_display_vector_length = 9.0
        
        # Heatmap parameters
        self.heatmap_frame_weight = 0.58
        self.heatmap_color_weight = 0.70
        self.heatmap_normalize_multiplier = 9.0
        self.heatmap_input_threshold_multiplier = 0.40
        self.heatmap_mask_threshold_multiplier = 0.32

    def prepare_blob(self, img):
        # Convert to RGB (swapRB=True in OpenCV is equivalent to BGR2RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize and convert to float32 NCHW format: (1, C, H, W)
        resized = cv2.resize(img_rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32)
        # Normalize to [0,1]
        arr /= 255.0
        # HWC -> CHW
        chw = np.transpose(arr, (2, 0, 1))
        blob = np.expand_dims(chw, axis=0)
        return blob

    def infer(self, prev_frame, curr_frame):
        prev_blob = self.prepare_blob(prev_frame)
        curr_blob = self.prepare_blob(curr_frame)

        # Build inputs for the ONNX model based on expected inputs
        input_meta = self.session.get_inputs()
        feed = {}

        try:
            if len(input_meta) == 2:
                # Model expects two inputs (e.g., named "0" and "1")
                feed[input_meta[0].name] = prev_blob
                feed[input_meta[1].name] = curr_blob
            elif len(input_meta) == 1:
                # Single input: concatenate along channel dimension -> (1,6,H,W)
                single_shape = input_meta[0].shape
                # Determine if model expects NHWC or NCHW by checking input shape layout
                if len(single_shape) == 4 and (single_shape[1] == 3 or single_shape[1] == 6):
                    # NCHW expected
                    concatenated = np.concatenate([prev_blob, curr_blob], axis=1)
                    feed[input_meta[0].name] = concatenated
                elif len(single_shape) == 4 and (single_shape[3] == 3 or single_shape[3] == 6):
                    # NHWC expected, convert blobs to NHWC
                    prev_nhwc = np.transpose(prev_blob, (0, 2, 3, 1))
                    curr_nhwc = np.transpose(curr_blob, (0, 2, 3, 1))
                    concatenated = np.concatenate([prev_nhwc, curr_nhwc], axis=3)
                    feed[input_meta[0].name] = concatenated
                else:
                    # Fallback: try channel concat
                    concatenated = np.concatenate([prev_blob, curr_blob], axis=1)
                    feed[input_meta[0].name] = concatenated
            else:
                # Generic: map first two inputs if available
                for i, meta in enumerate(input_meta[:2]):
                    feed[meta.name] = prev_blob if i == 0 else curr_blob

            outputs = self.session.run(None, feed)
            # Return first output
            return outputs[0]
        except Exception as e:
            raise RuntimeError(f"ONNX inference failed: {e}")

    def compute_centered_grid_start(self, size, step):
        if size <= step:
            return size // 2
        half_step = step // 2
        sample_count = (((size - 1) - half_step) // step) + 1
        occupied_span = (sample_count - 1) * step
        return round((size - 1 - occupied_span) / 2.0)

    def draw_heatmap(self, flow, frame):
        # flow shape: typically (1, 2, H, W) for NCHW or (1, H, W, 2) for NHWC
        if len(flow.shape) == 4 and flow.shape[1] == 2:
            u = flow[0, 0, :, :]
            v = flow[0, 1, :, :]
        elif len(flow.shape) == 4 and flow.shape[3] == 2:
            u = flow[0, :, :, 0]
            v = flow[0, :, :, 1]
        elif len(flow.shape) == 3 and flow.shape[0] == 2:
            u = flow[0, :, :]
            v = flow[1, :, :]
        elif len(flow.shape) == 3 and flow.shape[2] == 2:
            u = flow[:, :, 0]
            v = flow[:, :, 1]
        else:
            return frame

        flow_h, flow_w = u.shape
        frame_h, frame_w = frame.shape[:2]
        
        x_scale = frame_w / flow_w
        y_scale = frame_h / flow_h

        # Calculate magnitude
        fx = u * x_scale
        fy = v * y_scale
        magnitude = np.sqrt(fx**2 + fy**2)
        
        # Gaussian blur magnitude
        magnitude = cv2.GaussianBlur(magnitude, (9, 9), 0.0)
        
        max_magnitude = np.max(magnitude)
        if max_magnitude <= self.min_motion_magnitude * self.heatmap_input_threshold_multiplier:
            return frame
            
        normalize_max = max(max_magnitude, self.min_motion_magnitude * self.heatmap_normalize_multiplier)
        
        # Normalize to 0-255
        normalized = (magnitude / normalize_max) * 255.0
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        heatmap8u = cv2.GaussianBlur(normalized, (15, 15), 0.0)
        heatmap_bgr = cv2.applyColorMap(heatmap8u, cv2.COLORMAP_TURBO)
        heatmap_scaled_bgr = cv2.resize(heatmap_bgr, (frame_w, frame_h), interpolation=cv2.INTER_CUBIC)
        
        # Mask
        _, mask_small = cv2.threshold(magnitude, self.min_motion_magnitude * self.heatmap_mask_threshold_multiplier, 255.0, cv2.THRESH_BINARY)
        mask_small = mask_small.astype(np.uint8)
        mask = cv2.resize(mask_small, (frame_w, frame_h), interpolation=cv2.INTER_CUBIC)
        mask = cv2.GaussianBlur(mask, (31, 31), 0.0)
        _, mask = cv2.threshold(mask, 1.0, 255.0, cv2.THRESH_BINARY)
        
        # Blend
        blended = cv2.addWeighted(frame, self.heatmap_frame_weight, heatmap_scaled_bgr, self.heatmap_color_weight, 0.0)
        
        # Apply mask
        result_frame = np.copy(frame)
        mask_bool = mask > 0
        result_frame[mask_bool] = blended[mask_bool]
        
        return result_frame

    def draw_vectors(self, flow, frame, vector_direction_sign=-1.0):
        # Determine layout and extract u, v
        if len(flow.shape) == 4 and flow.shape[1] == 2:
            u = flow[0, 0, :, :]
            v = flow[0, 1, :, :]
        elif len(flow.shape) == 4 and flow.shape[3] == 2:
            u = flow[0, :, :, 0]
            v = flow[0, :, :, 1]
        elif len(flow.shape) == 3 and flow.shape[0] == 2:
            u = flow[0, :, :]
            v = flow[1, :, :]
        elif len(flow.shape) == 3 and flow.shape[2] == 2:
            u = flow[:, :, 0]
            v = flow[:, :, 1]
        else:
            return frame

        flow_h, flow_w = u.shape
        frame_h, frame_w = frame.shape[:2]
        
        x_scale = frame_w / flow_w
        y_scale = frame_h / flow_h
        
        start_x = self.compute_centered_grid_start(frame_w, self.draw_step)
        start_y = self.compute_centered_grid_start(frame_h, self.draw_step)
        
        min_motion_squared = self.min_motion_magnitude ** 2
        
        result_frame = np.copy(frame)
        
        screen_y = start_y
        while screen_y < frame_h:
            screen_x = start_x
            while screen_x < frame_w:
                flow_x = min(max(round(screen_x / x_scale), 0), flow_w - 1)
                flow_y = min(max(round(screen_y / y_scale), 0), flow_h - 1)
                
                fx = u[flow_y, flow_x] * x_scale
                fy = v[flow_y, flow_x] * y_scale
                
                magnitude_squared = fx**2 + fy**2
                
                if magnitude_squared >= min_motion_squared:
                    start_pt = (screen_x, screen_y)
                    display_fx = fx * vector_direction_sign * self.vector_length_multiplier
                    display_fy = fy * vector_direction_sign * self.vector_length_multiplier
                    
                    display_magnitude = math.sqrt(display_fx**2 + display_fy**2)
                    if 0.0 < display_magnitude < self.min_display_vector_length:
                        scale_up = self.min_display_vector_length / display_magnitude
                        display_fx *= scale_up
                        display_fy *= scale_up
                        
                    end_pt = (int(round(start_pt[0] + display_fx)), int(round(start_pt[1] + display_fy)))
                    
                    cv2.line(result_frame, start_pt, end_pt, self.vector_color, self.vector_thickness)
                    cv2.circle(result_frame, start_pt, self.dot_radius, self.vector_color, -1)
                
                screen_x += self.draw_step
            screen_y += self.draw_step
            
        return result_frame

    def process_video(
        self,
        input_video_path,
        output_video_path,
        mode="VECTORS",
        vector_direction_sign=-1.0,
        req_id: str = None,
        progress_callback=None,
    ):
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {input_video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30.0
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            out.release()
            return
        # Setup status tracking
        status_path = None
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception:
            total_frames = 0
        def report_progress(percent):
            percent = max(0, min(100, int(percent)))
            if progress_callback is not None:
                try:
                    progress_callback(percent)
                except Exception:
                    pass
            if status_path is not None:
                try:
                    with open(status_path, 'w') as f:
                        json.dump({"percent": percent}, f)
                except Exception:
                    pass

        if req_id is not None:
            status_path = os.path.join('temp_videos', f"{req_id}_status.json")
            try:
                os.makedirs(os.path.dirname(status_path), exist_ok=True)
            except Exception:
                status_path = None
        report_progress(0)
            
        completed = False
        try:
            # Write first frame
            out.write(prev_frame)
            
            frames_processed = 1
            # report initial progress (first frame written)
            if total_frames > 0:
                report_progress((frames_processed / total_frames) * 100)

            while True:
                ret, curr_frame = cap.read()
                if not ret:
                    break
                    
                flow_output = self.infer(prev_frame, curr_frame)
                
                if mode == "HEATMAP":
                    result_frame = self.draw_heatmap(flow_output, curr_frame)
                else:
                    result_frame = self.draw_vectors(flow_output, curr_frame, vector_direction_sign)
                    
                out.write(result_frame)
                prev_frame = curr_frame
                frames_processed += 1
                # update status every 5 frames
                if total_frames > 0 and frames_processed % 5 == 0:
                    report_progress((frames_processed / total_frames) * 100)
            completed = True
        finally:
            cap.release()
            out.release()
            if completed:
                report_progress(100)

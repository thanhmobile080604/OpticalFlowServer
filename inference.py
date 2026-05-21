import cv2
import numpy as np
import os
import math
import json
import logging
import onnxruntime as ort

logger = logging.getLogger("optical_flow.inference")

class OpticalFlowProcessor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Use onnxruntime session instead of OpenCV DNN to support quantized ONNX models
        providers = None
        try:
            # Prefer CUDAExecutionProvider if available
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception as e:
            logger.warning(
                "Preferred ONNX providers failed, falling back to default providers model=%s providers=%s error=%s",
                self.model_path,
                providers,
                e,
            )
            # Fallback to default CPU provider
            self.session = ort.InferenceSession(self.model_path)
        logger.info(
            "ONNX session ready model=%s providers=%s inputs=%s outputs=%s",
            self.model_path,
            self.session.get_providers(),
            [(item.name, item.shape, item.type) for item in self.session.get_inputs()],
            [(item.name, item.shape, item.type) for item in self.session.get_outputs()],
        )
        
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

    def summarize_flow(self, flow):
        arr = np.asarray(flow)
        finite_mask = np.isfinite(arr)
        finite_count = int(finite_mask.sum())
        total_count = int(arr.size)
        summary = {
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
            "finite": f"{finite_count}/{total_count}",
        }
        if np.issubdtype(arr.dtype, np.floating):
            summary["nan_count"] = int(np.isnan(arr).sum())
            summary["posinf_count"] = int(np.isposinf(arr).sum())
            summary["neginf_count"] = int(np.isneginf(arr).sum())
        if finite_count > 0:
            finite_values = arr[finite_mask]
            summary["min"] = float(np.min(finite_values))
            summary["max"] = float(np.max(finite_values))
            summary["mean"] = float(np.mean(finite_values))
        return summary

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
            input_summary = [(meta.name, meta.shape, meta.type) for meta in input_meta]
            feed_summary = {name: tuple(value.shape) for name, value in feed.items()}
            raise RuntimeError(f"ONNX inference failed: {e}; inputs={input_summary}; feed_shapes={feed_summary}") from e

    def compute_centered_grid_start(self, size, step):
        if size <= step:
            return size // 2
        half_step = step // 2
        sample_count = (((size - 1) - half_step) // step) + 1
        occupied_span = (sample_count - 1) * step
        return round((size - 1 - occupied_span) / 2.0)

    def draw_heatmap(self, flow, frame, job_id=None, frame_index=None):
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
            logger.warning(
                "Unsupported flow shape for heatmap job_id=%s frame=%s flow_shape=%s",
                job_id,
                frame_index,
                getattr(flow, "shape", None),
            )
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

    def draw_vectors(self, flow, frame, vector_direction_sign=-1.0, job_id=None, frame_index=None):
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
            logger.warning(
                "Unsupported flow shape for vectors job_id=%s frame=%s flow_shape=%s",
                job_id,
                frame_index,
                getattr(flow, "shape", None),
            )
            return frame

        flow_h, flow_w = u.shape
        frame_h, frame_w = frame.shape[:2]
        
        x_scale = frame_w / flow_w
        y_scale = frame_h / flow_h
        
        start_x = self.compute_centered_grid_start(frame_w, self.draw_step)
        start_y = self.compute_centered_grid_start(frame_h, self.draw_step)
        
        min_motion_squared = self.min_motion_magnitude ** 2
        
        result_frame = np.copy(frame)
        invalid_vectors = 0
        
        screen_y = start_y
        while screen_y < frame_h:
            screen_x = start_x
            while screen_x < frame_w:
                flow_x = min(max(round(screen_x / x_scale), 0), flow_w - 1)
                flow_y = min(max(round(screen_y / y_scale), 0), flow_h - 1)
                
                fx = u[flow_y, flow_x] * x_scale
                fy = v[flow_y, flow_x] * y_scale
                if not (math.isfinite(float(fx)) and math.isfinite(float(fy))):
                    invalid_vectors += 1
                    screen_x += self.draw_step
                    continue
                
                magnitude_squared = fx**2 + fy**2
                
                if magnitude_squared >= min_motion_squared:
                    start_pt = (screen_x, screen_y)
                    display_fx = fx * vector_direction_sign * self.vector_length_multiplier
                    display_fy = fy * vector_direction_sign * self.vector_length_multiplier
                    if not (math.isfinite(float(display_fx)) and math.isfinite(float(display_fy))):
                        invalid_vectors += 1
                        screen_x += self.draw_step
                        continue
                    
                    display_magnitude = math.hypot(float(display_fx), float(display_fy))
                    if 0.0 < display_magnitude < self.min_display_vector_length:
                        scale_up = self.min_display_vector_length / display_magnitude
                        display_fx *= scale_up
                        display_fy *= scale_up
                        
                    end_x = float(start_pt[0] + display_fx)
                    end_y = float(start_pt[1] + display_fy)
                    max_endpoint_x = max(frame_w * 4.0, 1.0)
                    max_endpoint_y = max(frame_h * 4.0, 1.0)
                    if (
                        not (math.isfinite(end_x) and math.isfinite(end_y))
                        or abs(end_x) > max_endpoint_x
                        or abs(end_y) > max_endpoint_y
                    ):
                        invalid_vectors += 1
                        screen_x += self.draw_step
                        continue

                    end_pt = (int(round(end_x)), int(round(end_y)))
                    
                    cv2.line(result_frame, start_pt, end_pt, self.vector_color, self.vector_thickness)
                    cv2.circle(result_frame, start_pt, self.dot_radius, self.vector_color, -1)
                
                screen_x += self.draw_step
            screen_y += self.draw_step
        if invalid_vectors and (frame_index is None or frame_index <= 3 or frame_index % 30 == 0):
            logger.warning(
                "Skipped non-finite vectors job_id=%s frame=%s invalid_samples=%s flow_summary=%s",
                job_id,
                frame_index,
                invalid_vectors,
                self.summarize_flow(flow),
            )
            
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
        mode = (mode or "VECTORS").upper()
        if mode == "VECTOR":
            mode = "VECTORS"
        if mode not in ("VECTORS", "HEATMAP"):
            logger.warning("Unknown mode requested, defaulting to VECTORS job_id=%s requested_mode=%s", req_id, mode)
            mode = "VECTORS"

        logger.info(
            "Video processing initializing job_id=%s mode=%s input_path=%s output_path=%s vector_direction_sign=%.1f",
            req_id,
            mode,
            input_video_path,
            output_video_path,
            vector_direction_sign,
        )

        status_path = None
        if req_id is not None:
            status_path = os.path.join('temp_videos', f"{req_id}_status.json")
            try:
                os.makedirs(os.path.dirname(status_path), exist_ok=True)
            except Exception as e:
                logger.warning("Could not create status directory job_id=%s status_path=%s error=%s", req_id, status_path, e)
                status_path = None

        last_logged_progress = -1
        def report_progress(percent):
            nonlocal last_logged_progress
            percent = max(0, min(100, int(percent)))
            if progress_callback is not None:
                try:
                    progress_callback(percent)
                except Exception as e:
                    logger.warning("Progress callback failed job_id=%s percent=%s error=%s", req_id, percent, e)
            if status_path is not None:
                try:
                    with open(status_path, 'w') as f:
                        json.dump({"percent": percent}, f)
                except Exception as e:
                    logger.warning("Failed to write progress status job_id=%s status_path=%s percent=%s error=%s", req_id, status_path, percent, e)
            if percent == 0 or percent == 100 or percent >= last_logged_progress + 10:
                logger.info("Video progress job_id=%s mode=%s percent=%s", req_id, mode, percent)
                last_logged_progress = percent

        report_progress(0)

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {input_video_path}")

        out = None
        completed = False
        frames_processed = 0
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                logger.warning("Invalid FPS metadata, using fallback FPS job_id=%s raw_fps=%s fallback_fps=30.0", req_id, fps)
                fps = 30.0

            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            except Exception:
                total_frames = 0

            ret, prev_frame = cap.read()
            if not ret:
                raise Exception(f"Failed to read first frame from video: {input_video_path}")

            height, width = prev_frame.shape[:2]
            if width <= 0 or height <= 0:
                raise Exception(f"Invalid frame dimensions width={width} height={height}")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise Exception(f"Failed to open output video writer: {output_video_path} codec=mp4v fps={fps} size={width}x{height}")

            logger.info(
                "Video metadata job_id=%s mode=%s fps=%.3f width=%s height=%s total_frames=%s input_path=%s output_path=%s",
                req_id,
                mode,
                fps,
                width,
                height,
                total_frames,
                input_video_path,
                output_video_path,
            )

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

                frame_index = frames_processed + 1
                flow_output = None
                try:
                    flow_output = self.infer(prev_frame, curr_frame)
                    if frame_index == 2:
                        logger.info(
                            "First flow output job_id=%s mode=%s frame=%s flow_summary=%s",
                            req_id,
                            mode,
                            frame_index,
                            self.summarize_flow(flow_output),
                        )

                    if mode == "HEATMAP":
                        result_frame = self.draw_heatmap(flow_output, curr_frame, job_id=req_id, frame_index=frame_index)
                    else:
                        result_frame = self.draw_vectors(flow_output, curr_frame, vector_direction_sign, job_id=req_id, frame_index=frame_index)
                except Exception as e:
                    flow_summary = self.summarize_flow(flow_output) if flow_output is not None else None
                    logger.exception(
                        "Frame processing failed job_id=%s mode=%s frame=%s prev_frame_shape=%s curr_frame_shape=%s flow_summary=%s error=%s",
                        req_id,
                        mode,
                        frame_index,
                        getattr(prev_frame, "shape", None),
                        getattr(curr_frame, "shape", None),
                        flow_summary,
                        e,
                    )
                    raise RuntimeError(f"Frame {frame_index} processing failed in {mode} mode: {e}") from e
                    
                out.write(result_frame)
                prev_frame = curr_frame
                frames_processed += 1
                # update status every 5 frames
                if total_frames > 0 and frames_processed % 5 == 0:
                    report_progress((frames_processed / total_frames) * 100)
            completed = True
            logger.info(
                "Video processing finished job_id=%s mode=%s frames_processed=%s total_frames=%s",
                req_id,
                mode,
                frames_processed,
                total_frames,
            )
        finally:
            cap.release()
            if out is not None:
                out.release()
            if completed:
                report_progress(100)

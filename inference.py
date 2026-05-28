import cv2
import numpy as np
import os
import math
import json
import logging
import shutil
import subprocess
import onnxruntime as ort

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

logger = logging.getLogger("optical_flow.inference")


class ProcessingCancelled(Exception):
    pass


class H264Mp4Writer:
    MAX_OUTPUT_FPS = 30.0

    def __init__(self, output_path, source_fps, width, height, req_id=None):
        self.output_path = output_path
        self.source_fps = self._valid_fps(source_fps)
        self.output_fps = min(self.source_fps, self.MAX_OUTPUT_FPS)
        self.width = int(width)
        self.height = int(height)
        self.req_id = req_id
        self.process = None
        self.frames_written = 0

    def open(self):
        ffmpeg_exe = self._ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.source_fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-profile:v",
            "main",
            "-level",
            "4.2",
            "-tag:v",
            "avc1",
            "-movflags",
            "+faststart",
            "-r",
            f"{self.output_fps:.6f}",
            self.output_path,
        ]

        logger.info(
            "Opening H.264 writer job_id=%s output_path=%s source_fps=%.3f output_fps=%.3f size=%sx%s",
            self.req_id,
            self.output_path,
            self.source_fps,
            self.output_fps,
            self.width,
            self.height,
        )
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return self

    def _ffmpeg_exe(self):
        if imageio_ffmpeg is not None:
            return imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_exe = shutil.which("ffmpeg")
        if ffmpeg_exe:
            return ffmpeg_exe
        raise RuntimeError(
            "ffmpeg is required for Android-compatible H.264 output. "
            "Run pip install -r requirements.txt or install ffmpeg in PATH."
        )

    def write(self, frame):
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("H.264 writer is not open")
        if frame is None:
            return
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        try:
            self.process.stdin.write(frame.tobytes())
            self.frames_written += 1
        except BrokenPipeError as e:
            raise RuntimeError(f"H.264 writer stopped unexpectedly: {self._stderr_text()}") from e

    def release(self):
        if self.process is None:
            return
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
            return_code = self.process.wait(timeout=60)
            if return_code != 0:
                raise RuntimeError(
                    f"H.264 encoding failed return_code={return_code} stderr={self._stderr_text()}"
                )
            logger.info(
                "H.264 writer closed job_id=%s output_path=%s frames_written=%s output_fps=%.3f",
                self.req_id,
                self.output_path,
                self.frames_written,
                self.output_fps,
            )
        finally:
            self.process = None

    def cancel(self):
        if self.process is None:
            return
        try:
            if self.process.stdin is not None:
                try:
                    self.process.stdin.close()
                except Exception:
                    pass
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            logger.info(
                "H.264 writer cancelled job_id=%s output_path=%s frames_written=%s",
                self.req_id,
                self.output_path,
                self.frames_written,
            )
        finally:
            self.process = None

    def _stderr_text(self):
        if self.process is None or self.process.stderr is None:
            return ""
        if self.process.poll() is None:
            return ""
        try:
            return self.process.stderr.read().decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    @staticmethod
    def _valid_fps(raw_fps):
        try:
            fps = float(raw_fps)
        except (TypeError, ValueError):
            return 30.0
        if not math.isfinite(fps) or fps <= 0.0:
            return 30.0
        return min(max(fps, 1.0), 120.0)


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
        self.flow_output_index = self.select_flow_output_index(self.session.get_outputs())
        logger.info(
            "ONNX session ready model=%s providers=%s inputs=%s outputs=%s flow_output_index=%s",
            self.model_path,
            self.session.get_providers(),
            [(item.name, item.shape, item.type) for item in self.session.get_inputs()],
            [(item.name, item.shape, item.type) for item in self.session.get_outputs()],
            self.flow_output_index,
        )
        
        self.input_width = 480
        self.input_height = 360
        self.flow_frame_offset = 3
        
        # Drawing parameters
        self.draw_step = 34
        self.min_motion_magnitude = 0.45
        self.dot_radius = 2
        self.vector_length_multiplier = 2.4
        self.min_display_vector_length = 10.0
        self.max_display_vector_length = 56.0
        self.vector_activity_percentile = 58.0
        self.vector_peak_percentile = 95.0
        self.vector_shadow_alpha = 0.42
        
        # Heatmap parameters
        self.heatmap_peak_percentile = 98.5
        self.heatmap_floor_percentile = 45.0
        self.heatmap_gamma = 0.68
        self.heatmap_max_alpha = 0.78
        self.heatmap_background_weight = 0.72
        self.heatmap_min_alpha = 0.08
        self.turbo_lut = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8).reshape(256, 1),
            cv2.COLORMAP_TURBO,
        ).reshape(256, 3)

    def select_flow_output_index(self, output_meta):
        best_index = len(output_meta) - 1 if output_meta else 0
        best_pixels = -1
        for index, meta in enumerate(output_meta):
            shape = list(getattr(meta, "shape", []) or [])
            if len(shape) == 4 and shape[1] == 2:
                height, width = shape[2], shape[3]
            elif len(shape) == 4 and shape[3] == 2:
                height, width = shape[1], shape[2]
            elif len(shape) == 3 and shape[0] == 2:
                height, width = shape[1], shape[2]
            elif len(shape) == 3 and shape[2] == 2:
                height, width = shape[0], shape[1]
            else:
                continue

            try:
                pixels = int(height) * int(width)
            except (TypeError, ValueError):
                pixels = 0
            if pixels > best_pixels:
                best_index = index
                best_pixels = pixels
        return best_index

    def extract_flow_channels(self, flow, context, job_id=None, frame_index=None):
        arr = np.asarray(flow)
        if len(arr.shape) == 4 and arr.shape[1] == 2:
            u = arr[0, 0, :, :]
            v = arr[0, 1, :, :]
        elif len(arr.shape) == 4 and arr.shape[3] == 2:
            u = arr[0, :, :, 0]
            v = arr[0, :, :, 1]
        elif len(arr.shape) == 3 and arr.shape[0] == 2:
            u = arr[0, :, :]
            v = arr[1, :, :]
        elif len(arr.shape) == 3 and arr.shape[2] == 2:
            u = arr[:, :, 0]
            v = arr[:, :, 1]
        else:
            logger.warning(
                "Unsupported flow shape for %s job_id=%s frame=%s flow_shape=%s",
                context,
                job_id,
                frame_index,
                getattr(flow, "shape", None),
            )
            return None

        u = np.nan_to_num(u.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(v.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return u, v

    def postprocess_flow(self, outputs):
        flow = np.asarray(outputs[min(self.flow_output_index, len(outputs) - 1)])
        if len(flow.shape) == 4 and flow.shape[1] == 2:
            return flow[0].transpose(1, 2, 0)
        if len(flow.shape) == 4 and flow.shape[3] == 2:
            return flow[0]
        if len(flow.shape) == 3 and flow.shape[0] == 2:
            return flow.transpose(1, 2, 0)
        return flow

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
        resized = cv2.resize(img_rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32)
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
            if not outputs:
                raise RuntimeError("model returned no outputs")
            return self.postprocess_flow(outputs)
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
        channels = self.extract_flow_channels(flow, "heatmap", job_id=job_id, frame_index=frame_index)
        if channels is None:
            return frame
        u, v = channels

        flow_h, flow_w = u.shape
        frame_h, frame_w = frame.shape[:2]
        
        x_scale = frame_w / flow_w
        y_scale = frame_h / flow_h

        fx = u * x_scale
        fy = v * y_scale
        magnitude = np.sqrt(fx**2 + fy**2).astype(np.float32)
        magnitude = cv2.GaussianBlur(magnitude, (0, 0), 1.35)

        active = magnitude[magnitude > self.min_motion_magnitude]
        if active.size == 0:
            return frame

        motion_floor = max(
            self.min_motion_magnitude,
            float(np.percentile(active, self.heatmap_floor_percentile)) * 0.75,
        )
        motion_peak = float(np.percentile(active, self.heatmap_peak_percentile))
        motion_peak = max(motion_peak, motion_floor + 1e-3)

        normalized = np.clip((magnitude - motion_floor) / (motion_peak - motion_floor), 0.0, 1.0)
        normalized = np.power(normalized, self.heatmap_gamma)
        normalized = cv2.GaussianBlur(normalized.astype(np.float32), (0, 0), 1.4)

        heatmap8u = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap8u, cv2.COLORMAP_TURBO)
        heatmap_scaled_bgr = cv2.resize(heatmap_bgr, (frame_w, frame_h), interpolation=cv2.INTER_CUBIC)

        alpha_small = np.where(
            normalized > 0.01,
            self.heatmap_min_alpha + normalized * (self.heatmap_max_alpha - self.heatmap_min_alpha),
            0.0,
        ).astype(np.float32)
        alpha = cv2.resize(alpha_small, (frame_w, frame_h), interpolation=cv2.INTER_CUBIC)
        alpha = np.clip(cv2.GaussianBlur(alpha, (0, 0), 1.8), 0.0, self.heatmap_max_alpha)
        alpha3 = alpha[:, :, np.newaxis]

        base = cv2.addWeighted(
            frame,
            self.heatmap_background_weight,
            np.zeros_like(frame),
            1.0 - self.heatmap_background_weight,
            0.0,
        ).astype(np.float32)
        result_frame = base * (1.0 - alpha3) + heatmap_scaled_bgr.astype(np.float32) * alpha3
        result_frame = np.clip(result_frame, 0, 255).astype(np.uint8)

        return result_frame

    def draw_vectors(self, flow, frame, vector_direction_sign=-1.0, job_id=None, frame_index=None):
        channels = self.extract_flow_channels(flow, "vectors", job_id=job_id, frame_index=frame_index)
        if channels is None:
            return frame
        u, v = channels

        flow_h, flow_w = u.shape
        frame_h, frame_w = frame.shape[:2]
        
        x_scale = frame_w / flow_w
        y_scale = frame_h / flow_h
        
        grid_step = max(self.draw_step, int(round(min(frame_w, frame_h) / 18.0)))
        start_x = self.compute_centered_grid_start(frame_w, grid_step)
        start_y = self.compute_centered_grid_start(frame_h, grid_step)

        samples = []
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
                    screen_x += grid_step
                    continue

                magnitude = math.hypot(float(fx), float(fy))
                if magnitude > 1e-3:
                    samples.append((screen_x, screen_y, float(fx), float(fy), magnitude))

                screen_x += grid_step
            screen_y += grid_step

        if not samples:
            return frame

        magnitudes = np.asarray([sample[4] for sample in samples], dtype=np.float32)
        motion_threshold = max(
            self.min_motion_magnitude,
            float(np.percentile(magnitudes, self.vector_activity_percentile)) * 0.60,
        )
        motion_peak = float(np.percentile(magnitudes, self.vector_peak_percentile))
        motion_peak = max(motion_peak, motion_threshold + 1e-3)

        arrow_specs = []
        max_endpoint_x = max(frame_w * 1.5, 1.0)
        max_endpoint_y = max(frame_h * 1.5, 1.0)
        for screen_x, screen_y, fx, fy, magnitude in samples:
            if magnitude < motion_threshold:
                continue
            strength = max(0.0, min(1.0, (magnitude - motion_threshold) / (motion_peak - motion_threshold)))
            raw_dx = fx * vector_direction_sign
            raw_dy = fy * vector_direction_sign
            raw_magnitude = math.hypot(raw_dx, raw_dy)
            if raw_magnitude <= 1e-6:
                continue

            display_length = raw_magnitude * self.vector_length_multiplier
            display_length = max(self.min_display_vector_length, min(self.max_display_vector_length, display_length))
            display_dx = (raw_dx / raw_magnitude) * display_length
            display_dy = (raw_dy / raw_magnitude) * display_length
            end_x = float(screen_x + display_dx)
            end_y = float(screen_y + display_dy)
            if (
                not (math.isfinite(end_x) and math.isfinite(end_y))
                or abs(end_x) > max_endpoint_x
                or abs(end_y) > max_endpoint_y
            ):
                invalid_vectors += 1
                continue

            color_index = int(max(48, min(255, round(64 + strength * 191))))
            color = tuple(int(channel) for channel in self.turbo_lut[color_index])
            arrow_specs.append(
                (
                    strength,
                    (int(screen_x), int(screen_y)),
                    (int(round(end_x)), int(round(end_y))),
                    color,
                )
            )

        if not arrow_specs:
            return frame

        arrow_specs.sort(key=lambda spec: spec[0])
        thickness = max(1, int(round(min(frame_w, frame_h) / 420.0)))
        shadow_thickness = thickness + 2
        shadow_layer = np.copy(frame)
        for _, start_pt, end_pt, _ in arrow_specs:
            cv2.arrowedLine(
                shadow_layer,
                start_pt,
                end_pt,
                (10, 10, 10),
                shadow_thickness,
                line_type=cv2.LINE_AA,
                tipLength=0.28,
            )

        result_frame = cv2.addWeighted(
            shadow_layer,
            self.vector_shadow_alpha,
            frame,
            1.0 - self.vector_shadow_alpha,
            0.0,
        )
        for _, start_pt, end_pt, color in arrow_specs:
            cv2.arrowedLine(
                result_frame,
                start_pt,
                end_pt,
                color,
                thickness,
                line_type=cv2.LINE_AA,
                tipLength=0.28,
            )
            cv2.circle(result_frame, start_pt, self.dot_radius, (245, 245, 245), -1, lineType=cv2.LINE_AA)

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
        cancel_callback=None,
    ):
        mode = (mode or "VECTORS").upper()
        if mode == "VECTOR":
            mode = "VECTORS"
        if mode not in ("VECTORS", "HEATMAP"):
            logger.warning("Unknown mode requested, defaulting to VECTORS job_id=%s requested_mode=%s", req_id, mode)
            mode = "VECTORS"

        logger.info(
            "Video processing initializing job_id=%s mode=%s input_path=%s output_path=%s vector_direction_sign=%.1f flow_frame_offset=%s",
            req_id,
            mode,
            input_video_path,
            output_video_path,
            vector_direction_sign,
            self.flow_frame_offset,
        )

        status_path = None
        if req_id is not None:
            status_path = os.path.join('temp_videos', f"{req_id}_status.json")
            try:
                os.makedirs(os.path.dirname(status_path), exist_ok=True)
            except Exception as e:
                logger.warning("Could not create status directory job_id=%s status_path=%s error=%s", req_id, status_path, e)
                status_path = None

        cancelled = False
        last_logged_progress = -1
        def raise_if_cancelled():
            nonlocal cancelled
            if cancel_callback is not None:
                try:
                    if cancel_callback():
                        cancelled = True
                        raise ProcessingCancelled(f"Video job cancelled job_id={req_id}")
                except ProcessingCancelled:
                    raise
                except Exception as e:
                    logger.warning("Cancel callback failed job_id=%s error=%s", req_id, e)

        def report_progress(percent):
            nonlocal last_logged_progress
            raise_if_cancelled()
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
        raise_if_cancelled()

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

            ret, first_frame = cap.read()
            if not ret:
                raise Exception(f"Failed to read first frame from video: {input_video_path}")

            height, width = first_frame.shape[:2]
            if width <= 0 or height <= 0:
                raise Exception(f"Invalid frame dimensions width={width} height={height}")

            out = H264Mp4Writer(output_video_path, fps, width, height, req_id=req_id).open()

            logger.info(
                "Video metadata job_id=%s mode=%s input_fps=%.3f output_fps=%.3f width=%s height=%s total_frames=%s flow_frame_offset=%s input_path=%s output_path=%s codec=h264",
                req_id,
                mode,
                fps,
                out.output_fps,
                width,
                height,
                total_frames,
                self.flow_frame_offset,
                input_video_path,
                output_video_path,
            )

            frame_buffer = [first_frame]

            while True:
                raise_if_cancelled()
                ret, curr_frame = cap.read()
                if not ret:
                    break

                frame_buffer.append(curr_frame)
                if len(frame_buffer) <= self.flow_frame_offset:
                    continue

                # Match the OpenCV Zoo demo: estimate flow across a 3-frame gap,
                # but keep this service's output frame count unchanged.
                source_frame = frame_buffer[0]
                comparison_frame = frame_buffer[-1]
                frame_index = frames_processed + 1
                flow_output = None
                try:
                    raise_if_cancelled()
                    flow_output = self.infer(source_frame, comparison_frame)
                    raise_if_cancelled()
                    if frames_processed == 0:
                        logger.info(
                            "First flow output job_id=%s mode=%s frame=%s flow_summary=%s",
                            req_id,
                            mode,
                            frame_index,
                            self.summarize_flow(flow_output),
                        )

                    if mode == "HEATMAP":
                        result_frame = self.draw_heatmap(flow_output, source_frame, job_id=req_id, frame_index=frame_index)
                    else:
                        result_frame = self.draw_vectors(flow_output, source_frame, vector_direction_sign, job_id=req_id, frame_index=frame_index)
                except Exception as e:
                    flow_summary = self.summarize_flow(flow_output) if flow_output is not None else None
                    logger.exception(
                        "Frame processing failed job_id=%s mode=%s frame=%s source_frame_shape=%s comparison_frame_shape=%s flow_summary=%s error=%s",
                        req_id,
                        mode,
                        frame_index,
                        getattr(source_frame, "shape", None),
                        getattr(comparison_frame, "shape", None),
                        flow_summary,
                        e,
                    )
                    raise RuntimeError(f"Frame {frame_index} processing failed in {mode} mode: {e}") from e
                    
                out.write(result_frame)
                frames_processed += 1
                frame_buffer.pop(0)
                # update status every 5 frames
                if total_frames > 0 and frames_processed % 5 == 0:
                    report_progress((frames_processed / total_frames) * 100)

            while frame_buffer:
                raise_if_cancelled()
                out.write(frame_buffer.pop(0))
                frames_processed += 1
                if total_frames > 0 and frames_processed % 5 == 0:
                    report_progress((frames_processed / total_frames) * 100)

            out.release()
            out = None
            completed = True
            logger.info(
                "Video processing finished job_id=%s mode=%s frames_processed=%s total_frames=%s codec=h264",
                req_id,
                mode,
                frames_processed,
                total_frames,
            )
        finally:
            cap.release()
            if out is not None:
                if cancelled:
                    out.cancel()
                else:
                    out.release()
            if completed:
                report_progress(100)

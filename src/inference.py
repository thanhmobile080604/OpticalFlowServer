import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import cv2
cv2.setNumThreads(max(1, int(os.getenv("OPTICAL_FLOW_OPENCV_THREADS", "1"))))
import numpy as np
import math
import json
import logging
import shutil
import subprocess
import onnxruntime as ort
import site
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

logger = logging.getLogger("optical_flow.inference")


class ProcessingCancelled(Exception):
    pass


@dataclass(frozen=True)
class NormalizedPoint:
    x: float
    y: float


@dataclass(frozen=True)
class NormalizedRoi:
    left: float
    top: float
    right: float
    bottom: float
    view_aspect_ratio: float
    path_points: list
    selected_position_ms: int = 0


@dataclass(frozen=True)
class ActiveRoi:
    x: int
    y: int
    width: int
    height: int
    mask: Optional[np.ndarray] = None


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
        self.cuda_dll_dirs = []
        self._cuda_dll_directory_handles = []
        self.gpu_fallback_error = None
        # Use onnxruntime session instead of OpenCV DNN to support quantized ONNX models
        self.available_providers = ort.get_available_providers()
        providers = self.resolve_execution_providers()
        session_options = self.create_session_options()
        self.preload_cuda_dlls_if_needed(providers)
        logger.info(
            "Initializing ONNX session model=%s requested_providers=%s available_providers=%s",
            self.model_path,
            self.provider_names(providers),
            self.available_providers,
        )
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers,
            )
        except Exception as e:
            if "CPUExecutionProvider" not in self.available_providers:
                raise
            logger.warning(
                "GPU/preferred ONNX providers failed, falling back to CPU model=%s providers=%s error=%s",
                self.model_path,
                self.provider_names(providers),
                e,
            )
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
        self.active_providers = self.session.get_providers()
        self.execution_provider = self.active_providers[0] if self.active_providers else "unknown"
        if self.execution_provider in ("CUDAExecutionProvider", "DmlExecutionProvider"):
            logger.info("ONNX Runtime is using GPU provider=%s model=%s", self.execution_provider, self.model_path)
        else:
            logger.warning(
                "ONNX Runtime is not using a GPU provider=%s available_providers=%s. "
                "Install a compatible GPU provider/runtime to enable GPU inference.",
                self.execution_provider,
                self.available_providers,
            )
        self.flow_output_index = self.select_flow_output_index(self.session.get_outputs())
        logger.info(
            "ONNX session ready model=%s providers=%s inputs=%s outputs=%s flow_output_index=%s",
            self.model_path,
            self.active_providers,
            [(item.name, item.shape, item.type) for item in self.session.get_inputs()],
            [(item.name, item.shape, item.type) for item in self.session.get_outputs()],
            self.flow_output_index,
        )
        
        self.input_width = max(64, int(os.getenv("OPTICAL_FLOW_INPUT_WIDTH", "480")))
        self.input_height = max(64, int(os.getenv("OPTICAL_FLOW_INPUT_HEIGHT", "360")))
        self.flow_frame_offset = max(1, int(os.getenv("OPTICAL_FLOW_FRAME_OFFSET", "3")))
        self.roi_flow_frame_offset = max(1, int(os.getenv("OPTICAL_FLOW_ROI_FRAME_OFFSET", "2")))
        self.roi_min_motion_magnitude = max(0.0, float(os.getenv("OPTICAL_FLOW_ROI_MIN_MOTION", "0.05")))
        
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
        self.cutie_model = None
        self.cutie_cfg = None
        self.cutie_device = None
        self.roi_segmentation_backend = os.getenv("ROI_SEGMENTATION_BACKEND", "cutie").strip().lower()
        if self.roi_segmentation_backend not in ("static", "cutie"):
            logger.warning(
                "Unsupported ROI_SEGMENTATION_BACKEND=%s; falling back to static ROI",
                self.roi_segmentation_backend,
            )
            self.roi_segmentation_backend = "static"
        self.roi_fallback_to_static = self.env_bool("ROI_FALLBACK_TO_STATIC", default=False)
        self.cutie_repo_path = os.getenv("CUTIE_REPO_PATH", "").strip()
        self.cutie_weights = os.getenv("CUTIE_WEIGHTS", "").strip()
        self.cutie_device_preference = os.getenv("CUTIE_DEVICE", "auto").strip().lower()
        self.cutie_max_internal_size = int(os.getenv("CUTIE_MAX_INTERNAL_SIZE", "1080"))
        self.cutie_mem_every = max(1, int(os.getenv("CUTIE_MEM_EVERY", "3")))
        self.cutie_auto_download = self.env_bool("CUTIE_AUTO_DOWNLOAD", default=True)

    def create_session_options(self):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return session_options

    def resolve_execution_providers(self):
        configured = os.getenv("OPTICAL_FLOW_ONNX_PROVIDERS")
        if configured:
            requested = [item.strip() for item in configured.split(",") if item.strip()]
        elif self.env_bool("OPTICAL_FLOW_DISABLE_GPU", default=False):
            requested = ["CPUExecutionProvider"]
        else:
            requested = [
                provider
                for provider in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
                if provider in self.available_providers
            ]

        providers = []
        missing = []
        for provider in requested:
            if provider not in self.available_providers:
                missing.append(provider)
                continue
            if provider == "CUDAExecutionProvider":
                providers.append((provider, self.cuda_provider_options()))
            else:
                providers.append(provider)

        if missing:
            logger.warning(
                "Requested ONNX providers are unavailable requested_missing=%s available_providers=%s",
                missing,
                self.available_providers,
            )
        if providers:
            return providers
        if "CPUExecutionProvider" in self.available_providers:
            return ["CPUExecutionProvider"]
        return None

    def cuda_provider_options(self):
        device_id = os.getenv("OPTICAL_FLOW_CUDA_DEVICE_ID", "0").strip()
        try:
            int(device_id)
        except ValueError:
            logger.warning("Invalid OPTICAL_FLOW_CUDA_DEVICE_ID=%s, using 0", device_id)
            device_id = "0"
        return {"device_id": device_id}

    @staticmethod
    def env_bool(name, default=False):
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in ("1", "true", "yes", "on")

    def should_force_cpu_for_roi(self):
        return self.env_bool("OPTICAL_FLOW_ROI_FORCE_CPU", default=True)

    @staticmethod
    def provider_names(providers):
        if not providers:
            return providers
        names = []
        for provider in providers:
            if isinstance(provider, tuple):
                names.append(provider[0])
            else:
                names.append(provider)
        return names

    def runtime_info(self):
        return {
            "onnxruntime_version": ort.__version__,
            "available_providers": self.available_providers,
            "active_providers": self.active_providers,
            "execution_provider": self.execution_provider,
            "gpu_enabled": self.execution_provider in ("CUDAExecutionProvider", "DmlExecutionProvider"),
            "gpu_fallback_error": self.gpu_fallback_error,
            "cuda_device_id": os.getenv("OPTICAL_FLOW_CUDA_DEVICE_ID", "0"),
            "cuda_dll_dirs": self.cuda_dll_dirs,
            "roi_segmentation_backend": self.roi_segmentation_backend,
            "roi_fallback_to_static": self.roi_fallback_to_static,
            "cutie_repo_path": self.cutie_repo_path,
            "cutie_weights": self.cutie_weights,
            "cutie_device": self.cutie_device,
            "cutie_device_preference": self.cutie_device_preference,
            "cutie_max_internal_size": self.cutie_max_internal_size,
            "cutie_mem_every": self.cutie_mem_every,
            "flow_input_size": [self.input_width, self.input_height],
            "flow_frame_offset": self.flow_frame_offset,
            "roi_flow_frame_offset": self.roi_flow_frame_offset,
            "roi_min_motion_magnitude": self.roi_min_motion_magnitude,
        }

    def switch_to_cpu_provider(self, reason):
        if "CPUExecutionProvider" not in self.available_providers:
            return False
        self.gpu_fallback_error = str(reason)
        logger.warning("Switching ONNX session from CUDA to CPU after inference failure error=%s", reason)
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=self.create_session_options(),
            providers=["CPUExecutionProvider"],
        )
        self.active_providers = self.session.get_providers()
        self.execution_provider = self.active_providers[0] if self.active_providers else "unknown"
        self.flow_output_index = self.select_flow_output_index(self.session.get_outputs())
        return True

    def run_session_with_provider_fallback(self, feed):
        try:
            return self.session.run(None, feed)
        except Exception as e:
            if self.execution_provider in ("CUDAExecutionProvider", "DmlExecutionProvider") and self.switch_to_cpu_provider(e):
                return self.session.run(None, feed)
            raise

    def preload_cuda_dlls_if_needed(self, providers):
        provider_names = self.provider_names(providers) or []
        if "CUDAExecutionProvider" not in provider_names:
            return
        self.add_cuda_dll_search_paths()
        preload = getattr(ort, "preload_dlls", None)
        if preload is None:
            logger.debug("ONNX Runtime does not expose preload_dlls; skipping CUDA DLL preload")
            return
        try:
            preload()
            logger.info("Preloaded ONNX Runtime CUDA/cuDNN/MSVC DLL dependencies")
        except Exception as e:
            logger.warning("Failed to preload ONNX Runtime CUDA DLL dependencies error=%s", e)

    def add_cuda_dll_search_paths(self):
        directories = self.find_cuda_dll_directories()
        if not directories:
            logger.warning("No NVIDIA CUDA/cuDNN DLL directories found in Python site-packages or env")
            return

        existing_path_parts = os.environ.get("PATH", "").split(os.pathsep)
        existing_path_norm = {os.path.normcase(os.path.abspath(item)) for item in existing_path_parts if item}
        path_updates = []

        for directory in directories:
            directory_text = str(directory)
            directory_norm = os.path.normcase(os.path.abspath(directory_text))
            if directory_norm not in existing_path_norm:
                path_updates.append(directory_text)
                existing_path_norm.add(directory_norm)
            add_dll_directory = getattr(os, "add_dll_directory", None)
            if add_dll_directory is not None:
                try:
                    self._cuda_dll_directory_handles.append(add_dll_directory(directory_text))
                except OSError as e:
                    logger.warning("Failed to add CUDA DLL directory path=%s error=%s", directory_text, e)

        if path_updates:
            os.environ["PATH"] = os.pathsep.join(path_updates + existing_path_parts)

        self.cuda_dll_dirs = [str(directory) for directory in directories]
        logger.info("Configured CUDA DLL search directories dirs=%s", self.cuda_dll_dirs)

    def find_cuda_dll_directories(self):
        candidates = []
        configured = os.getenv("OPTICAL_FLOW_CUDA_DLL_DIRS")
        if configured:
            candidates.extend(Path(item.strip()) for item in configured.split(os.pathsep) if item.strip())

        site_roots = []
        try:
            site_roots.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            site_roots.append(site.getusersitepackages())
        except Exception:
            pass

        preferred_packages = (
            "cuda_runtime",
            "cublas",
            "cudnn",
            "cuda_nvrtc",
            "cufft",
            "curand",
            "nvjitlink",
        )
        for root in site_roots:
            nvidia_root = Path(root) / "nvidia"
            for package_name in preferred_packages:
                candidates.append(nvidia_root / package_name / "bin")
            candidates.extend(nvidia_root.glob("*/bin"))

        seen = set()
        directories = []
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            key = os.path.normcase(str(resolved))
            if key in seen or not resolved.is_dir():
                continue
            seen.add(key)
            directories.append(resolved)
        return directories

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

            outputs = self.run_session_with_provider_fallback(feed)
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

    def active_roi(self, frame, roi: Optional[NormalizedRoi], job_id=None):
        if roi is None or frame is None:
            return None

        frame_h, frame_w = frame.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return None

        view_height = 1.0
        view_width = max(float(getattr(roi, "view_aspect_ratio", 1.0) or 1.0), 0.01)
        scale = max(view_width / float(frame_w), view_height / float(frame_h))
        offset_x = ((frame_w * scale) - view_width) / 2.0
        offset_y = ((frame_h * scale) - view_height) / 2.0

        def map_x(normalized_x):
            return int(round(((float(normalized_x) * view_width) + offset_x) / scale))

        def map_y(normalized_y):
            return int(round(((float(normalized_y) * view_height) + offset_y) / scale))

        left = min(max(map_x(roi.left), 0), frame_w - 1)
        top = min(max(map_y(roi.top), 0), frame_h - 1)
        right = min(max(map_x(roi.right), left + 1), frame_w)
        bottom = min(max(map_y(roi.bottom), top + 1), frame_h)
        width = right - left
        height = bottom - top
        if width < 32 or height < 32:
            logger.warning(
                "Ignoring ROI because mapped frame region is too small job_id=%s roi=%s frame_size=%sx%s mapped=%s,%s,%s,%s",
                job_id,
                self.roi_summary(roi),
                frame_w,
                frame_h,
                left,
                top,
                right,
                bottom,
            )
            return None

        mask = self.create_roi_mask(roi, frame_w, frame_h, left, top, width, height, map_x, map_y)
        return ActiveRoi(x=left, y=top, width=width, height=height, mask=mask)

    def create_roi_mask(self, roi, frame_w, frame_h, left, top, width, height, map_x, map_y):
        path_points = getattr(roi, "path_points", None) or []
        if len(path_points) < 3:
            return None

        polygon = []
        for point in path_points:
            point_x = min(max(map_x(point.x), 0), frame_w - 1) - left
            point_y = min(max(map_y(point.y), 0), frame_h - 1) - top
            polygon.append([point_x, point_y])
        if len(polygon) < 3:
            return None

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        return mask

    def selected_frame_index(self, roi: Optional[NormalizedRoi], fps: float, total_frames: int):
        if roi is None:
            return 0
        selected_ms = max(0, int(getattr(roi, "selected_position_ms", 0) or 0))
        frame_index = int(round((selected_ms / 1000.0) * max(float(fps), 1.0)))
        if total_frames > 0:
            frame_index = min(frame_index, total_frames - 1)
        return max(0, frame_index)

    def roi_index_mask(self, frame, roi: NormalizedRoi, job_id=None):
        active = self.active_roi(frame, roi, job_id=job_id)
        if active is None:
            raise RuntimeError("ROI could not be mapped to a valid Cutie prompt mask.")

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if active.mask is None:
            mask[
                active.y:active.y + active.height,
                active.x:active.x + active.width,
            ] = 1
        else:
            mask[
                active.y:active.y + active.height,
                active.x:active.x + active.width,
            ] = (active.mask > 0).astype(np.uint8)
        return mask

    def cutie_frame_to_torch(self, frame, device):
        import torch

        # Match Cutie's process_video.py: OpenCV BGR array, CHW float in [0, 1].
        frame = np.ascontiguousarray(frame.transpose(2, 0, 1))
        return torch.from_numpy(frame).float().to(device, non_blocking=True) / 255.0

    def resolve_cutie_device(self, torch):
        requested = self.cutie_device_preference
        if requested in ("", "auto"):
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUTIE_DEVICE=cuda requested but CUDA is unavailable; using CPU")
            return "cpu"
        if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("CUTIE_DEVICE=mps requested but MPS is unavailable; using CPU")
            return "cpu"
        return requested

    def load_cutie_model(self):
        if self.cutie_model is not None:
            return self.cutie_model, self.cutie_cfg, self.cutie_device

        import sys
        repo_root = Path(__file__).resolve().parents[1]
        vendor_cutie_root = repo_root / "third_party" / "Cutie"
        configured_cutie_root = Path(self.cutie_repo_path).resolve() if self.cutie_repo_path else None
        preferred_cutie_root = configured_cutie_root or (vendor_cutie_root if vendor_cutie_root.exists() else None)
        if preferred_cutie_root is not None:
            cutie_repo = str(preferred_cutie_root)
            if cutie_repo not in sys.path:
                sys.path.insert(0, cutie_repo)

        try:
            import torch
            import cutie.config as cutie_config
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            from omegaconf import open_dict
            from cutie.model.cutie import CUTIE
        except Exception as e:
            raise RuntimeError(
                "Cutie is required for ROI segmentation. Install it with: "
                "git clone https://github.com/hkchengrex/Cutie.git && cd Cutie && pip install -e ."
            ) from e

        cutie_root = preferred_cutie_root or Path(cutie_config.__file__).resolve().parents[2]
        config_dir = cutie_root / "cutie" / "config"
        if not config_dir.exists():
            raise RuntimeError(
                f"Cutie config directory not found: {config_dir}. Set CUTIE_REPO_PATH to the Cutie checkout."
            )

        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3.2", config_dir=str(config_dir), job_name="optical_flow_cutie"):
            cfg = compose(config_name="video_config")

        weights_path = self.cutie_weights
        if not weights_path:
            if self.cutie_auto_download:
                from cutie.utils.download_models import download_models_if_needed
                weights_path = str(Path(download_models_if_needed()) / "cutie-base-mega.pth")
            else:
                weights_path = str(cutie_root / "weights" / "cutie-base-mega.pth")
        if not Path(weights_path).exists():
            raise RuntimeError(
                f"Cutie weights not found: {weights_path}. Run python cutie/utils/download_models.py "
                "in the Cutie repo or set CUTIE_WEIGHTS."
            )

        device = self.resolve_cutie_device(torch)
        with open_dict(cfg):
            cfg["weights"] = weights_path
            cfg["device"] = device
            cfg["max_internal_size"] = self.cutie_max_internal_size
            cfg["mem_every"] = self.cutie_mem_every

        logger.info(
            "Loading Cutie model device=%s weights=%s max_internal_size=%s mem_every=%s",
            device,
            weights_path,
            self.cutie_max_internal_size,
            self.cutie_mem_every,
        )
        model = CUTIE(cfg).to(device).eval()
        model_weights = torch.load(weights_path, map_location=device)
        model.load_weights(model_weights)

        self.cutie_model = model
        self.cutie_cfg = cfg
        self.cutie_device = device
        return self.cutie_model, self.cutie_cfg, self.cutie_device

    def read_video_frames(self, input_video_path, first_frame=None, total_frames=0, req_id=None, cancel_callback=None):
        frames = []
        if first_frame is not None:
            frames.append(first_frame)

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for Cutie segmentation: {input_video_path}")
        try:
            if first_frame is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            while True:
                if cancel_callback is not None and cancel_callback():
                    raise ProcessingCancelled(f"Video job cancelled job_id={req_id}")
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if total_frames > 0 and len(frames) >= total_frames:
                    break
        finally:
            cap.release()
        return frames

    def cutie_segment_video(
        self,
        input_video_path,
        first_frame,
        roi: NormalizedRoi,
        fps: float,
        total_frames: int,
        req_id=None,
        progress_callback=None,
        cancel_callback=None,
    ):
        import torch

        def raise_if_cancelled():
            if cancel_callback is not None and cancel_callback():
                raise ProcessingCancelled(f"Video job cancelled job_id={req_id}")

        prompt_frame_idx = self.selected_frame_index(roi, fps, total_frames)
        frames = self.read_video_frames(
            input_video_path,
            first_frame=first_frame,
            total_frames=total_frames,
            req_id=req_id,
            cancel_callback=cancel_callback,
        )
        if not frames:
            raise RuntimeError("Cutie could not read any video frames.")
        prompt_frame_idx = min(prompt_frame_idx, len(frames) - 1)
        prompt_mask = self.roi_index_mask(frames[prompt_frame_idx], roi, job_id=req_id)
        if cv2.countNonZero(prompt_mask) <= 0:
            raise RuntimeError("Cutie prompt mask is empty.")

        model, cfg, device = self.load_cutie_model()
        from cutie.inference.inference_core import InferenceCore
        use_amp = bool(getattr(cfg, "amp", True)) and device == "cuda"
        masks_by_frame = {}

        def run_sequence(frame_indices, progress_base, progress_span):
            processor = InferenceCore(model, cfg=cfg)
            processor.max_internal_size = self.cutie_max_internal_size
            for step_idx, frame_idx in enumerate(frame_indices):
                raise_if_cancelled()
                frame_t = self.cutie_frame_to_torch(frames[frame_idx], device)
                if step_idx == 0:
                    mask_t = torch.from_numpy(prompt_mask.astype(np.int64)).to(device)
                    prob = processor.step(frame_t, mask_t, objects=[1], force_permanent=True)
                else:
                    prob = processor.step(frame_t)
                out_mask = processor.output_prob_to_mask(prob).detach().to("cpu").numpy().astype(np.uint8)
                masks_by_frame[int(frame_idx)] = (out_mask == 1).astype(np.uint8) * 255
                if progress_callback is not None and len(frame_indices) > 0:
                    progress = progress_base + int(((step_idx + 1) / len(frame_indices)) * progress_span)
                    progress_callback(min(100, progress))

        logger.info(
            "Cutie segmentation starting job_id=%s prompt_frame=%s prompt_mask_area=%s frames=%s device=%s",
            req_id,
            prompt_frame_idx,
            cv2.countNonZero(prompt_mask),
            len(frames),
            device,
        )
        with torch.inference_mode():
            with torch.amp.autocast(device, enabled=use_amp):
                run_sequence(range(prompt_frame_idx, len(frames)), 1, 54)
                if prompt_frame_idx > 0:
                    run_sequence(range(prompt_frame_idx, -1, -1), 55, 44)

        if not masks_by_frame:
            raise RuntimeError("Cutie did not produce any object masks.")
        logger.info(
            "Cutie segmentation finished job_id=%s mask_frames=%s total_frames=%s motion=%s",
            req_id,
            len(masks_by_frame),
            total_frames,
            self.mask_motion_summary(masks_by_frame),
        )
        return masks_by_frame

    def points_hit_mask(mask, points):
        if mask is None or points is None:
            return 0
        if mask.ndim == 3:
            mask = mask[0]
        height, width = mask.shape[:2]
        hits = 0
        for point_x, point_y in np.asarray(points):
            x = int(round(float(point_x)))
            y = int(round(float(point_y)))
            if 0 <= x < width and 0 <= y < height and mask[y, x] > 0:
                hits += 1
        return hits

    def mask_for_frame(self, masks_by_frame, frame_idx):
        if not masks_by_frame:
            return None
        if frame_idx in masks_by_frame:
            return masks_by_frame[frame_idx]
        nearest_idx = min(masks_by_frame.keys(), key=lambda existing_idx: (abs(int(existing_idx) - int(frame_idx)), int(existing_idx)))
        return masks_by_frame.get(nearest_idx)

    def mask_motion_summary(self, masks_by_frame):
        if not masks_by_frame:
            return None
        centers = []
        areas = []
        for frame_idx in sorted(masks_by_frame.keys()):
            mask = masks_by_frame.get(frame_idx)
            if mask is None:
                continue
            if mask.ndim == 3:
                mask = mask[0]
            mask = (mask > 0).astype(np.uint8) * 255
            moments = cv2.moments(mask)
            area = float(moments["m00"])
            if area <= 0:
                continue
            centers.append((float(moments["m10"] / area), float(moments["m01"] / area)))
            areas.append(cv2.countNonZero(mask))
        if not centers:
            return None
        xs = [item[0] for item in centers]
        ys = [item[1] for item in centers]
        return {
            "frames": len(centers),
            "x_span": round(max(xs) - min(xs), 2),
            "y_span": round(max(ys) - min(ys), 2),
            "area_min": int(min(areas)),
            "area_max": int(max(areas)),
        }

    def active_roi_from_mask(self, mask, extra_masks=None):
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = mask[0]
        mask = (mask > 0).astype(np.uint8) * 255
        box_mask = mask
        for extra_mask in extra_masks or []:
            if extra_mask is None:
                continue
            if extra_mask.ndim == 3:
                extra_mask = extra_mask[0]
            extra_mask = (extra_mask > 0).astype(np.uint8) * 255
            if extra_mask.shape != box_mask.shape:
                extra_mask = cv2.resize(
                    extra_mask,
                    (box_mask.shape[1], box_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            box_mask = cv2.bitwise_or(box_mask, extra_mask)

        coords = cv2.findNonZero(box_mask)
        if coords is None:
            return None
        x, y, width, height = cv2.boundingRect(coords)
        if width < 2 or height < 2:
            return None
        return ActiveRoi(
            x=int(x),
            y=int(y),
            width=int(width),
            height=int(height),
            mask=mask[y:y + height, x:x + width],
        )

    def draw_flow_result(self, flow_output, frame, mode, vector_direction_sign, active_roi=None, job_id=None, frame_index=None):
        if active_roi is None:
            if mode == "HEATMAP":
                return self.draw_heatmap(flow_output, frame, job_id=job_id, frame_index=frame_index)
            return self.draw_vectors(flow_output, frame, vector_direction_sign, job_id=job_id, frame_index=frame_index)

        roi_view = frame[
            active_roi.y:active_roi.y + active_roi.height,
            active_roi.x:active_roi.x + active_roi.width,
        ]
        if mode == "HEATMAP":
            result_roi = self.draw_heatmap(
                flow_output,
                roi_view,
                job_id=job_id,
                frame_index=frame_index,
                min_motion_magnitude=self.roi_min_motion_magnitude,
            )
        else:
            result_roi = self.draw_vectors(flow_output, roi_view, vector_direction_sign, job_id=job_id, frame_index=frame_index)

        if active_roi.mask is None:
            roi_view[:] = result_roi
        else:
            np.copyto(roi_view, result_roi, where=active_roi.mask[:, :, np.newaxis] > 0)
        return frame

    def roi_summary(self, roi):
        if roi is None:
            return None
        return {
            "left": round(float(roi.left), 4),
            "top": round(float(roi.top), 4),
            "right": round(float(roi.right), 4),
            "bottom": round(float(roi.bottom), 4),
            "view_aspect_ratio": round(float(roi.view_aspect_ratio), 4),
            "path_points": len(getattr(roi, "path_points", None) or []),
        }

    def draw_heatmap(self, flow, frame, job_id=None, frame_index=None, min_motion_magnitude=None):
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

        motion_min = self.min_motion_magnitude if min_motion_magnitude is None else float(min_motion_magnitude)
        active = magnitude[magnitude > motion_min]
        if active.size == 0 and min_motion_magnitude is not None:
            fallback_floor = max(1e-4, float(np.percentile(magnitude, 65.0)) * 0.35)
            active = magnitude[magnitude > fallback_floor]
            motion_min = fallback_floor
        if active.size == 0:
            return frame

        motion_floor = max(
            motion_min,
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
        roi: Optional[NormalizedRoi] = None,
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
        effective_flow_frame_offset = self.roi_flow_frame_offset if roi is not None else self.flow_frame_offset

        logger.info(
            "Video processing initializing job_id=%s mode=%s input_path=%s output_path=%s vector_direction_sign=%.1f flow_frame_offset=%s roi=%s",
            req_id,
            mode,
            input_video_path,
            output_video_path,
            vector_direction_sign,
            effective_flow_frame_offset,
            self.roi_summary(roi),
        )
        if (
            roi is not None
            and self.execution_provider == "DmlExecutionProvider"
            and self.should_force_cpu_for_roi()
        ):
            self.switch_to_cpu_provider("ROI segmentation jobs default to CPU optical-flow provider to avoid DirectML iGPU stalls")

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

            object_masks_by_frame = None
            if roi is not None and self.roi_segmentation_backend == "cutie":
                try:
                    object_masks_by_frame = self.cutie_segment_video(
                        input_video_path=input_video_path,
                        first_frame=first_frame,
                        roi=roi,
                        fps=fps,
                        total_frames=total_frames,
                        req_id=req_id,
                        progress_callback=lambda percent: report_progress(min(60, int(percent * 0.6))),
                        cancel_callback=cancel_callback,
                    )
                except ProcessingCancelled:
                    raise
                except Exception as e:
                    if not self.roi_fallback_to_static:
                        raise
                    object_masks_by_frame = None
                    logger.error(
                        "Cutie ROI segmentation failed; falling back to static ROI job_id=%s mode=%s error=%s",
                        req_id,
                        mode,
                        e,
                        exc_info=True,
                    )
            elif roi is not None:
                logger.info(
                    "Using static ROI segmentation backend job_id=%s mode=%s backend=%s",
                    req_id,
                    mode,
                    self.roi_segmentation_backend,
                )

            active_roi = None if object_masks_by_frame is not None else self.active_roi(first_frame, roi, job_id=req_id)
            out = H264Mp4Writer(output_video_path, fps, width, height, req_id=req_id).open()

            logger.info(
                "Video metadata job_id=%s mode=%s input_fps=%.3f output_fps=%.3f width=%s height=%s total_frames=%s flow_frame_offset=%s active_roi=%s input_path=%s output_path=%s codec=h264",
                req_id,
                mode,
                fps,
                out.output_fps,
                width,
                height,
                total_frames,
                effective_flow_frame_offset,
                None if active_roi is None else {
                    "x": active_roi.x,
                    "y": active_roi.y,
                    "width": active_roi.width,
                    "height": active_roi.height,
                    "has_mask": active_roi.mask is not None,
                },
                input_video_path,
                output_video_path,
            )

            frame_buffer = [first_frame]

            def report_flow_progress(processed_frames):
                if total_frames <= 0:
                    return
                raw_percent = (processed_frames / total_frames) * 100
                if object_masks_by_frame is not None:
                    report_progress(60 + int(raw_percent * 0.4))
                else:
                    report_progress(raw_percent)

            while True:
                raise_if_cancelled()
                ret, curr_frame = cap.read()
                if not ret:
                    break

                frame_buffer.append(curr_frame)
                if len(frame_buffer) <= effective_flow_frame_offset:
                    continue

                # Keep ROI output close to the object mask; non-ROI keeps the wider
                # frame gap for stronger full-frame motion visualization.
                source_frame = frame_buffer[0]
                comparison_frame = frame_buffer[-1]
                frame_index = frames_processed + 1
                source_for_inference = source_frame
                comparison_for_inference = comparison_frame
                frame_active_roi = active_roi
                if object_masks_by_frame is not None:
                    frame_mask = self.mask_for_frame(object_masks_by_frame, frames_processed)
                    comparison_mask = self.mask_for_frame(
                        object_masks_by_frame,
                        frames_processed + effective_flow_frame_offset,
                    )
                    frame_active_roi = self.active_roi_from_mask(
                        frame_mask,
                        extra_masks=[comparison_mask],
                    )
                    if frame_active_roi is None:
                        out.write(source_frame)
                        frames_processed += 1
                        frame_buffer.pop(0)
                        if total_frames > 0 and frames_processed % 5 == 0:
                            report_flow_progress(frames_processed)
                        continue

                if frame_active_roi is not None:
                    source_for_inference = source_frame[
                        frame_active_roi.y:frame_active_roi.y + frame_active_roi.height,
                        frame_active_roi.x:frame_active_roi.x + frame_active_roi.width,
                    ]
                    comparison_for_inference = comparison_frame[
                        frame_active_roi.y:frame_active_roi.y + frame_active_roi.height,
                        frame_active_roi.x:frame_active_roi.x + frame_active_roi.width,
                    ]

                flow_output = None
                try:
                    raise_if_cancelled()
                    flow_output = self.infer(source_for_inference, comparison_for_inference)
                    raise_if_cancelled()
                    if frames_processed == 0:
                        logger.info(
                            "First flow output job_id=%s mode=%s frame=%s roi_active=%s flow_summary=%s",
                            req_id,
                            mode,
                            frame_index,
                            frame_active_roi is not None,
                            self.summarize_flow(flow_output),
                        )

                    result_frame = self.draw_flow_result(
                        flow_output,
                        source_frame,
                        mode,
                        vector_direction_sign,
                        active_roi=frame_active_roi,
                        job_id=req_id,
                        frame_index=frame_index,
                    )
                except Exception as e:
                    flow_summary = self.summarize_flow(flow_output) if flow_output is not None else None
                    logger.exception(
                        "Frame processing failed job_id=%s mode=%s frame=%s roi_active=%s source_frame_shape=%s comparison_frame_shape=%s flow_summary=%s error=%s",
                        req_id,
                        mode,
                        frame_index,
                        frame_active_roi is not None,
                        getattr(source_for_inference, "shape", None),
                        getattr(comparison_for_inference, "shape", None),
                        flow_summary,
                        e,
                    )
                    raise RuntimeError(f"Frame {frame_index} processing failed in {mode} mode: {e}") from e
                    
                out.write(result_frame)
                frames_processed += 1
                frame_buffer.pop(0)
                # update status every 5 frames
                if total_frames > 0 and frames_processed % 5 == 0:
                    report_flow_progress(frames_processed)

            while frame_buffer:
                raise_if_cancelled()
                out.write(frame_buffer.pop(0))
                frames_processed += 1
                if total_frames > 0 and frames_processed % 5 == 0:
                    report_flow_progress(frames_processed)

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

# Optical Flow Server - Tài liệu code và API

Tài liệu này mô tả server FastAPI trong repo, các endpoint API, nơi xử lý từng phần trong code, vai trò của model RAFT ONNX, và luồng xử lý video từ lúc client upload đến lúc nhận file MP4 kết quả.

## 1. Tổng quan

Project này là một server xử lý video bằng optical flow.

Client upload một video MP4 lên API. Server đọc video từng frame, dùng model RAFT ONNX để ước lượng chuyển động giữa hai frame, rồi render kết quả thành video MP4 mới theo một trong hai chế độ:

- `VECTORS`: vẽ mũi tên chuyển động lên frame.
- `HEATMAP`: phủ heatmap biểu diễn cường độ chuyển động lên frame.

Server hỗ trợ hai kiểu xử lý:

- Đồng bộ: gọi `POST /process-video`, request giữ mở cho đến khi video xử lý xong.
- Bất đồng bộ theo job: gọi `POST /process-video/jobs`, lấy `job_id`, poll trạng thái, rồi tải kết quả sau. Kiểu này phù hợp hơn khi chạy qua Cloudflare Tunnel hoặc video xử lý lâu.

## 2. Cấu trúc repo

```text
OpticalFlowServer/
# Optical Flow Server — Tài liệu tóm tắt & Cách hoạt động (ngắn gọn)

Tài liệu này trình bày ngắn gọn cơ chế chính của server, đặc biệt giải thích cách server giới hạn số job xử lý đồng thời và cách job được xếp hàng, bất kể client (máy) nào gửi request.

## Tổng quan ngắn

Server là một FastAPI app dùng `OpticalFlowProcessor` (`inference.py`) để xử lý video bằng model RAFT (ONNX). Có hai chế độ output: `VECTORS` (vẽ mũi tên) và `HEATMAP` (phủ heatmap).

Hai luồng chính:
- Sync: `POST /process-video` — xử lý ngay trong request và trả file kết quả.
- Async (job): `POST /process-video/jobs` — tạo `job_id`, xử lý nền bằng `BackgroundTasks`, client poll trạng thái và tải kết quả sau.

## Cơ chế giới hạn job (concurrency)

- Biến môi trường điều khiển:
   - `OPTICAL_FLOW_MAX_CONCURRENT_VIDEO_JOBS` (mặc định `3`) — số job async xử lý song song tối đa trong cùng 1 process.
   - `OPTICAL_FLOW_MAX_PENDING_VIDEO_JOBS` (mặc định `8`) — tổng job đang `queued` + `processing` tối đa.

- Cách hoạt động thực tế (async jobs):
   1. `POST /process-video/jobs` lưu file input tạm và tạo một `VideoJob` với `status = "queued"`.
   2. Server thêm `run_video_job(job_id)` vào `BackgroundTasks`.
   3. `run_video_job()` sẽ chờ `video_job_slots.acquire()` — `video_job_slots` là `threading.Semaphore(MAX_CONCURRENT_VIDEO_JOBS)`.
   4. Khi acquire thành công, job chuyển `status = "processing"` và `processor.process_video()` chạy (blocking).
   5. Sau khi xong (completed/failed/cancelled) semaphore được `release()` để nhường slot cho job khác.

- Kết luận quan trọng: giới hạn là GLOBAl trong process — không phân biệt máy (client) nào gửi job. Nếu `MAX_CONCURRENT_VIDEO_JOBS=5`, server process này cho phép tối đa 5 job async cùng chạy tại một thời điểm.

## Lưu ý / Caveats

- Sync endpoint `POST /process-video` KHÔNG sử dụng semaphore; nó gọi `processor.process_video()` trực tiếp. Do đó các request đồng bộ có thể gây chạy thêm các xử lý song song ngoài giới hạn semaphore.
- Nếu bạn chạy nhiều worker (ví dụ Uvicorn với `--workers N`) hoặc có nhiều process, mỗi process có semaphore riêng — tổng concurrency trên tất cả process = `MAX_CONCURRENT_VIDEO_JOBS * N`.
- Nếu deploy nhiều server (load balancer), trạng thái job (`video_jobs`, semaphore) KHÔNG được chia sẻ giữa các máy trừ khi bạn thay đổi cơ chế (Redis queue, worker service, v.v.).

## Ngắn gọn về endpoints (chính)

- `POST /process-video` — xử lý đồng bộ, trả `video/mp4`.
- `POST /process-video/jobs` — tạo async job, trả `job_id` + queue info (status `queued`).
- `GET /process-video/jobs/{job_id}` — lấy trạng thái job (queued/processing/completed/failed/cancelled/cancelling).
- `GET /process-video/jobs/{job_id}/result` — tải file kết quả (sau đó server cleanup file và xóa job khỏi memory).
- `POST /process-video/jobs/{job_id}/cancel` — yêu cầu hủy job; nếu job đang queued -> cancelled; nếu processing -> chuyển thành `cancelling` và pipeline sẽ dừng ở lần check tiếp theo.
- `GET /health` — trạng thái server + `video_jobs` summary.

## Ví dụ: muốn cho chạy 5 job cùng lúc

Trên Windows PowerShell:

```powershell
$env:OPTICAL_FLOW_MAX_CONCURRENT_VIDEO_JOBS = "5"
uvicorn main:app --host 0.0.0.0 --port 8000
```

Sau đó, trong 1 process server này, tối đa 5 job async sẽ ở trạng thái `processing` cùng lúc. Client/máy gửi job nào cũng đều dùng cùng pool slot đó.

## Gợi ý trực tiếp cho câu hỏi của bạn

- Nếu bạn set `OPTICAL_FLOW_MAX_CONCURRENT_VIDEO_JOBS=5` thì đúng: server process này có 5 slot processing (không phân biệt máy gửi job). Máy A gửi job A và máy B gửi job B đều dùng cùng bộ slot.
- Nhớ rằng `POST /process-video` (sync) không bị semaphore giới hạn — nó chạy ngay lập tức trong request handler.
- Nếu bạn chạy nhiều worker/process, tổng số job đồng thời = `MAX_CONCURRENT_VIDEO_JOBS * number_of_processes`.

## Nơi chỉnh sửa liên quan

- Giới hạn concurrent/pending: [main.py](main.py#L1)
- Logic job enqueue/run/cancel: [main.py](main.py#L1)
- Pipeline xử lý video và cancel checks: [inference.py](inference.py#L1)

---
Phiên bản này rút gọn và nhấn mạnh phần concurrency/behaviour. Nếu bạn muốn tôi mở rộng lại phần API examples hoặc giữ toàn bộ nội dung chi tiết trước đó, nói tôi sẽ thêm lại.

## 15. Render mode `VECTORS`

Hàm:

```python
draw_vectors(flow, frame, vector_direction_sign=-1.0, job_id=None, frame_index=None)
```

Mục tiêu: vẽ mũi tên optical flow lên frame.

Luồng chính:

1. Tách flow thành `u`, `v`.
2. Scale flow từ kích thước model về kích thước frame gốc:
   - `x_scale = frame_w / flow_w`
   - `y_scale = frame_h / flow_h`
3. Tạo grid lấy mẫu trên frame.
4. Lấy flow tại từng điểm grid.
5. Tính magnitude:
   - `sqrt(fx^2 + fy^2)`
6. Lọc vector yếu bằng percentile.
7. Đổi chiều vector bằng `vector_direction_sign`.
8. Clamp độ dài hiển thị để mũi tên không quá ngắn/quá dài.
9. Tô màu mũi tên bằng Turbo colormap.
10. Vẽ shadow layer rồi vẽ mũi tên và dot gốc.

Các tham số render chính:

```python
self.draw_step = 34
self.min_motion_magnitude = 0.45
self.dot_radius = 2
self.vector_length_multiplier = 2.4
self.min_display_vector_length = 10.0
self.max_display_vector_length = 56.0
self.vector_activity_percentile = 58.0
self.vector_peak_percentile = 95.0
self.vector_shadow_alpha = 0.42
```

### `is_moving` ảnh hưởng gì?

Trong `main.py`:

```python
def vector_direction_sign_for_motion(is_moving: bool) -> float:
    return 1.0 if is_moving else -1.0
```

Nếu client gửi:

- `is_moving=true` hoặc `isMoving=true`: `vector_direction_sign = 1.0`
- Không gửi hoặc gửi `false`: `vector_direction_sign = -1.0`

Giá trị này chỉ ảnh hưởng mode `VECTORS`. Mode `HEATMAP` dùng magnitude nên không quan tâm hướng vector.

## 16. Render mode `HEATMAP`

Hàm:

```python
draw_heatmap(flow, frame, job_id=None, frame_index=None)
```

Mục tiêu: tô màu vùng có chuyển động mạnh.

Luồng chính:

1. Tách flow thành `u`, `v`.
2. Scale flow về kích thước frame gốc.
3. Tính magnitude:
   - `sqrt(fx^2 + fy^2)`
4. Gaussian blur magnitude để heatmap mượt hơn.
5. Lọc vùng chuyển động yếu.
6. Tính floor và peak bằng percentile.
7. Normalize magnitude về `0..1`.
8. Apply gamma để tăng độ nhìn.
9. Convert sang `uint8 0..255`.
10. Apply `cv2.COLORMAP_TURBO`.
11. Resize heatmap về kích thước frame gốc.
12. Alpha blend heatmap lên frame nền đã làm tối nhẹ.

Các tham số chính:

```python
self.heatmap_peak_percentile = 98.5
self.heatmap_floor_percentile = 45.0
self.heatmap_gamma = 0.68
self.heatmap_max_alpha = 0.78
self.heatmap_background_weight = 0.72
self.heatmap_min_alpha = 0.08
```

## 17. Async job hoạt động như thế nào?

Luồng async:

1. Client gọi `POST /process-video/jobs`.
2. Server kiểm tra model đã load chưa.
3. Server kiểm tra queue có đầy không.
4. Server lưu upload vào temp input file.
5. Server tạo temp output file.
6. Server tạo `VideoJob` với status `queued`.
7. Server lưu job vào dict `video_jobs`.
8. Server add background task:
   ```python
   background_tasks.add_task(run_video_job, job_id)
   ```
9. Client nhận `job_id`.
10. `run_video_job()` chờ semaphore slot.
11. Khi có slot, status chuyển sang `processing`.
12. `processor.process_video()` chạy inference.
13. Progress callback cập nhật `job.progress`.
14. Nếu xong, status chuyển sang `completed`.
15. Client gọi `/result` để lấy file.

Lưu ý: `BackgroundTasks` của FastAPI/Starlette chạy trong cùng process server, không phải queue worker độc lập như Celery/RQ. Nếu server process chết, job đang chạy mất.

## 18. Temp file và lifecycle

### Sync endpoint

`POST /process-video`:

1. Tạo input temp `.mp4`.
2. Tạo output temp `.mp4`.
3. Xử lý video.
4. Trả `FileResponse`.
5. `BackgroundTasks` cleanup input và output sau response.

### Async endpoint

`POST /process-video/jobs`:

1. Tạo input temp `.mp4`.
2. Tạo output temp `.mp4`.
3. Lưu path trong `VideoJob`.

`run_video_job()`:

- Luôn cleanup input file trong `finally`.
- Nếu failed/cancelled, cleanup output file.
- Nếu completed, giữ output file để client tải.

`GET /process-video/jobs/{job_id}/result`:

- Trả output file.
- Sau response, cleanup output file.
- Sau response, remove job khỏi memory.

## 19. Error handling

Các lỗi thường gặp:

| Lỗi | Nơi phát sinh | Kết quả |
| --- | --- | --- |
| Model file thiếu hoặc load lỗi | Startup trong `main.py` | `processor = None`, API báo model chưa load. |
| Không mở được video input | `cv2.VideoCapture` trong `process_video()` | Job failed hoặc sync trả JSON error. |
| Không đọc được first frame | `process_video()` | Job failed. |
| ONNX inference lỗi | `infer()` | RuntimeError có input metadata và feed shape. |
| ffmpeg không có | `H264Mp4Writer._ffmpeg_exe()` | RuntimeError yêu cầu cài ffmpeg hoặc requirements. |
| ffmpeg encode lỗi | `H264Mp4Writer.release()` | RuntimeError kèm stderr nếu đọc được. |
| Client cancel | `cancel_callback` trong `process_video()` | Raise `ProcessingCancelled`, job chuyển `cancelled`. |
| Queue đầy | `create_process_video_job()` | HTTP `429`. |

## 20. Gợi ý dùng API từ Android/client

Base URL khi chạy local:

```text
http://localhost:8000
```

Khi dùng Android emulator, `localhost` trong emulator không phải máy host. Thường cần dùng:

```text
http://10.0.2.2:8000
```

Khi dùng Cloudflare Tunnel:

```powershell
cloudflared tunnel --url http://localhost:8000
```

Sau đó lấy URL dạng:

```text
https://xxxxx.trycloudflare.com
```

và cấu hình client:

```text
opticalFlowServerBaseUrl=https://xxxxx.trycloudflare.com
```

Với video dài hoặc mạng qua tunnel, nên dùng async API:

1. Upload bằng `POST /process-video/jobs`.
2. Poll `GET /process-video/jobs/{job_id}`.
3. Khi `status=completed`, tải `GET /process-video/jobs/{job_id}/result`.

## 21. Các điểm cần chú ý khi maintain code

- `video_jobs` chỉ lưu trong memory. Không dùng nhiều worker nếu chưa đổi sang Redis/database/shared queue.
- CORS đang mở toàn bộ origin.
- Không có authentication/authorization.
- Không có limit kích thước upload ở FastAPI layer.
- Sync endpoint có thể timeout với video dài hoặc khi đi qua Cloudflare Tunnel.
- File status JSON trong `temp_videos/` được ghi khi có `req_id`, nhưng API status chính lấy từ memory.
- Job `failed` hoặc `cancelled` có thể còn metadata trong memory nếu client không có flow cleanup riêng.
- Output FPS bị giới hạn tối đa 30 FPS.
- Input frame đưa vào RAFT luôn resize về `480x360`, sau đó flow được scale lại để vẽ lên frame gốc.
- `mode=HEATMAP` không dùng `is_moving`.
- `mode=VECTOR` được chấp nhận nhưng normalize thành `VECTORS`.

## 22. Map nhanh: muốn sửa gì thì vào đâu?

| Muốn sửa | File/hàm |
| --- | --- |
| Thêm/sửa endpoint API | `main.py` |
| Đổi logic queue/job/cancel/progress | `main.py`, các hàm quanh `VideoJob`, `run_video_job()` |
| Đổi model path hoặc thứ tự ưu tiên model | `main.py`, block `DEFAULT_MODEL`, `DEQUANT_MODEL`, `ALT_MODEL` |
| Đổi kích thước input RAFT | `inference.py`, `OpticalFlowProcessor.__init__`, `input_width`, `input_height` |
| Đổi khoảng cách frame optical flow | `inference.py`, `flow_frame_offset` |
| Đổi cách preprocess frame | `inference.py`, `prepare_blob()` |
| Đổi cách feed input ONNX | `inference.py`, `infer()` |
| Đổi vẽ mũi tên | `inference.py`, `draw_vectors()` và các vector parameters |
| Đổi heatmap | `inference.py`, `draw_heatmap()` và các heatmap parameters |
| Đổi codec/output video | `inference.py`, `H264Mp4Writer` |
| Đổi progress/cancel check | `inference.py`, `process_video()` và `main.py/run_video_job()` |

## 23. Checklist test thủ công

Sau khi sửa code, nên test tối thiểu:

1. Server start được và `/health` trả `model_loaded=true`.
2. `POST /process-video` với `mode=VECTORS` trả MP4 mở được.
3. `POST /process-video` với `mode=HEATMAP` trả MP4 mở được.
4. `POST /process-video/jobs` trả `job_id`.
5. Poll job thấy `progress` tăng.
6. Khi job `completed`, tải `/result` được MP4.
7. Cancel job đang queue hoặc đang processing trả trạng thái hợp lý.
8. Test video lỗi/không phải video để chắc job chuyển `failed`.
9. Test khi queue đầy bằng cách giảm `OPTICAL_FLOW_MAX_PENDING_VIDEO_JOBS`.


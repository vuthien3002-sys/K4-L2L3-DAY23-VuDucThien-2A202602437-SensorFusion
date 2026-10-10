# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Vũ Đức Thiện
- MSSV: 2A202602437
- Email: vuthien3002@gmail.com
- Link repo (fork): https://github.com/vuthien3002-sys/K4-Track4-Day23-VuDucthien-2A202602437-Sensor-Fusion-Student
- Commit hash nộp (`git rev-parse HEAD`): nộp trên LMS (commit cuối cùng của nhánh `main`)

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]` (199 frame), `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `seed = 0`
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: precision 0.970, recall 0.700, tp 519 / fp 16 / fn 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse **0.1503 m**, matches 502, sum_sq_err 11.344, ghost 0, missed 239, mean_confirmed 2.523
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse **0.1359 m**, matches 502, sum_sq_err 9.267, ghost 0, missed 239, mean_confirmed 2.523
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: xem bên dưới.

**Chỉ số chất lượng (tính lại từ `grade_run.log`):**

| Mode | RMSE (m) | matches | ghost | miss | precision_track = m/(m+ghost) | coverage = m/det_tp |
|---|---|---|---|---|---|---|
| lidar | 0.1503 | 502 | 0 | 239 | 1.000 | 502/519 = 0.967 |
| fused | 0.1359 | 502 | 0 | 239 | 1.000 | 502/519 = 0.967 |

`rmse_fused − rmse_lidar = −0.0145 m` (≤ 0.05 m).

**Diễn giải.** Hai mode có **cùng** `matches` (502), `ghost` (0), `missed` (239) và
`mean_confirmed_tracks` (2.523); kiểm tra từng frame trong `grade_run.log` thì
`det_tp`, `confirmed`, `matches` của `lidar` và `fused` trùng nhau ở **cả 199 frame**.
Nghĩa là camera không làm thay đổi tập track được xác nhận — đúng thiết kế, vì
vòng đời track chỉ do LiDAR quyết định. Khác biệt duy nhất là **độ chính xác vị trí**
trên cùng 502 cặp: `sum_sq_err` giảm từ 11.34 xuống 9.27 m² (−18%), RMSE giảm
0.150 → 0.136 m. Vì số cặp như nhau nên so sánh RMSE ở đây là công bằng (không phải
RMSE giảm nhờ bỏ bớt cặp khó). Theo từng frame, fused có `sum_sq_err` nhỏ hơn lidar
ở 137/195 frame có ghép; ở vài frame đầu (frame 4–6) fused còn kém hơn một chút vì
track mới khởi tạo có `P` vận tốc lớn, update camera 2D kéo trạng thái theo hướng
ngang/dọc ảnh trước khi vận tốc ổn định.

`missed_gt_frames = 239` gồm phần lớn do detector (det_fn = 222 xe không được phát
hiện, tracker không thể có track cho các xe đó), cộng 8 lần miss ở frame 0–3 (2 xe
đang chờ đủ điểm để confirmed, xác nhận ở frame 4) và vài frame track mới khởi tạo
lại. `ghost = 0` cho thấy cổng χ² cùng ngưỡng xác nhận 0.8 lọc được 16 detection FP.

**Giới hạn:** đo camera là tâm hộp 2D **ground-truth** FRONT cộng nhiễu seeded, không
phải camera detector, nên mức cải thiện RMSE ở đây là cận trên lạc quan, không chứng
minh chất lượng một hệ perception camera thật.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. **Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?**
   LiDAR: `z = (x, y, z)` là tâm hộp 3D (m), `R = diag(0.1², 0.1², 0.1²)` m²;
   `h(x)` tuyến tính (`H` 3×6 = phép xoay sensor, cột vận tốc bằng 0). Camera:
   `z = (u, v)` pixel, `R = diag(5², 5²)` px² (`build_camera_measurement`);
   `h(x)` phi tuyến pinhole `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s`
   (`camera_measurement_prediction`), nên `H` là Jacobian tính tại `x` → đây là
   lý do cần **E**KF. Camera không đo được độ sâu: một đo 2D chỉ ràng buộc hướng nhìn,
   không ràng buộc khoảng cách. Cùng hàm `ekf_update` dùng cho cả hai vì nó lấy
   `H`, `h(x)`, `R` từ `meas.sensor`.

2. **Vì sao cần gating Mahalanobis trước khi gán?**
   `d² = γᵀ S⁻¹ γ` chuẩn hoá innovation theo độ bất định `S = HPHᵀ + R`: track mới
   có `P` lớn thì chấp nhận sai lệch lớn, track ổn định thì chặt hơn — khoảng cách
   Euclid không làm được điều đó và không so sánh được m với pixel. Cổng
   `d² < χ²(0.995, dim_meas)` (12.84 với lidar 3D, 10.60 với camera 2D) loại các cặp
   không thể xảy ra trước khi greedy chọn, tránh một detection FP hay một xe khác
   "kéo" track đi; cặp ngoài FOV bị gán `inf` **trước** khi tính Mahalanobis
   (`association_cost_matrix`) để không chiếu điểm sau camera. Kết quả: 16 FP của
   detector không tạo ghost nào (`ghost_track_frames = 0`).

3. **Track-then-fuse hay fuse-then-track? Chỉ ra trên log.**
   **Track-then-fuse.** Mỗi frame trong `run_lab.py`: predict mọi track một lần →
   `associate_and_update(..., lidar_sensor)` → `associate_and_update(..., camera_sensor)`.
   Chỉ có một tracker; camera chỉ tinh chỉnh trạng thái track sẵn có. Bằng chứng trong
   `grade_run.log`: ở cả 199 frame, record `mode: "lidar"` và `mode: "fused"` có cùng
   `det_tp`, `confirmed`, `matches` (vd. frame 0–3 cả hai `confirmed: 0`, frame 4 cả hai
   `confirmed: 2`), chỉ `sum_sq_err` khác (frame 4: 0.0113 vs 0.0320). Nếu là
   fuse-then-track, đo camera sẽ tham gia tạo/xác nhận track và `confirmed` sẽ khác nhau
   giữa hai mode.

4. **Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?**
   Extrinsic sai làm `h(x)` lệch có hệ thống, nên innovation camera `γ = z − h(x)`
   **không còn trung bình 0**: có bias cùng dấu theo thời gian (vd. `u` luôn dương nếu
   lệch yaw), `d²` trung bình tăng vượt `dim_meas = 2`. Lệch nhỏ: vẫn qua cổng χ², update
   camera kéo track lệch khỏi vị trí LiDAR, rồi LiDAR kéo lại → RMSE fused tăng, track
   "rung" giữa hai sensor. Lệch lớn: `d²` vượt ngưỡng, cổng loại hết đo camera → fused
   suy biến về lidar-only (RMSE ≈ lidar), track không hỏng nhưng mất lợi ích camera.

5. **Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Vì sao lidar quyết định score/init/delete còn camera chỉ EKF update?**
   Khi `meas_list` rỗng không thể suy ra sensor từ `meas_list[0].sensor`, nhưng frame
   LiDAR rỗng vẫn phải chạy `manage_tracks`: mọi track trong FOV bị tính một miss
   (`−1/window`) và có thể bị xoá; nếu bỏ qua, track ma sẽ sống mãi khi detector không
   thấy gì. Ngược lại, lượt camera rỗng không được trừ score. `manage_tracks` dùng
   `sensor.name` để phân biệt. LiDAR quyết định tồn tại vì nó đo 3D đầy đủ (có độ sâu),
   phủ toàn bộ vùng BEV phía trước mà detector xét (FOV lidar trong lab `(−π/2, π/2)`)
   và là nguồn duy nhất khởi tạo được `x` 3D; camera FRONT chỉ phủ một góc hẹp hơn
   (FOV suy từ intrinsics và bề rộng ảnh), không có độ sâu và trong lab là nhãn GT có nhiễu, nên nếu camera
   cộng/trừ score thì track ngoài FOV camera bị phạt oan và lifecycle phụ thuộc dữ liệu
   không có trên xe thật. Test `test_camera_pass_never_deletes_or_spawns` và
   `test_lidar_hits_confirm_despite_camera_empty_or_unmatched` kiểm tra điều này.

6. **Điều kiện xác nhận, giữ confirmed sau miss, điều kiện xoá track.**
   Khởi tạo từ đo LiDAR chưa ghép: `score = 1/6`, `state = "initialized"`, vị trí từ
   `z` đổi sang hệ xe, vận tốc 0 với `σ = (50, 50, 5)`. Mỗi hit LiDAR `+1/6` (tối đa 1),
   trạng thái `tentative`; miss LiDAR trong FOV `−1/6`. **Xác nhận** khi
   `score > 0.8` — tức sau 5 hit liên tiếp (frame 0 → 4, khớp log: `confirmed` từ 0 lên 2
   ở frame 4). **Giữ confirmed:** khi đã confirmed, miss chỉ giảm score, không hạ trạng
   thái (1.0 → 0.833 vẫn confirmed). **Xoá** (OR): `P[0,0]` hoặc `P[1,1] > 3² = 9 m²`;
   hoặc confirmed và `score < 0.6` (≥ 3 miss liên tiếp từ 1.0); hoặc chưa confirmed và
   `score ≤ 0` (vd. track mới 1/6 gặp 1 miss).

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude (Claude Code, trong VS Code)
- Dùng cho phần nào (hàm, câu hỏi, debug): cài môi trường (uv, Python 3.12, lỗi mã hoá GBK trên Windows) và tải dữ liệu; viết code Part E–H (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`); chạy `fusion-run-lab`; soạn nháp phần tóm tắt kết quả và 6 câu giải thích. Số liệu trong báo cáo lấy trực tiếp từ `metrics.json` và `grade_run.log` của lần chạy của tôi, không do AI tự đặt ra.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): `pytest student/tests -q` → 128 passed, 0 xfailed; chạy `fusion-run-lab --fusion compare --seed 0` trên frame 0–198; tính lại precision_track, coverage và invariant `matches+ghosts==confirmed`, `matches+misses==valid_gt` từ `grade_run.log`; đối chiếu công thức EKF (`K = PHᵀS⁻¹`), pinhole và χ² với `docs/HUONG_DAN_KY_THUAT.md`; `python tools/check_submission.py`.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))

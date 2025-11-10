import os, glob, json, time
import numpy as np
import cv2 as cv
import yaml
import pyrealsense2 as rs
from dataclasses import dataclass

# ========================= 사용자 설정 =========================
SAVE_DIR         = "out"
FRAME_DIR        = os.path.join(SAVE_DIR, "frames")
IMG_SIZE         = (640, 480)      # (width, height) color
FPS               = 30 
MIN_DET           = 500              # 최소 체커보드 검출 프레임
CHECKER_ROWS      = 6              # 내부 코너(행)
CHECKER_COLS      = 8              # 내부 코너(열)
SQUARE_SIZE_M     = 0.025           # 한 칸 길이[m] (예: 25mm -> 0.025)
ALIGN_TO_COLOR    = True            # depth->color 정렬 사용 권장
DEPTH_MM_TO_M     = 1.0/1000.0      # RealSense z16(mm) -> m
# ===============================================================
@dataclass
class CheckerSpec:
    rows: int
    cols: int
    square: float  # meters

SPEC = CheckerSpec(CHECKER_ROWS, CHECKER_COLS, SQUARE_SIZE_M)

def ensure_dirs():
    os.makedirs(FRAME_DIR, exist_ok=True)

def save_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def obj_points(spec: CheckerSpec):
    objp = np.zeros((spec.rows*spec.cols, 3), np.float32)
    grid = np.mgrid[0:spec.cols, 0:spec.rows].T.reshape(-1, 2)
    objp[:, :2] = grid * spec.square
    return objp

def find_corners(img_bgr, spec: CheckerSpec):
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (spec.cols, spec.rows))
    if not ret:
        return False, None
    term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), term)
    return True, corners

def robust_corner_depth_mm(depth_mm, uv, half=2):
    uv_r = np.round(uv).astype(int)
    H, W = depth_mm.shape[:2]
    z = []
    for u,v in uv_r:
        u0, u1 = max(0,u-half), min(W,u+half+1)
        v0, v1 = max(0,v-half), min(H,v+half+1)
        patch = depth_mm[v0:v1, u0:u1]
        vals = patch[patch>0]
        z.append(np.median(vals) if vals.size>0 else np.nan)
    return np.array(z, dtype=np.float32)

def record_rgbd():
    ensure_dirs()
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, IMG_SIZE[0], IMG_SIZE[1], rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, IMG_SIZE[0], IMG_SIZE[1], rs.format.z16, FPS)
    # IMU 스트림 가능하면 활성화
    imu_enabled = False
    try:
        cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
        cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
        imu_enabled = True
    except Exception:
        imu_enabled = False

    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color) if ALIGN_TO_COLOR else None

    # 공장 intrinsics 저장
    cintr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    dintr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    save_yaml(os.path.join(SAVE_DIR, "factory_intrinsics.yaml"), {
        'color': dict(w=cintr.width, h=cintr.height, fx=cintr.fx, fy=cintr.fy, cx=cintr.ppx, cy=cintr.ppy, dist=list(cintr.coeffs)),
        'depth': dict(w=dintr.width, h=dintr.height, fx=dintr.fx, fy=dintr.fy, cx=dintr.ppx, cy=dintr.ppy, dist=list(dintr.coeffs)),
        'note': 'RealSense factory intrinsics'
    })

    print("[INFO] Recording RGB-D... (화면에서 체커보드가 보이도록 위치 조정, q 누르면 중단)")
    det = 0
    idx = 0

    # IMU 로그 버퍼
    imu_log = []
    last_g = [0.0, 0.0, 0.0]
    last_a = [0.0, 0.0, 0.0]

    try:
        while True:
            frames = pipe.wait_for_frames()
            if ALIGN_TO_COLOR:
                frames = align.process(frames)

            cf = frames.get_color_frame()
            df = frames.get_depth_frame()

            # motion frames 읽기(있을 때만)
            if imu_enabled:
                try:
                    gyro_frame = frames.first_or_default(rs.stream.gyro)
                    accel_frame = frames.first_or_default(rs.stream.accel)
                except Exception:
                    gyro_frame = None
                    accel_frame = None
                if gyro_frame:
                    g = gyro_frame.as_motion_frame().get_motion_data()
                    last_g = [g.x, g.y, g.z]
                if accel_frame:
                    a = accel_frame.as_motion_frame().get_motion_data()
                    last_a = [a.x, a.y, a.z]
                if gyro_frame or accel_frame:
                    # timestamp: motion frame timestamp 우선
                    t_stamp = None
                    if gyro_frame:
                        t_stamp = gyro_frame.get_timestamp()
                    elif accel_frame:
                        t_stamp = accel_frame.get_timestamp()
                    else:
                        t_stamp = frames.get_timestamp()
                    t_ns_imu = int(t_stamp * 1e6)
                    imu_log.append([t_ns_imu, last_g[0], last_g[1], last_g[2], last_a[0], last_a[1], last_a[2]])

            if not cf or not df:
                continue

            color = np.asanyarray(cf.get_data())
            depth = np.asanyarray(df.get_data())  # uint16, mm

            # 체커보드 검출 및 표시
            ok, corners = find_corners(color, SPEC)
            disp = color.copy()
            if ok and corners is not None:
                cv.drawChessboardCorners(disp, (SPEC.cols, SPEC.rows), corners, True)
            # overlay 텍스트
            text1 = f"idx:{idx} det_count:{det}/{MIN_DET}"
            text2 = f"IMU: gx={last_g[0]:.3f}, gy={last_g[1]:.3f}, gz={last_g[2]:.3f}"
            cv.putText(disp, text1, (8,20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv.putText(disp, text2, (8,44), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv.putText(disp, "Press 'q' to stop. Move checkerboard until detection count increases.", (8,470), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            # depth 컬러맵 생성(시각화용)
            # 무한대(0) 제외하고 0-5m 범위를 0-255로 정규화
            depth_vis = depth.copy().astype(np.float32) * DEPTH_MM_TO_M  # m
            depth_mask = (depth_vis > 0)
            if depth_mask.any():
                # normalize to 0..255 using 0..3m clip
                clip_m = 3.0
                dv = np.clip(depth_vis / clip_m * 255.0, 0, 255).astype(np.uint8)
                dv[~depth_mask] = 0
                depth_color = cv.applyColorMap(cv.equalizeHist(dv), cv.COLORMAP_JET)
            else:
                depth_color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

            # 윈도우에 표시 (리사이즈해서 모니터에 맞추기)
            cv.imshow("color_view", disp)
            cv.imshow("depth_view", depth_color)

            ts_ns = int(frames.get_timestamp() * 1e6)
            if ok:
                det += 1

            cv.imwrite(os.path.join(FRAME_DIR, f"color_{idx:06d}.png"), color)
            np.save(os.path.join(FRAME_DIR, f"depth_{idx:06d}.npy"), depth)
            with open(os.path.join(FRAME_DIR, f"meta_{idx:06d}.json"), "w") as f:
                json.dump({'t_ns': ts_ns, 'detected': bool(ok)}, f)

            idx += 1
            if det >= MIN_DET:
                print(f"[INFO] Enough checkerboard detections: {det}")
                break

            k = cv.waitKey(1) & 0xFF
            if k == ord('q'):
                break
    finally:
        pipe.stop()
        cv.destroyAllWindows()
        # IMU 로그 파일로 저장
        if imu_enabled and len(imu_log) > 0:
            import csv
            imu_path = os.path.join(FRAME_DIR, "imu_log.csv")
            with open(imu_path, "w", newline='') as f:
                w = csv.writer(f)
                w.writerow(['t_ns', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])
                w.writerows(imu_log)
            print(f"[INFO] Saved IMU log: {imu_path}")

def calibrate_color_intrinsics():
    objp = obj_points(SPEC)
    objpts, imgpts = [], []
    size = None
    for p in sorted(glob.glob(os.path.join(FRAME_DIR, "color_*.png"))):
        img = cv.imread(p, cv.IMREAD_COLOR)
        if img is None: continue
        if size is None: size = (img.shape[1], img.shape[0])
        ok, corners = find_corners(img, SPEC)
        if ok:
            objpts.append(objp.copy())
            imgpts.append(corners)
    if len(objpts) < 8:
        print("[WARN] Checkerboard detections too few. 더 찍어줘.")
    flags = cv.CALIB_RATIONAL_MODEL
    rms, K, dist, rvecs, tvecs = cv.calibrateCamera(objpts, imgpts, size, None, None, flags=flags)
    save_yaml(os.path.join(SAVE_DIR, "color_intrinsics.yaml"), {
        'image_size': {'w': size[0], 'h': size[1]},
        'K': K.tolist(), 'dist': dist.flatten().tolist(), 'rms': float(rms),
        'checkerboard': {'rows': SPEC.rows, 'cols': SPEC.cols, 'square_m': SPEC.square}
    })
    print(f"[INFO] Color intrinsics RMS: {rms:.4f}")
    return K, dist

def estimate_depth_scale_bias_and_extrinsics(K, dist):
    objp = obj_points(SPEC)
    z_true_all, z_meas_all = [], []
    R_list, t_list = [], []

    paths = sorted(glob.glob(os.path.join(FRAME_DIR, "color_*.png")))
    for p in paths:
        idx = int(os.path.splitext(os.path.basename(p))[0].split('_')[-1])
        img = cv.imread(p, cv.IMREAD_COLOR)
        if img is None: continue
        ok, corners = find_corners(img, SPEC)
        if not ok: continue

        # 보드 pose (Cam<-Board)
        okp, rvec, tvec = cv.solvePnP(objp, corners, K, dist, flags=cv.SOLVEPNP_ITERATIVE)
        if not okp: continue
        R,_ = cv.Rodrigues(rvec)

        depth_mm = np.load(os.path.join(FRAME_DIR, f"depth_{idx:06d}.npy"))
        uv = corners.reshape(-1,2)
        z_mm = robust_corner_depth_mm(depth_mm, uv, half=2)
        valid = np.isfinite(z_mm) & (z_mm>0)
        if valid.sum()<10: continue

        # 카메라 좌표계에서 true depth: Xc = R*Xb + t => Zc
        Xb = objp
        Xc = (R @ Xb.T).T + tvec.reshape(1,3)
        z_true = Xc[:,2].astype(np.float32)
        z_meas = (z_mm * DEPTH_MM_TO_M).astype(np.float32)

        z_true_all.append(z_true[valid])
        z_meas_all.append(z_meas[valid])

        # Depth->Color extrinsics 정련 (depth가 color로 align되어 있다 가정)
        # 3D는 깊이로 역투영하지 않고, depth=Z로 보는 단순화 대신, solvePnP로 컬러 좌표계에서 미세 보정
        R_list.append(R); t_list.append(tvec)

    if not z_true_all:
        print("[ERROR] 유효 프레임 부족. 더 촬영 필요.")
        s, b = 1.0, 0.0
    else:
        z_true_all = np.concatenate(z_true_all)
        z_meas_all = np.concatenate(z_meas_all)
        A = np.vstack([z_true_all, np.ones_like(z_true_all)]).T
        s, b = np.linalg.lstsq(A, z_meas_all, rcond=None)[0]
        mae = float(np.mean(np.abs(s*z_true_all + b - z_meas_all)))
        print(f"[INFO] Depth scale={float(s):.6f}, bias={float(b):.6f} (m), MAE={mae:.5f} m")

    save_yaml(os.path.join(SAVE_DIR, "depth_model.yaml"), {
        'scale': float(s), 'bias': float(b),
        'equation': 'z_measured(m) ≈ scale * z_true(m) + bias'
    })

    # Depth->Color extrinsics: 다수 프레임의 R,t 평균(간단화)
    if len(R_list) > 0:
        R_avg = sum(R_list)/len(R_list)
        U,Sv,Vt = np.linalg.svd(R_avg); R_avg = U@Vt
        t_avg = sum(t_list)/len(t_list)
    else:
        R_avg = np.eye(3); t_avg = np.zeros((3,1))
    save_yaml(os.path.join(SAVE_DIR, "depth_to_color_extrinsics.yaml"), {
        'R': R_avg.tolist(),
        't': t_avg.reshape(3).tolist(),
        'note': 'If frames were align(color), this is near-identity. 필요시 별도 재추정 권장.'
    })

def main():
    ensure_dirs()
    print("=== 1) RGB-D 캡처 ===")
    record_rgbd()
    print("=== 2) Color intrinsics 보정 ===")
    K, dist = calibrate_color_intrinsics()
    print("=== 3) Depth scale/bias + Depth->Color extrinsics ===")
    estimate_depth_scale_bias_and_extrinsics(K, dist)
    print("[OK] 결과: out/*_intrinsics.yaml, depth_model.yaml, depth_to_color_extrinsics.yaml")

if __name__ == "__main__":
    main()
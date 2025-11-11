#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, json, time
import numpy as np
import cv2 as cv
import yaml
import pyrealsense2 as rs
from dataclasses import dataclass

# ========================= 사용자 설정 =========================
SAVE_DIR         = "out"
FRAME_DIR        = os.path.join(SAVE_DIR, "frames")
# 희망 해상도(실제는 장치 지원/선택된 프로필을 따름)
IMG_SIZE         = (640, 480)      # (width, height)
FPS              = 30
MIN_DET          = 500             # 최소 체커보드 검출 프레임
CHECKER_ROWS     = 6               # 내부 코너(행)
CHECKER_COLS     = 8               # 내부 코너(열)
SQUARE_SIZE_M    = 0.025           # 한 칸 길이[m] (예: 25mm -> 0.025)
ALIGN_TO_COLOR   = True            # depth->color 정렬 권장
DEPTH_MM_TO_M    = 1.0/1000.0      # RealSense z16(mm) -> m
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

# ---------------- 공용 유틸: 시작/웜업/프레임 대기 ----------------

def start_with_retry(pipe, cfg, tries=3, sleep_s=0.5):
    """
    파이프라인 시작 재시도. macOS 전원/백엔드 초기화시 간헐 실패 대응.
    """
    last = None
    for _ in range(tries):
        try:
            return pipe.start(cfg)
        except Exception as e:
            last = e
            try: pipe.stop()
            except: pass
            time.sleep(sleep_s)
    raise last

def warmup_frames(pipe, need_color, need_depth, timeout_ms=15000, warmup_count=60):
    """
    초기 몇 프레임 안정화 대기. 컬러/깊이 각각 도착 확인.
    """
    got_color = not need_color
    got_depth = not need_depth
    for _ in range(warmup_count):
        frames = pipe.wait_for_frames(timeout_ms)
        if frames:
            if need_color and frames.get_color_frame(): got_color = True
            if need_depth and frames.get_depth_frame(): got_depth = True
        if got_color and got_depth:
            return True
    return False

def wait_frames_strong(pipe, timeout_ms=10000):
    """
    메인 루프에서 좀 더 관대한 대기.
    """
    frames = pipe.wait_for_frames(timeout_ms)
    if not frames:
        frames = pipe.wait_for_frames(timeout_ms)
    return frames

# ----------------------------------------------------------------

def list_supported_video_profiles(dev):
    """
    장치 센서가 광고하는 비디오 스트림 프로파일을 수집.
    """
    have = []
    for s in dev.query_sensors():
        for p in s.get_stream_profiles():
            try:
                v = p.as_video_stream_profile()  # 비디오가 아니면 예외
                have.append((
                    v.stream_type(),  # rs.stream.color / rs.stream.depth ...
                    v.width(),
                    v.height(),
                    v.format(),       # rs.format.rgb8 / z16 / mjpeg / yuyv ...
                    v.fps()
                ))
            except Exception:
                continue
    return have

def pick_color_depth_profiles(dev):
    """
    macOS 안정화를 위해 컬러는 MJPEG 우선, 그 다음 BGR8/YUYV.
    깊이는 z16 고정. 실제 광고값과 일치하는 조합만 선택.
    """
    supported = set(list_supported_video_profiles(dev))

    color_candidates = [
        (rs.stream.color, IMG_SIZE[0], IMG_SIZE[1], rs.format.mjpeg, 30),
        (rs.stream.color, IMG_SIZE[0], IMG_SIZE[1], rs.format.mjpeg, 15),
        (rs.stream.color, IMG_SIZE[0], IMG_SIZE[1], rs.format.bgr8,  15),
        (rs.stream.color, 1280, 720,   rs.format.mjpeg, 30),
        (rs.stream.color, 1280, 720,   rs.format.mjpeg, 15),
        (rs.stream.color, 640,  360,   rs.format.mjpeg, 30),
        (rs.stream.color, 640,  360,   rs.format.bgr8,  15),
        (rs.stream.color, 640,  480,   rs.format.rgb8,  30),  # 장치 따라 가능
    ]
    depth_candidates = [
        (rs.stream.depth, 640, 480, rs.format.z16, 30),
        (rs.stream.depth, 848, 480, rs.format.z16, 30),
        (rs.stream.depth, 640, 360, rs.format.z16, 30),
    ]

    color_sel = None
    for c in color_candidates:
        if c in supported:
            color_sel = c
            break

    depth_sel = None
    for d in depth_candidates:
        if d in supported:
            depth_sel = d
            break

    return color_sel, depth_sel

def record_rgbd():
    ensure_dirs()

    # 0) 장치 탐색
    print("RS2_USB_BACKEND =", os.environ.get("RS2_USB_BACKEND"))
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense device detected at librealsense level.")
    dev = devs[0]
    serial = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)
    print("[INFO] Binding to device:", name, serial)

    # 1) depth-only로 웜업 (전력/USB 초기화 안정화)
    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_device(serial)
    # 깊이는 비교적 잘 붙는다
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = start_with_retry(pipe, cfg)
    ok = warmup_frames(pipe, need_color=False, need_depth=True,
                       timeout_ms=15000, warmup_count=60)
    if not ok:
        try: pipe.stop()
        except: pass
        raise RuntimeError("No depth frames during warmup (depth-only).")

    # depth-only 성공 → 잠깐 멈춘 뒤 color+depth 구성으로 재시작
    pipe.stop()
    time.sleep(0.3)

    # 2) 실제 캡처용 프로파일 선택
    color_sel, depth_sel = pick_color_depth_profiles(dev)
    if color_sel is None or depth_sel is None:
        # 컬러가 전혀 없거나 깊이가 전혀 없으면 실패
        raise RuntimeError("Supported profiles not found (color or depth).")

    print("[INFO] Selected color profile:", color_sel)
    print("[INFO] Selected depth  profile:", depth_sel)

    # 3) 파이프라인 재구성 (컬러+깊이)
    started = False
    last_err = None
    for fps_try in (FPS, 15):
        try:
            pipe = rs.pipeline()
            cfg  = rs.config()
            cfg.enable_device(serial)
            # 선택된 프로파일로 설정
            c = (color_sel[0], color_sel[1], color_sel[2], color_sel[3], fps_try)
            d = (depth_sel[0], depth_sel[1], depth_sel[2], depth_sel[3], depth_sel[4])
            cfg.enable_stream(*c)
            cfg.enable_stream(*d)

            profile = start_with_retry(pipe, cfg)
            # 컬러/깊이 모두 도착하는지 웜업
            ok = warmup_frames(pipe, need_color=True, need_depth=True,
                               timeout_ms=15000, warmup_count=60)
            if ok:
                started = True
                chosen_color = c
                chosen_depth = d
                break
            else:
                try: pipe.stop()
                except: pass
                time.sleep(0.2)
        except Exception as e:
            last_err = e
            try: pipe.stop()
            except: pass
            time.sleep(0.2)

    if not started:
        # 최종 폴백: depth-only로라도 진행 (캘리브레이션은 불가)
        pipe = rs.pipeline()
        cfg  = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = start_with_retry(pipe, cfg)
        ok = warmup_frames(pipe, need_color=False, need_depth=True,
                           timeout_ms=20000, warmup_count=60)
        if not ok:
            try: pipe.stop()
            except: pass
            raise RuntimeError(f"Failed to start color+depth; depth-only also failed. Last error: {last_err}")
        print("[WARN] Color stream unavailable. Proceeding with depth-only capture.")
        color_available = False
    else:
        color_available = True
        print("[INFO] Using profiles -> color:", chosen_color, ", depth:", chosen_depth)

    # 4) 정렬자
    align = rs.align(rs.stream.color) if (ALIGN_TO_COLOR and color_available) else None

    # 5) 공장 intrinsics 저장(가능한 경우)
    try:
        if color_available:
            cintr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        else:
            cintr = None
        dintr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        save_yaml(os.path.join(SAVE_DIR, "factory_intrinsics.yaml"), {
            'color': None if cintr is None else dict(w=cintr.width, h=cintr.height, fx=cintr.fx, fy=cintr.fy, cx=cintr.ppx, cy=cintr.ppy, dist=list(cintr.coeffs)),
            'depth': dict(w=dintr.width, h=dintr.height, fx=dintr.fx, fy=dintr.fy, cx=dintr.ppx, cy=dintr.ppy, dist=list(dintr.coeffs)),
            'note': 'RealSense factory intrinsics'
        })
    except Exception:
        pass

    # 현재 color 포맷 확인(있을 때만)
    color_format = None
    if color_available:
        try:
            color_stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            color_format = color_stream_profile.format()
        except Exception:
            color_format = None

    print("[INFO] Recording RGB-D... (체커보드가 보이도록 천천히 여러 거리/각도로 촬영, 'q' 종료)")
    det = 0
    idx = 0

    # macOS 안정성: IMU 비활성(필요하면 별도 파이프라인으로)
    imu_enabled = False
    imu_log = []
    last_g = [0.0, 0.0, 0.0]
    last_a = [0.0, 0.0, 0.0]

    try:
        while True:
            frames = wait_frames_strong(pipe, timeout_ms=10000)
            if not frames:
                raise RuntimeError("No frames (timeout) in main loop")

            if align is not None:
                frames = align.process(frames)

            df = frames.get_depth_frame()
            if not df:
                continue
            depth = np.asanyarray(df.get_data())  # uint16, mm

            if color_available:
                cf = frames.get_color_frame()
                if not cf:
                    # color 일시 미도착: depth만 계속 받으면서 대기
                    continue
                color = np.asanyarray(cf.get_data())
                # RGB8로 들어오면 BGR로 변환
                if color_format == rs.format.rgb8:
                    color_bgr = cv.cvtColor(color, cv.COLOR_RGB2BGR)
                else:
                    # MJPEG 디코딩 경로 등: 대부분 BGR 메모리로 전달됨
                    color_bgr = color

                # 체커보드 검출
                ok, corners = find_corners(color_bgr, SPEC)
                disp = color_bgr.copy()
                if ok and corners is not None:
                    cv.drawChessboardCorners(disp, (SPEC.cols, SPEC.rows), corners, True)

                text1 = f"idx:{idx} det_count:{det}/{MIN_DET}"
                cv.putText(disp, text1, (8,20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # depth 컬러맵(0-3m 클립)
                depth_vis = depth.copy().astype(np.float32) * DEPTH_MM_TO_M  # m
                depth_mask = (depth_vis > 0)
                if depth_mask.any():
                    clip_m = 3.0
                    dv = np.clip(depth_vis / clip_m * 255.0, 0, 255).astype(np.uint8)
                    dv[~depth_mask] = 0
                    depth_color = cv.applyColorMap(cv.equalizeHist(dv), cv.COLORMAP_JET)
                else:
                    depth_color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

                cv.imshow("color_view", disp)
                cv.imshow("depth_view", depth_color)

                ts_ns = int(frames.get_timestamp() * 1e6)
                if ok:
                    det += 1

                # 저장(BGR 이미지 저장)
                cv.imwrite(os.path.join(FRAME_DIR, f"color_{idx:06d}.png"), color_bgr)
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

            else:
                # color 불가: depth만 저장(캘리브레이션 단계는 스킵해야 함)
                ts_ns = int(frames.get_timestamp() * 1e6)
                np.save(os.path.join(FRAME_DIR, f"depth_{idx:06d}.npy"), depth)
                with open(os.path.join(FRAME_DIR, f"meta_{idx:06d}.json"), "w") as f:
                    json.dump({'t_ns': ts_ns, 'detected': False}, f)
                idx += 1
                if idx >= MIN_DET:
                    print(f"[INFO] Saved {idx} depth-only frames.")
                    break

    finally:
        try: pipe.stop()
        except: pass
        cv.destroyAllWindows()
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
        if img is None:
            continue
        if size is None:
            size = (img.shape[1], img.shape[0])
        ok, corners = find_corners(img, SPEC)
        if ok:
            objpts.append(objp.copy())
            imgpts.append(corners)
    if len(objpts) < 8:
        print("[WARN] Checkerboard detections too few. 더 촬영 필요.")
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
        if img is None:
            continue
        ok, corners = find_corners(img, SPEC)
        if not ok:
            continue

        # 보드 pose (Cam<-Board)
        okp, rvec, tvec = cv.solvePnP(objp, corners, K, dist, flags=cv.SOLVEPNP_ITERATIVE)
        if not okp:
            continue
        R,_ = cv.Rodrigues(rvec)

        depth_mm = np.load(os.path.join(FRAME_DIR, f"depth_{idx:06d}.npy"))
        uv = corners.reshape(-1,2)
        z_mm = robust_corner_depth_mm(depth_mm, uv, half=2)
        valid = np.isfinite(z_mm) & (z_mm>0)
        if valid.sum() < 10:
            continue

        # 카메라 좌표계에서 true depth: Xc = R*Xb + t => Zc
        Xb = objp
        Xc = (R @ Xb.T).T + tvec.reshape(1,3)
        z_true = Xc[:,2].astype(np.float32)
        z_meas = (z_mm * DEPTH_MM_TO_M).astype(np.float32)

        z_true_all.append(z_true[valid])
        z_meas_all.append(z_meas[valid])

        # solvePnP 결과 누적 (간단 평균으로 Depth->Color 근사)
        R_list.append(R)
        t_list.append(tvec)

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

    if len(R_list) > 0:
        R_avg = sum(R_list)/len(R_list)
        U,Sv,Vt = np.linalg.svd(R_avg)
        R_avg = U @ Vt
        t_avg = sum(t_list)/len(t_list)
    else:
        R_avg = np.eye(3)
        t_avg = np.zeros((3,1))
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
    # macOS 권장 실행:
    #   export RS2_USB_BACKEND=libusb
    #   sudo -E "$(which python)" camera_calibration.py
    main()
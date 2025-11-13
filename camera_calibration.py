import os
import csv
import time
import json
import glob
import numpy as np
import cv2 as cv
import yaml
import pyrealsense2 as rs





def obj_points(rows, cols, square_size):
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size
    return objp

def find_corners(img_bgr, rows, cols):
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (cols, rows),
                                           cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
    if not ret:
        return False, None
    term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), term)
    return True, corners

# IMU 기능 여부 확인
def device_IMU():
    
    # RealSense 장비 조회
    context = rs.context()
    if context.devices.size() == 0:
        raise RuntimeError("No device found.")
    
    # 연결된 RealSense 첫 번째 장치 선택
    device = context.devices[0]
    
    # IMU 센서 선택
    for sensor in device.sensors:
        # 센서 내부 프로파일 조회
        for profile in sensor.get_stream_profiles():

            st = profile.stream_type()
            if st in (rs.stream.gyro, rs.stream.accel):
                return True

    return False

# 호환 가능한 color/depth 프로파일 선택
def device_RGBD():
    
    # RealSense 장비 조회
    context = rs.context()
    
    if context.devices.size() == 0:
        raise RuntimeError("No device found.")
    
    # 연결된 RealSense 첫 번째 장치 선택
    device = context.devices[0]
    color_profiles = []
    depth_profiles = []

    # color 및 depth 센서 선택
    for sensor in device.sensors:
        
        # 각 센서의 프로파일 조회
        # RealSense 장치는 Stero(1,2), Depth, Color 센서가 존재
        for profile in sensor.get_stream_profiles():

            # 프로파일 정보 객체
            # ex) vp: <pyrealsense2.video_stream_profile: Color(0) 424x240 @ 6fps YUYV>
            vp = profile.as_video_stream_profile()
            
            # 넓이, 높이, fps, 포맷, 스트림 타입 추출
            stream = vp.stream_type()
            w, h, fps = vp.width(), vp.height(), vp.fps()
            if stream == rs.stream.color:
                color_profiles.append((w, h, fps))
            if stream == rs.stream.depth:
                depth_profiles.append((w, h, fps))
    
    # color와 depth 프로파일에서 공통 조합 찾기
    menu = []
    seen = []   # (w, h, fps)

    for cf in color_profiles:
        for df in depth_profiles:
            if cf[0] == df[0] and cf[1] == df[1] and cf[2] == df[2]:
                key = (cf[0], cf[1], cf[2])
                if key not in seen:
                    seen.append(key)
                    menu.append(key)

    if not menu:
        raise RuntimeError("[INFO] There is no compatible color/depth profile.")

    # 터미널에 선택지 출력
    print("Choose a compatible color/depth profile:")
    for idx, (w, h, fps) in enumerate(menu):
        print(f"\t[{idx+1}] {w}x{h} @ {fps}fps")

    # 사용자에게 번호 입력 받기

    choice = input(f"Select the profile number to use (1~{len(menu)}): ").strip()

    # 사용자가 고른 프로파일 반환
    chosen_w, chosen_h, chosen_fps = menu[int(choice) - 1]
    print(f"[INFO] Selected profile: {chosen_w}x{chosen_h} @ {chosen_fps}fps")
    return (chosen_w, chosen_h, chosen_fps)

def record(target_shots):
    
    # RealSense 카메라 데이터(Depth, Color, IMU) 처리 객체 생성
    pipe = rs.pipeline()
    
    # RealSense 설정 객체 생성
    cfg = rs.config()

    # 호환 프로필
    profile = device_RGBD()
    w,h,fps = profile

    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    print(f"[INFO] Selected profile: {w}x{h} @ {fps}")

    # 초기 공장 보정값 저장
    factory_c = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    factory_d = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    with open(os.path.join(SAVE_DIR, "factory_intrinsics.yaml"), "w") as f:
        yaml.safe_dump({
            'color': dict(w=factory_c.width, h=factory_c.height, fx=factory_c.fx, fy=factory_c.fy, cx=factory_c.ppx, cy=factory_c.ppy, dist=list(factory_c.coeffs)),
            'depth': dict(w=factory_d.width, h=factory_d.height, fx=factory_d.fx, fy=factory_d.fy, cx=factory_d.ppx, cy=factory_d.ppy, dist=list(factory_d.coeffs))
        }, f, sort_keys=False, allow_unicode=True)
    print("[INFO] factory_intrinsics.yaml saved")

    # IMU 탑재 여부
    imu_enabled = False
    if device_IMU():
        cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f)
        cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
        imu_enabled = True
        print("[INFO] IMU enabled")

    else:
        print("[INFO] IMU disabled")

    # IMU가 활성화된 경우: IMU + RGBD 동시에 설정된 cfg로 파이프라인 시작
    # IMU가 비활성화된 경우: RGBD만 포함된 cfg로 파이프라인 시작
    profile = pipe.start(cfg)
    print("[INFO] pipeline started (IMU enabled: {})".format(imu_enabled))

    # imu_log: list of [t_ns, gx,gy,gz, ax,ay,az]
    imu_log = []
    gyro = [0.0, 0.0, 0.0]
    accel = [0.0, 0.0, 0.0]

    print(f"[INFO] Capture: press Enter/Space/'c' to save, 'q' to quit. Target: {target_shots} shots")
    
    shots = 0
    idx = 0

    # 캡쳐 시작
    while shots < target_shots:
        frames = pipe.wait_for_frames()
        
        # depth 프레임을 color에 정렬
        if ALIGN_TO_COLOR:
            align = rs.align(rs.stream.color) 
            frames = align.process(frames)

        # IMU 프레임 읽기
        if imu_enabled:
            gf = frames.first_or_default(rs.stream.gyro)
            af = frames.first_or_default(rs.stream.accel)

            if gf and af:
                t_ns = int(frames.get_timestamp() * 1e6)
                
                g = gf.as_motion_frame().get_motion_data()
                a = af.as_motion_frame().get_motion_data()


                gyro = [g.x, g.y, g.z]
                accel = [a.x, a.y, a.z]
                imu_log.append([t_ns, gyro[0], gyro[1], gyro[2], accel[0], accel[1], accel[2]])

        cf = frames.get_color_frame()
        df = frames.get_depth_frame()

        if not cf or not df:
            continue

        color = np.asanyarray(cf.get_data())
        depth = np.asanyarray(df.get_data())

        ok, corners = find_corners(color, CHECKER_ROWS, CHECKER_COLS)
        disp = color.copy()
        if ok:
            cv.drawChessboardCorners(disp, (CHECKER_COLS, CHECKER_ROWS), corners, True)
        cv.putText(disp, f"shots: {shots}/{target_shots}", (8,20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv.imshow("color", disp)
        depth_vis = (depth.astype(np.float32) * (1.0/1000.0))
        dv = np.clip(depth_vis/3.0*255.0, 0, 255).astype(np.uint8)
        dv[depth == 0] = 0
        cv.imshow("depth", cv.applyColorMap(dv, cv.COLORMAP_JET))

        key = cv.waitKey(1) & 0xFF
        if key in (10, 13, 32, ord('c')):  # Enter/Space/c
            t_ns = int(frames.get_timestamp() * 1e6)
            cv.imwrite(os.path.join(FRAME_DIR, f"color_{idx:06d}.png"), color)
            np.save(os.path.join(FRAME_DIR, f"depth_{idx:06d}.npy"), depth)
            with open(os.path.join(FRAME_DIR, f"meta_{idx:06d}.json"), "w") as f:
                json.dump({'t_ns': t_ns, 'detected': bool(ok)}, f)
            shots += 1
            idx += 1
            print(f"[SAVE] {shots}/{target_shots} saved (detected={ok})")
        elif key == ord('q'):
            print("[INFO] 사용자 중단")
            break

    # 루프 종료 후 자원 정리 및 IMU 로그 저장
    pipe.stop()
    cv.destroyAllWindows()

    if imu_enabled and len(imu_log) > 0:
        # User requested raw write: save imu_log rows in appended order without merging.
        imu_path = os.path.join(FRAME_DIR, "imu_log.csv")
        with open(imu_path, "w", newline='') as f:
            w = csv.writer(f)
            w.writerow(['t_ns', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])
            w.writerows(imu_log)
        print(f"[INFO] Saved IMU log: {imu_path}")

def calibrate_color_intrinsics():
    objp = obj_points(CHECKER_ROWS, CHECKER_COLS, SQUARE_SIZE_M)
    objpts = []
    imgpts = []
    size = None
    files = sorted(glob.glob(os.path.join(FRAME_DIR, "color_*.png")))
    for p in files:
        img = cv.imread(p)
        if img is None:
            continue
        if size is None:
            size = (img.shape[1], img.shape[0])
        ok, corners = find_corners(img, CHECKER_ROWS, CHECKER_COLS)
        if ok:
            objpts.append(objp.copy())
            imgpts.append(corners)
    if len(objpts) < 6:
        print("[ERROR] 유효한 캡처 이미지가 부족합니다. 더 촬영하세요.")
        return None, None
    flags = cv.CALIB_RATIONAL_MODEL
    rms, K, dist, rvecs, tvecs = cv.calibrateCamera(objpts, imgpts, size, None, None, flags=flags)
    print(f"[INFO] 보정 완료 RMS={rms:.4f}")
    with open(os.path.join(SAVE_DIR, "color_intrinsics.yaml"), "w") as f:
        yaml.safe_dump({
            'image_size': {'w': size[0], 'h': size[1]},
            'K': K.tolist(),
            'dist': dist.flatten().tolist(),
            'rms': float(rms),
            'checkerboard': {'rows': CHECKER_ROWS, 'cols': CHECKER_COLS, 'square_m': SQUARE_SIZE_M}
        }, f, sort_keys=False, allow_unicode=True)
    return K, dist





if __name__ == "__main__":
    
    # 결과 저장 경로
    SAVE_DIR = "intrinsics_out"
    FRAME_DIR = os.path.join(SAVE_DIR, "frames")

    # 캡쳐 개수 설정
    TARGET_SHOTS = 30

    # 체크보드 크기(정사각형 사이즈 단위는 m)
    CHECKER_ROWS = 6
    CHECKER_COLS = 8
    SQUARE_SIZE_M = 0.025

    # depth 프레임을 color에 정렬할지 여부
    # RGB-D 카메라에서 depth와 color의 물리적인 측정 차이를 align 처리
    # color 이미지의 각 픽셀에 해당하는 depth 값을 가져오도록 변환
    ALIGN_TO_COLOR = True

    # 저장 폴더 생성
    os.makedirs(FRAME_DIR, exist_ok=True)
    
    # 캡처 및 보정 수행
    record(TARGET_SHOTS)
    K, dist = calibrate_color_intrinsics()
    if K is not None:
        print("[OK] Intrinsics saved")
    else:
        print("[ERROR] Intrinsics calibration failed")
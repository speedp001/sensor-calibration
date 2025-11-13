import os
import time
import json
import glob
import numpy as np
import cv2 as cv
import yaml
import pyrealsense2 as rs





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





def save_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

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

def device_has_motion():
    try:
        ctx = rs.context()
        if len(ctx.devices) == 0:
            return False
        dev = ctx.devices[0]
        for s in dev.sensors:
            try:
                # try checking stream profiles for motion types
                for p in s.get_stream_profiles():
                    try:
                        st = p.stream_type()
                        if st in (rs.stream.gyro, rs.stream.accel):
                            return True
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    return False

# 호환 가능한 color, depth 프로파일 선택
def compatible_profile():
    
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
            format = vp.format()
            w, h, fps = vp.width(), vp.height(), vp.fps()
            if stream == rs.stream.color:
                color_profiles.append((format, w, h, fps))
            if stream == rs.stream.depth:
                depth_profiles.append((format, w, h, fps))
    
    # color와 depth 프로파일에서 공통 조합 찾기
    menu = []
    seen = []   # (fmt, w, h, fps)를 해시로 추적

    for cf in color_profiles:
        for df in depth_profiles:
            if cf[1] == df[1] and cf[2] == df[2] and cf[3] == df[3]:
                key = (cf[0], cf[1], cf[2], cf[3])
                if key not in seen:
                    seen.append(key)
                    menu.append(key)

    if not menu:
        raise RuntimeError("[INFO] There is no compatible color&depth profile.")

    # 터미널에 선택지 출력
    print("Choose a compatible color&depth profile:")
    for idx, (fmt, w, h, fps) in enumerate(menu):
        print(f"\t[{idx}] {w}x{h} @ {fps}fps, format={fmt}")

    # 사용자에게 번호 입력 받기
    default_idx = 0
    choice = input(f"사용할 프로파일 번호를 선택하세요 (0~{len(menu)-1}, 기본={default_idx}): ").strip()

    chosen_fmt, chosen_w, chosen_h, chosen_fps = menu[choice]
    print(f"[INFO] 선택된 프로파일: {chosen_w}x{chosen_h} @ {chosen_fps}fps, format={chosen_fmt}")
    return (chosen_fmt, chosen_w, chosen_h, chosen_fps)

def record(target_shots=TARGET_SHOTS):
    
    # RealSense 카메라 데이터(Depth, Color, IMU) 처리 객체 생성
    pipe = rs.pipeline()
    # RealSense 설정 객체 생성
    cfg = rs.config()

    profile = compatible_profile()
    if profile:
        fmt,w,h,fps = profile
        try:
            cfg.enable_stream(rs.stream.color, w, h, fmt, fps)
            cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            print(f"[INFO] 선택된 프로필: color {w}x{h}@{fps}")
        except Exception as e:
            print("[WARN] 선택 프로필 활성화 실패:", e)
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color)
            cfg.enable_stream(rs.stream.depth)
            print("[INFO] 디폴트 프로필로 폴백")
    else:
        cfg.enable_stream(rs.stream.color)
        cfg.enable_stream(rs.stream.depth)
        print("[WARN] 호환 프로필 찾지 못함. 디폴트 사용")

    # IMU 활성화는 장치 지원 여부 확인 후 시도
    imu_enabled = False
    if device_has_motion():
        try:
            cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f)
            cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
            imu_enabled = True
            print("[INFO] IMU 스트림 활성화 시도")
        except Exception as e:
            imu_enabled = False
            print("[WARN] IMU 스트림 활성화 실패, IMU 없이 진행:", e)
    else:
        print("[INFO] 장치에 IMU 스트림 없음 -> IMU 비활성")

    # start pipeline: IMU 포함으로 실패하면 IMU 제거하고 재시도
    profile = None
    started = False
    for attempt in range(2 if imu_enabled else 3):
        try:
            profile = pipe.start(cfg)
            started = True
            break
        except Exception as e:
            print(f"[WARN] pipe.start 실패: {e} (attempt {attempt+1})")
            time.sleep(0.7)
            if imu_enabled:
                # 재구성: IMU 제거 후 재시도
                print("[INFO] IMU 관련 문제로 보이므로 IMU 스트림 제거 후 재시도합니다.")
                cfg = rs.config()
                if profile:
                    try:
                        cfg.enable_stream(rs.stream.color, w, h, fmt, fps)
                        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
                    except Exception:
                        cfg.enable_stream(rs.stream.color)
                        cfg.enable_stream(rs.stream.depth)
                else:
                    cfg.enable_stream(rs.stream.color)
                    cfg.enable_stream(rs.stream.depth)
                imu_enabled = False
    if not started:
        raise RuntimeError("파이프라인 시작 실패")

    align = rs.align(rs.stream.color) if ALIGN_TO_COLOR else None

    # save factory intrinsics if available
    try:
        cin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        din = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        save_yaml(os.path.join(SAVE_DIR, "factory_intrinsics.yaml"), {
            'color': dict(w=cin.width, h=cin.height, fx=cin.fx, fy=cin.fy, cx=cin.ppx, cy=cin.ppy, dist=list(cin.coeffs)),
            'depth': dict(w=din.width, h=din.height, fx=din.fx, fy=din.fy, cx=din.ppx, cy=din.ppy, dist=list(din.coeffs))
        })
        print("[INFO] factory_intrinsics.yaml 저장")
    except Exception:
        pass

    shots = 0
    idx = 0

    # IMU 로그: list of [t_ns, gx,gy,gz, ax,ay,az]
    imu_log = []
    last_g = [0.0,0.0,0.0]
    last_a = [0.0,0.0,0.0]

    print(f"[INFO] 캡처: Enter/Space/'c' 저장, 'q' 종료. 목표 {target_shots}장")
    try:
        while shots < target_shots:
            frames = pipe.wait_for_frames()  # blocking
            if ALIGN_TO_COLOR:
                try:
                    frames = align.process(frames)
                except Exception:
                    pass

            # IMU 프레임 읽기(있으면 즉시 로그 추가)
            if imu_enabled:
                try:
                    gf = frames.first_or_default(rs.stream.gyro)
                    af = frames.first_or_default(rs.stream.accel)
                except Exception:
                    gf = None; af = None
                if gf:
                    try:
                        g = gf.as_motion_frame().get_motion_data()
                        t_ns = int(gf.get_timestamp() * 1e6)
                        last_g = [g.x, g.y, g.z]
                        imu_log.append([t_ns, last_g[0], last_g[1], last_g[2], last_a[0], last_a[1], last_a[2]])
                    except Exception:
                        pass
                if af:
                    try:
                        a = af.as_motion_frame().get_motion_data()
                        t_ns = int(af.get_timestamp() * 1e6)
                        last_a = [a.x, a.y, a.z]
                        imu_log.append([t_ns, last_g[0], last_g[1], last_g[2], last_a[0], last_a[1], last_a[2]])
                    except Exception:
                        pass

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
            cv.putText(disp, f"shots: {shots}/{target_shots}", (8,20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv.imshow("color", disp)
            depth_vis = (depth.astype(np.float32) * (1.0/1000.0))
            dv = np.clip(depth_vis/3.0*255.0, 0, 255).astype(np.uint8)
            dv[depth==0] = 0
            cv.imshow("depth", cv.applyColorMap(dv, cv.COLORMAP_JET))

            key = cv.waitKey(1) & 0xFF
            if key in (10,13,32, ord('c')):  # Enter/Space/c
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
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
        cv.destroyAllWindows()

        # IMU 로그 병합 및 저장 (t_ns,gx,gy,gz,ax,ay,az)
        if imu_enabled and len(imu_log) > 0:
            merged = {}
            for row in imu_log:
                t = int(row[0])
                gx,gy,gz,ax,ay,az = row[1],row[2],row[3],row[4],row[5],row[6]
                if t not in merged:
                    merged[t] = [t, 0.0,0.0,0.0, 0.0,0.0,0.0]
                # prefer non-zero values
                if not (abs(gx) < 1e-12 and abs(gy) < 1e-12 and abs(gz) < 1e-12):
                    merged[t][1]=gx; merged[t][2]=gy; merged[t][3]=gz
                if not (abs(ax) < 1e-12 and abs(ay) < 1e-12 and abs(az) < 1e-12):
                    merged[t][4]=ax; merged[t][5]=ay; merged[t][6]=az
            rows = sorted(merged.values(), key=lambda x: x[0])
            import csv
            imu_path = os.path.join(FRAME_DIR, "imu_log.csv")
            with open(imu_path, "w", newline='') as f:
                w = csv.writer(f)
                w.writerow(['t_ns', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])
                w.writerows(rows)
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
    save_yaml(os.path.join(SAVE_DIR, "color_intrinsics.yaml"), {
        'image_size': {'w': size[0], 'h': size[1]},
        'K': K.tolist(),
        'dist': dist.flatten().tolist(),
        'rms': float(rms),
        'checkerboard': {'rows': CHECKER_ROWS, 'cols': CHECKER_COLS, 'square_m': SQUARE_SIZE_M}
    })
    return K, dist





if __name__ == "__main__":
    os.makedirs(FRAME_DIR, exist_ok=True)
    record(TARGET_SHOTS)
    K, dist = calibrate_color_intrinsics()
    if K is not None:
        print("[OK] color_intrinsics saved in out/")
    else:
        print("[ERR] 보정 실패")
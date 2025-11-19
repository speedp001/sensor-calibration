import os
import csv
import glob
import json
import yaml
import cv2 as cv
import numpy as np





# csv로드 함수
def load_csv(path, cols=4):

    rows = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        # 첫 줄은 건너뜀
        _ = next(reader, None)
        
        for r in reader:
            # 빈 줄은 건너뜀
            if not r:
                continue
            
            # 열 개수가 맞지 않으면 건너뜀
            if len(r) != cols:
                continue

            rows.append(r)

    data = np.array(rows, dtype=np.float64)

    # ts, x, y, z 성분이 누락된 경우
    if data.shape[1] != cols:
        raise ValueError(f"Unexpected csv shape {data.shape}, expected Nx{cols}")
    
    return data



# gyro / accel 로그 병합 함수
def merge_gyro_accel(gyro, accel):
    """
    Inputs:
    gyro: (N,4) [t_ns,gx,gy,gz]
    accel: (N,4) [t_ns,ax,ay,az]
    
    Outputs:
    imu (N,7) [t_ns,gx,gy,gz,ax,ay,az]
    """

    t_gyro = gyro[:, 0]
    t_accel = accel[:, 0]

    imu_rows = []
    for i in range(len(gyro)):
        tg = t_gyro[i]
        gx, gy, gz = gyro[i, 1:4]

        # 타임스탬프 값이 가장 가까운 accel 인덱스 검색
        idx = np.searchsorted(t_accel, tg)
        
        # t_accel이 t_gyro 처음보다 이른 경우
        if idx == 0:
            nearest_idx = 0
            
        # t_accel이 t_gyro  마지막보다 늦은 경우
        elif idx >= len(t_accel):
            nearest_idx = len(t_accel) - 1
            
        # 사이에 있는 경우우
        else:
            # 앞/뒤 중 더 가까운 것을 선택
            before = idx - 1
            after = idx
            if abs(t_accel[before] - tg) <= abs(t_accel[after] - tg):
                nearest_idx = before
            else:
                nearest_idx = after

        ta = t_accel[nearest_idx]
        ax, ay, az = accel[nearest_idx, 1:4]

        imu_rows.append([tg, gx, gy, gz, ax, ay, az])

    if len(imu_rows) == 0:
        raise RuntimeError("There are no merged IMU samples.")

    imu = np.array(imu_rows, dtype=np.float64)
    
    return imu





# 해당 이미지에서의 카메라 포즈 
def load_camera_poses(frame_dir):
    """
    Outputs:
    [{'idx', 't_ns', 'rvec', 'tvec'}, ...]
    [{index, timestamp(ns), 회전벡터, 이동벡터}, ...]
    """

    poses = []

    # 체커보드 3D 좌표 생성
    corner_3d = np.zeros((CHECKER_ROWS * CHECKER_COLS, 3), np.float32)
    
    # 체커보드 2D 그리드 생성
    corner_2d = np.mgrid[0:CHECKER_COLS, 0:CHECKER_ROWS].T.reshape(-1, 2)
    
    # 체커보드 3D 좌표 설정
    corner_3d[:, :2] = corner_2d * SQUARE_SIZE_M

    # K, dist 초기화
    # K: Intrinsic matrix
    # dist: 왜곡 계수
    K = None
    dist = None

    color_files = sorted(glob.glob(os.path.join(frame_dir, "color_*.png")))

    for color_file in color_files:
        # 인덱스 번호
        idx = int(os.path.splitext(os.path.basename(color_file))[0].split("_")[-1])
        
        # 이미지 로드
        img = cv.imread(color_file)

        # 해당 이미지에 해당하는 meta 파일 로드
        meta_file = os.path.join(frame_dir, f"meta_{idx:06d}.json")

        with open(meta_file, "r") as f:
            meta = json.load(f)
        t_ns = int(meta.get("t_ns"))

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        find, corners = cv.findChessboardCorners(gray, (CHECKER_COLS, CHECKER_ROWS))
        
        # 체커보드 코너를 찾지 못한 경우 건너뜀
        if not find:
            continue

        # 초기 코너 검출의 낮은 정밀도를 보완하기 위해 서브픽셀 활용
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

        # K, dist 설정
        if K is None:
            # 직접 계산한 calculated_intrinsics.yaml 사용
            calc_path = os.path.join(SAVE_DIR, "calculated_intrinsics.yaml")
            if os.path.exists(calc_path):
                with open(calc_path, "r") as f:
                    d = yaml.safe_load(f)
                c = d.get("color", d)
                fx = c.get("fx")
                fy = c.get("fy")
                cx = c.get("cx")
                cy = c.get("cy")
                
                if fx is not None and fy is not None and cx is not None and cy is not None:
                    K = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0,  0,  1]], dtype=np.float64)
                    
                    # 왜곡 계수는 5차 다항식까지만 고려
                    # reshape의 이유는 OpenCV 함수가 (N,1) 형태를 요구
                    dist = np.array(c.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1, 1)

            # 공장 초기값 factory_intrinsics.yaml 사용
            factory_path = os.path.join(SAVE_DIR, "factory_intrinsics.yaml")
            if os.path.exists(factory_path):
                with open(factory_path, "r") as f:
                    d = yaml.safe_load(f)
                c = d.get("color", d)
                fx = c.get("fx")
                fy = c.get("fy")
                cx = c.get("cx")
                cy = c.get("cy")
                
                if fx is not None and fy is not None and cx is not None and cy is not None:
                    K = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]], dtype=np.float64)
                    dist = np.array(c.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1, 1)

        # PnP를 사용하여 카메라 포즈 추정 
        ret, rvec, tvec = cv.solvePnP(corner_3d, corners, K, dist, flags=cv.SOLVEPNP_ITERATIVE)
        
        # 포즈 추정 실패 시 건너뜀
        if not ret:
            continue

        poses.append({
            "idx": idx,
            "t_ns": t_ns,
            "rvec": rvec.flatten(),
            "tvec": tvec.flatten()
        })

    return poses





# 중력 벡터 프레임 별 매칭 함수
def gravity_pairs(imu, poses, gyro_thresh=0.1):
    """
    Inputs:
    imu(N,7): [t_ns, gx, gy, gz, ax, ay, az]
    poses: load_camera_poses() 결과 리스트
    gyro_thresh: 정지 구간 기준 임계치

    Outputs:
    카메라 좌표계 중력 벡터
    gravity_cam_list: [array(3,), ...]
    # IMU 좌표계 중력 벡터
    gravity_imu_list: [array(3,), ...]
    
    각 프레임 별로 쌍으로 묶인다.
    """

    t_ns   = imu[:, 0]
    gyro    = imu[:, 1:4]
    accel   = imu[:, 4:7]

    # 체커보드 좌표계에서의 중력 방향(아래쪽)이라고 가정하는 축
    board_y = np.array([0.0, 1.0, 0.0])

    gravity_cam_list = []
    gravity_imu_list = []

    # gyro 크기가 담긴 리스트
    gyro_norm = np.linalg.norm(gyro, axis=1)

    # gyro_norm 순회
    for i, is_static in enumerate(gyro_norm < gyro_thresh):
        
        # 정지 구간이 아니면 건너뜀
        if not is_static:
            continue

        # IMU 프레임 기준 중력 방향 (단위 벡터)
        g_imu = accel[i] / (np.linalg.norm(accel[i]) + 1e-12)
        
        # poses 리스트에서 각 포즈의 t_ns
        poses_ts = []
        for pose in poses:
            poses_ts.append(int(pose["t_ns"]))

        pose_ts = np.array(poses_ts, dtype=np.int64)

        # target_ts 이상이 처음 나오는 위치 찾기
        idx = np.searchsorted(pose_ts, t_ns[i])

        # 맨 앞
        if idx == 0:
            nearest = 0
        
        # 맨 뒤
        elif idx >= len(pose_ts):
            nearest = len(pose_ts) - 1
            
        # 중간
        else:
            before = idx - 1
            after  = idx
            if abs(pose_ts[before] - t_ns[i]) <= abs(pose_ts[after] - t_ns[i]):
                nearest = before
            else:
                nearest = after
        
        p = poses[nearest]
        rvec = p["rvec"].reshape(3, 1)
        R, _ = cv.Rodrigues(rvec)  # 보드 -> 카메라

        # 카메라 프레임 기준 중력 방향 (보드 +Y를 중력 방향으로 가정)
        g_cam = R @ board_y
        g_cam = g_cam.flatten()
        g_cam /= (np.linalg.norm(g_cam) + 1e-12)

        gravity_cam_list.append(g_cam)
        gravity_imu_list.append(g_imu)

    return gravity_cam_list, gravity_imu_list




























# ------------------------------------------------------------------
# gyro 적분으로 회전 벡터(축 * 각도)를 추출         
# ------------------------------------------------------------------
def integrate_gyro(imu, t0_ns, t1_ns):
    """
    t0_ns ~ t1_ns 구간의 gyro를 적분하여 회전 벡터(axis * angle)를 반환한다.
    imu: ndarray Nx7 (t_ns,gx,gy,gz,ax,ay,az)
    """
    if imu.size == 0:
        return None

    sel = (imu[:, 0] >= t0_ns) & (imu[:, 0] <= t1_ns)
    s = imu[sel]
    if s.shape[0] == 0:
        return None
    if s.shape[0] < 2:
        return None

    ts = s[:, 0].astype(np.float64) * 1e-9  # ns → s
    ws = s[:, 1:4].astype(np.float64)       # [gx,gy,gz] (rad/s 가정)

    rot = np.zeros(3, dtype=np.float64)

    # 단순 trapz 적분 (작은 각도 근사)
    for i in range(len(ws) - 1):
        dt = ts[i + 1] - ts[i]
        if dt <= 0:
            continue
        w_avg = 0.5 * (ws[i] + ws[i + 1])
        dtheta = w_avg * dt
        rot += dtheta

    return rot  # (3,) axis*angle vector


# ------------------------------------------------------------------
# 카메라 회전 표현 변환 (Rodrigues <-> R)
# ------------------------------------------------------------------
def rotmat_to_rotvec(R):
    """
    회전행렬 R(3x3)을 axis-angle 벡터(3,)로 변환.
    """
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))
    if abs(theta) < 1e-8:
        return np.zeros(3, dtype=np.float64)
    rx = (1.0 / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2],
         R[0, 2] - R[2, 0],
         R[1, 0] - R[0, 1]]
    )
    return rx * theta


# ------------------------------------------------------------------
# extrinsic 회전 추정 (SVD / Kabsch)
#   imu_pairs: IMU 기준 델타 회전 (N,3)
#   cam_pairs: 카메라 기준 델타 회전 (N,3)
#   여러 개의 중력 방향 쌍(gravity_cam_list, gravity_imu_list)을 포함하여 추정
# ------------------------------------------------------------------
def estimate_rotation_extrinsic(camera_pairs, imu_pairs,
                                gravity_cam_list=None, gravity_imu_list=None, gravity_weight=5):
    """
    delta 회전 쌍(cam_pairs, imu_pairs)에 더해,
    여러 개의 중력 방향 쌍(gravity_cam_list, gravity_imu_list)을 포함하여
    extrinsic 회전 R (IMU->CAM)을 Kabsch로 추정한다.

    gravity_cam_list: [(3,), ...] 카메라 프레임 중력 방향들
    gravity_imu_list: [(3,), ...] IMU 프레임 중력 방향들
    gravity_weight: 각 중력 쌍을 몇 번 반복해서(가중치처럼) 포함할지.
    """
    A_list = list(imu_pairs)
    B_list = list(camera_pairs)

    if gravity_cam_list is not None and gravity_imu_list is not None:
        for g_cam, g_imu in zip(gravity_cam_list, gravity_imu_list):
            g_cam_n = g_cam / (np.linalg.norm(g_cam) + 1e-12)
            g_imu_n = g_imu / (np.linalg.norm(g_imu) + 1e-12)
            for _ in range(gravity_weight):
                A_list.append(g_imu_n)
                B_list.append(g_cam_n)

    A = np.stack(A_list, axis=0)
    B = np.stack(B_list, axis=0)

    A0 = A - A.mean(axis=0)
    B0 = B - B.mean(axis=0)

    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R





# ------------------------------------------------------------------
# 연속 프레임 간 카메라 델타 회전 + 동일 시간 구간의 IMU 델타 회전 쌍 만들기
#   - ts_offset_ns: 카메라 타임스탬프에 더해줄 IMU 타임 오프셋(ns).
#                   두 센서 시계가 어긋난 경우 보정용으로 사용한다.
#                   현재는 0으로 두고, 시계가 이미 동기라고 가정한다.
#   - min_angle_deg: 카메라 델타 회전 각도가 이 값보다 작으면 쌍에서 제외.
#                    (너무 작은 회전은 노이즈에 민감하므로 keyframe처럼 필터링)
# ------------------------------------------------------------------
def build_delta_pairs(poses, imu, gap=3, min_angle_deg=1.0):
    """연속된 포즈 사이의 델타 회전 쌍(cam, imu)을 만든다.

    poses: [{'t_ns', 'rvec', 'tvec'}, ...]
    imu:   (N,7) [t_ns,gx,gy,gz,ax,ay,az]

    반환:
      cam_pairs: [ (3,), ... ]  # 카메라 기준 델타 회전 벡터들
      imu_pairs: [ (3,), ... ]  # 같은 구간 IMU 적분 델타 회전 벡터들
    """
    cam_pairs = []
    imu_pairs = []

    for i in range(len(poses) - 1):
        p0 = poses[i]
        p1 = poses[i + 1]

        # 타임스탬프 범위
        t0 = p0["t_ns"]
        t1 = p1["t_ns"]

        # 너무 긴 구간은 제외 (나노 초 단위)
        if (t1 - t0) * 1e-9 > gap:
            continue

        R0, _ = cv.Rodrigues(p0["rvec"].reshape(3, 1))
        R1, _ = cv.Rodrigues(p1["rvec"].reshape(3, 1))
        
        # 카메라 좌표계에서의 상대 회전
        Rc = R0.T @ R1
        rc_vec = rotmat_to_rotvec(Rc)

        # 회전각이 너무 작으면(거의 안 움직인 구간) 노이즈에 민감하므로 skip
        angle_rad = np.linalg.norm(rc_vec)
        angle_deg = np.degrees(angle_rad)
        if angle_deg < min_angle_deg:
            continue

        imu_rot = integrate_gyro(imu, t0, t1)
        if imu_rot is None:
            continue

        cam_pairs.append(rc_vec)
        imu_pairs.append(imu_rot)

    return cam_pairs, imu_pairs






# 타임 오프셋 grid search + extrinsic R 추정
## 타임 오프셋 grid search + extrinsic R 추정 함수는 사용하지 않으므로 제거했습니다.





# main 함수
if __name__ == "__main__":
    
    # 저장 경로 설정
    SAVE_DIR = "intrinsics_out"
    FRAME_DIR = os.path.join(SAVE_DIR, "frames")

    # 체커보드 크기
    CHECKER_ROWS = 6
    CHECKER_COLS = 8
    SQUARE_SIZE_M = 0.025

    # gyro/accel log 경로
    gyro_path = os.path.join(SAVE_DIR, "gyro_log.csv")
    accel_path = os.path.join(SAVE_DIR, "accel_log.csv")

    # 경로 출력
    # print(f"[INFO] gyro_log:  {gyro_path}")
    # print(f"[INFO] accel_log: {accel_path}")

    gyro = load_csv(gyro_path, cols=4)
    accel = load_csv(accel_path, cols=4)

    imu = merge_gyro_accel(gyro, accel)
    
    # # 병합 imu log 출력
    # print(f"[INFO] merged IMU samples: {imu.shape[0]}")

    # 해당 이미지의 카메라 포즈 
    poses = load_camera_poses(FRAME_DIR)

    # (IMU 중력 벡터, 카메라 좌표계 중력 벡터) 추출
    gravity_cam_list, gravity_imu_list = gravity_pairs(
        imu,
        poses,
        gyro_thresh=0.1,    # 정지 기준
    )
    
    if len(gravity_cam_list) == 0 or len(gravity_imu_list) == 0:
        gravity_cam_list = None
        gravity_imu_list = None

    # gyro 적분으로 얻은 델타 회전 + 중력 방향 제약 사용
    # 체커보드 특징점을 활용한 카메라 회전 행렬 추정
    # 이 두 값을 근사시켜 IMU 변환 회전 행렬을 구한다.
    cam_pairs, imu_pairs = build_delta_pairs(poses, imu)

    if len(cam_pairs) < 8:
        raise RuntimeError("[Error] Not enough matching rotation pairs to estimate IMU extrinsics.")

    R = estimate_rotation_extrinsic(
        cam_pairs,
        imu_pairs,
        gravity_cam_list=gravity_cam_list,
        gravity_imu_list=gravity_imu_list,
        gravity_weight=5,
    )

    # IMU extrinsics를 YAML로 저장
    out_dict = {
        "imu_extrinsics": {
            "description": "Rotation from IMU frame to Camera frame (R_imu_to_cam)",
            "R_imu_to_cam": R.tolist(),   # 3x3 회전 행렬
            "t_imu_to_cam_m": [0.0, 0.0, 0.0]
        }
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, "IMU_extrinsics.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(out_dict, f, sort_keys=False, allow_unicode=True)

    # --------------------------------------------------------------
    # factory에서 제공하는 IMU->Camera extrinsics가 있으면 별도 YAML로 저장
    #   기대 파일: intrinsics_out/factory_intrinsics.yaml
    #   예상 구조 예시:
    #   imu_extrinsics:
    #     R_imu_to_cam: [[...],[...],[...]]
    #     t_imu_to_cam_m: [x,y,z]
    # --------------------------------------------------------------
    factory_intr_path = os.path.join(SAVE_DIR, "factory_intrinsics.yaml")
    if os.path.exists(factory_intr_path):
        try:
            with open(factory_intr_path, "r") as f:
                factory_data = yaml.safe_load(f)

            # 1) 최상단에 바로 imu_extrinsics가 있는 경우
            imu_ext = factory_data.get("imu_extrinsics")

            # 2) 없으면, "imu" 또는 "motion" 같은 키 아래에 있을 수도 있으니 한번 더 찾아봄
            if imu_ext is None:
                for key in ["imu", "motion", "imu_to_color", "imu_to_cam"]:
                    if key in factory_data and isinstance(factory_data[key], dict):
                        cand = factory_data[key].get("imu_extrinsics")
                        if cand is not None:
                            imu_ext = cand
                            break

            if imu_ext is not None:
                R_factory = imu_ext.get("R_imu_to_cam")
                t_factory = imu_ext.get("t_imu_to_cam_m", [0.0, 0.0, 0.0])

                if R_factory is not None:
                    factory_out = {
                        "imu_extrinsics_factory": {
                            "description": "Factory IMU to Camera extrinsics (R_imu_to_cam)",
                            "R_imu_to_cam": R_factory,
                            "t_imu_to_cam_m": t_factory,
                        }
                    }

                    factory_out_path = os.path.join(SAVE_DIR, "IMU_extrinsics_factory.yaml")
                    with open(factory_out_path, "w") as f:
                        yaml.safe_dump(factory_out, f, sort_keys=False, allow_unicode=True)

                    print(f"[OK] Factory IMU extrinsics saved to {factory_out_path}")
            else:
                print("[INFO] No imu_extrinsics entry found in factory_intrinsics.yaml")
        except Exception as e:
            print(f"[WARN] Failed to load factory IMU extrinsics: {e}")
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
    gravity_cam_list: [(3,), ...]
    # IMU 좌표계 중력 벡터
    gravity_imu_list: [(3,), ...]
    
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



# 회전 벡터 프레임 별 매칭 함수
def rotation_pairs(poses, imu, gap, min_angle):
    """
    Inputs:
    poses: [{'t_ns', 'rvec', 'tvec'}, ...]
    imu:   (N,7) [t_ns,gx,gy,gz,ax,ay,az]

    Outputs:
    cam_pairs: [ (3,), ... ]  # 카메라 기준 델타 회전 벡터들
    imu_pairs: [ (3,), ... ]  # 같은 구간 IMU 적분 델타 회전 벡터들
    """
    rotation_cam_list = []
    rotation_imu_list = []

    # 카메라 프레임 별 조회
    for i in range(len(poses) - 1):
        p0 = poses[i]
        p1 = poses[i + 1]

        # 타임스탬프 범위
        t0 = p0["t_ns"]
        t1 = p1["t_ns"]

        # 너무 긴 구간은 제외 (ns->s 단위)
        if (t1 - t0) * 1e-9 > gap:
            continue

        # 회전 행렬 -> 회전 벡터 로드리게스 변환
        R0, _ = cv.Rodrigues(p0["rvec"].reshape(3, 1))
        R1, _ = cv.Rodrigues(p1["rvec"].reshape(3, 1))

        """
        상대 회전 Rc 설명

        카메라0, 카메라1 포즈를 월드 좌표계에서 표현하면:
            p_cam0 = R0 @ p_world
            p_cam1 = R1 @ p_world

        우리는 카메라0 좌표계 기준에서 카메라1의 상대 회전을 원하므로:
            p_cam1 = Rc @ p_cam0
        가 되도록 Rc를 정의하고 싶다.

        첫 번째 식에서 p_world를 풀면:
            p_world = R0.T @ p_cam0   (회전행렬이라 R0^{-1} = R0.T)

        이를 두 번째 식에 대입하면:
            p_cam1 = R1 @ (R0.T @ p_cam0)
                     = (R1 @ R0.T) @ p_cam0

        따라서 이론적으로는 Rc = R1 @ R0.T 로 둘 수 있고,
        코드에서는 같은 상대 회전을 R0.T @ R1 형태로 사용하고 있다.
        """

        # 카메라 좌표계에서의 상대 회전 행렬
        Rc = R0.T @ R1

        # 회전 행렬 -> 회전 벡터 로드리게스 역변환
        # theta는 회전 값
        theta = np.arccos(np.clip((np.trace(Rc) - 1.0) / 2.0, -1.0, 1.0))
        
        # 회전 값이 작으면 회전 벡터를 0으로 설정 (회전하여도 거의 의미가 없기 때문)
        if abs(theta) < 1e-8:
            r_cam = np.zeros(3, dtype=np.float64)
            
        # 회전 값을 회전 벡터에 반영
        # axis는 단위 회전축
        else:
            axis = (1.0 / (2.0 * np.sin(theta))) * np.array(
                [Rc[2, 1] - Rc[1, 2],
                 Rc[0, 2] - Rc[2, 0],
                 Rc[1, 0] - Rc[0, 1]],
                dtype=np.float64
            )
            r_cam = axis * theta

        # 총 회전각 계산
        angle_rad = np.linalg.norm(r_cam)
        angle_deg = np.degrees(angle_rad)
        
        # 총 회전각이 임계값보다 작으면 건너뜀
        if angle_deg < min_angle:
            continue

        # IMU gyroscope 적분을 통한 회전 행렬 계산
        # t0 ~ t1 구간의 IMU 데이터 선택
        select = (imu[:, 0] >= t0) & (imu[:, 0] <= t1)
        s = imu[select]
        
        # 선택된 gyro 데이터가 2개 미만이면 건너뜀
        if s.shape[0] < 2:
            continue

        # 카메라 프레임 사이 측정된 IMU 센서값들의 시간 집합
        ts = s[:, 0].astype(np.float64) * 1e-9
        # 카메라 프레임 사이 측정된 IMU 각속도 값들의 집합
        ws = s[:, 1:4].astype(np.float64)

        # 회전 적분 근사
        r_imu = np.zeros(3, dtype=np.float64)
        for k in range(len(ws) - 1):
            # 부분 시간 간격
            dt = ts[k + 1] - ts[k]
            
            # 부분 각속도 평균
            w_avg = 0.5 * (ws[k] + ws[k + 1])
            
            if dt <= 0:
                continue
            
            # 부분 각도 변화량 (각속도 * 시간 = 각속도 변화량)
            dtheta = w_avg * dt
            
            # 누적 각도 갱신
            r_imu += dtheta

        rotation_cam_list.append(r_cam)
        rotation_imu_list.append(r_imu)

    return rotation_cam_list, rotation_imu_list





# IMU 변환 행렬 추정 함수
# Gyroscope와 Accelerometer 데이터를 모두 활용
def estimate_rotation_extrinsic(rotation_cam_list,
                                rotation_imu_list,
                                gravity_cam_list,
                                gravity_imu_list):

    # Gyroscope 회전 제약 추가
    A_list = list(rotation_imu_list)
    B_list = list(rotation_cam_list)

    if gravity_cam_list is not None and gravity_imu_list is not None:
        for g_cam, g_imu in zip(gravity_cam_list, gravity_imu_list):
            g_cam_n = g_cam / (np.linalg.norm(g_cam) + 1e-12)
            g_imu_n = g_imu / (np.linalg.norm(g_imu) + 1e-12)
            
            # Accelerometer 중력 벡터 제약 추가
            A_list.append(g_imu_n)
            B_list.append(g_cam_n)

    # 2차원 행렬 행태로 변환
    A = np.stack(A_list, axis=0)
    B = np.stack(B_list, axis=0)

    # 최적화를 위한 중심화
    A0 = A - A.mean(axis=0)
    B0 = B - B.mean(axis=0)

    # 공분산 행렬
    H = A0.T @ B0
    
    # 공분산 행렬 H의 SVD 분해
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        
    return R





# main 함수
if __name__ == "__main__":
    
    # 저장 경로 설정
    SAVE_DIR = "result"
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
    gravity_cam_lists, gravity_imu_lists = gravity_pairs(
        imu,
        poses,
        gyro_thresh=0.1,    # 정지 기준
    )
    
    if len(gravity_cam_lists) == 0 or len(gravity_imu_lists) == 0:
        gravity_cam_lists = None
        gravity_imu_lists = None

    # gyro 적분으로 얻은 델타 회전 + 중력 방향 제약 사용
    # 체커보드 특징점을 활용한 카메라 회전 행렬 추정
    # 이 두 값을 근사시켜 IMU 변환 회전 행렬을 구한다.
    # gap은 최대 시간 간격 임계값 (s)
    # min_angle는 최소 회전 각도 임계값 (deg)
    GAP = 3.0
    MIN_ANGLE = 10.0
    rotation_cam_lists, rotation_imu_lists = rotation_pairs(poses, imu, GAP, MIN_ANGLE)
    
    if len(rotation_cam_lists) < 30:
        raise RuntimeError("[Error] Not enough matching rotation pairs to estimate IMU extrinsics.")

    # Accelerometer 중력 벡터 가중치
    R = estimate_rotation_extrinsic(
        rotation_cam_lists,
        rotation_imu_lists,
        gravity_cam_lists,
        gravity_imu_lists,
    )

    out_dict = {
        "IMU_extrinsics": {
            # 3x3 회전 행렬
            "R_imu_to_cam": R.tolist(),
            "t_imu_to_cam": [0.0, 0.0, 0.0]
        }
    }
    
    # IMU extrinsics를 YAML로 저장
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, "IMU_extrinsics.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(out_dict, f, sort_keys=False, allow_unicode=True)
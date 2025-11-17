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
        term = (cv.TERM_CRITERIA_EPS + cv.CALIB_CB_MAX_ITER, 30, 1e-3)
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





# IMU accel로부터 중력 방향 추정 함수
def gravity_from_accel(imu, poses, gyro_thresh=0.1):
    """
    Inputs:
    imu: (N,7) [t_ns,gx,gy,gz,ax,ay,az]
    gyro_thresh: 정지 구간 판별 기준(rad/s)
    
    Outputs:
    gravity_mean: (3,) 중력 방향 벡터 추정값
    board_up_cam: (3,) 체커보드 위쪽 방향(카메라 프레임) 추정값
    """
    gyro = imu[:, 1:4]
    accel = imu[:, 4:7]

    # 각속도의 크기가 기준치 이하인 구간 선택 (정지 구간을 의미)
    gravity_mean = None
    if imu.size > 0:
        gyro_norm = np.linalg.norm(gyro, axis=1)
        mask = gyro_norm < gyro_thresh
        if np.any(mask):
            gravity_samples = accel[mask]
            gravity_mean = gravity_samples.mean(axis=0)

    # 카메라 프레임에서 체커보드의 법선 벡터 방향 계산
    """
    X축 : 체커보드 가로 방향 (cols)
    Y축 : 체커보드 세로 방향 (rows)
    Z축 : 체커보드 법선 방향 (위쪽)
    """
    
    normals = []
    # board 좌표계 -> 카메라 좌표계
    # 지금 구한 rvec(slovePnP 결과)는 체커보드 좌표계-> 카메라 좌표계 변환 회전 행렬
    # 체커보드 좌표계의 세로 방향(Y축): 보드에서 아래 방향
    board_y = np.array([0.0, 1.0, 0.0])

    for pose in poses:
        rvec = pose["rvec"].reshape(3, 1)
        
        # 회전 벡터 -> 회전 행렬
        # 축과 각도 표현을 행렬 형태로 변환(로드리게스 공식 사용)
        R, _ = cv.Rodrigues(rvec)
        
        # 카메라 좌표계에서 본 보드 아래 방향
        cam_y = R @ board_y
        cam_y = cam_y.flatten()
        cam_y /= (np.linalg.norm(cam_y) + 1e-12)
        normals.append(cam_y)
        
    if len(normals) > 0:
        normals = np.stack(normals, axis=0)
        cam_y_mean = normals.mean(axis=0)
        cam_y_mean /= (np.linalg.norm(cam_y_mean) + 1e-12)
        cam_y_mean

    print(cam_y_mean)
    # 두 값을 함께 반환
    return gravity_mean, cam_y_mean




























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
#   최소제곱으로 R을 찾음: cam ≈ R * imu
# ------------------------------------------------------------------
def estimate_rotation_extrinsic(camera_pairs, imu_pairs, g_cam=None, g_imu=None, gravity_weight=5):
    """
    기존 delta 회전 쌍(cam_pairs, imu_pairs)에 더해 중력 벡터 쌍(g_cam, g_imu)을 포함하여
    extrinsic 회전 R (IMU->CAM)을 Kabsch로 추정.

    gravity_weight: 중력 벡터 쌍을 몇 번 반복해서(가중치처럼) 포함할지.
    """
    A_list = list(imu_pairs)
    B_list = list(camera_pairs)

    if g_cam is not None and g_imu is not None:
        # 둘 다 단위벡터로 가정; 부족하면 정규화
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
# ------------------------------------------------------------------
def build_delta_pairs(poses, imu, max_dt_sec=0.5, ts_offset_ns=0):
    """
    poses: [{'t_ns', 'rvec', 'tvec'}, ...]
    imu:   (N,7) [t_ns,gx,gy,gz,ax,ay,az]

    반환:
      cam_pairs: [ (3,), ... ]
      imu_pairs: [ (3,), ... ]
    """
    cam_pairs = []
    imu_pairs = []

    for i in range(len(poses) - 1):
        p0 = poses[i]
        p1 = poses[i + 1]

        t0 = p0["t_ns"] + ts_offset_ns
        t1 = p1["t_ns"] + ts_offset_ns

        # 너무 긴 구간은 회전 오차가 커질 수 있으므로 제한
        if (t1 - t0) * 1e-9 > max_dt_sec:
            continue

        R0, _ = cv.Rodrigues(p0["rvec"].reshape(3, 1))
        R1, _ = cv.Rodrigues(p1["rvec"].reshape(3, 1))
        
        # 카메라 좌표계에서의 상대 회전
        Rc = R0.T @ R1
        rc_vec = rotmat_to_rotvec(Rc)

        imu_rot = integrate_gyro(imu, t0, t1)
        if imu_rot is None:
            continue

        cam_pairs.append(rc_vec)
        imu_pairs.append(imu_rot)

    return cam_pairs, imu_pairs





# ------------------------------------------------------------------
# 타임 오프셋 grid search + extrinsic R 추정
# ------------------------------------------------------------------
def grid_search_time_offset(poses, imu, search_range_s=0.2, step_s=0.01, g_cam=None, g_imu=None, gravity_weight=5):
    """
    search_range_s: [-search_range_s, +search_range_s] 범위에서 오프셋 탐색
    step_s: 검색 스텝 (초)
    """
    best = None
    best_err = 1e9

    offsets = np.arange(-search_range_s, search_range_s + 1e-9, step_s)

    for off in offsets:
        ts_off_ns = int(off * 1e9)
        cam_pairs, imu_pairs = build_delta_pairs(poses, imu, ts_offset_ns=ts_off_ns)

        if len(cam_pairs) < 8:
            # 쌍이 너무 적으면 신뢰성 떨어져서 skip
            continue

        R = estimate_rotation_extrinsic(cam_pairs, imu_pairs, g_cam=g_cam, g_imu=g_imu, gravity_weight=gravity_weight)

        A = np.stack(imu_pairs, axis=0)
        B = np.stack(cam_pairs, axis=0)

        A_rot = (R @ A.T).T
        diff = A_rot - B
        err = np.mean(np.linalg.norm(diff, axis=1))

        if err < best_err:
            best_err = err
            best = {
                "offset_s": off,
                "R": R,
                "err": err,
                "pairs": len(A)
            }

    return best





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

    # accel을 이용해 중력 방향 출력
    # 카메라 프레임에서 체커보드 위쪽 방향 추출
    world_gravity, cam_gravity = gravity_from_accel(imu, poses)
    
    if world_gravity is not None:
        g_norm = float(np.linalg.norm(world_gravity))
        # gravity 벡터와 gravity의 크기 출력
        print(f"[INFO] estimated gravity ≈ {world_gravity} (norm={g_norm:.3f})")
    else:
        print("[WARN] Cannot estimate gravity from IMU accel data. (No stationary periods found.)")

    # 카메라 프레임에서 중력 방향
    if cam_gravity is not None:
        print(f"[INFO] mean board normal direction (camera frame) ≈ {cam_gravity}")
    else:
        print("[WARN] Fail to extract checker board normal direction.")

    # # 중력 벡터는 아래 방향이고, 체커보드는 벽에 걸려 있지만 격자 rows(+Y)가
    # # 실제 세계의 "위쪽"(중력 반대 방향)과 평행하도록 설치되었다고 가정한다.
    # # 따라서:
    # #   - accel 평균으로 IMU 프레임에서의 중력 방향 g_imu_vec(아래)을 얻고,
    # #   - solvePnP로 얻은 rvec을 통해 체커보드 +Y 축을 카메라 프레임으로 옮긴
    # #     board_up_cam을 world-up proxy로 사용하며,
    # #   - 카메라 프레임에서의 중력 방향은 -board_up_cam 으로 본다.
    # # 이 (g_imu_vec, g_cam_vec) 쌍을 delta 회전 쌍(cam_pairs, imu_pairs)에 추가하여
    # # extrinsic R(imu->cam)을 추정할 때 중력 방향을 추가 구속 조건으로 활용한다.
    # g_cam_vec = None
    # g_imu_vec = None
    # if world_gravity is not None and cam_gravity is not None:
    #     # IMU에서 얻은 중력 벡터 (IMU 프레임 기준, 아래 방향)
    #     g_imu_vec = world_gravity / (np.linalg.norm(world_gravity) + 1e-12)

    #     # 체커보드 +Y를 월드 "위쪽"으로 가정했으므로,
    #     # 카메라 프레임에서의 중력(아래 방향)은 board_normal_vector의 반대 방향으로 본다.
    #     g_cam_vec = -cam_gravity
    #     print("[INFO] using gravity constraint (board +Y as world-up proxy): g_imu -> g_cam")
    # else:
    #     print("[INFO] gravity constraint unavailable; falling back to gyro-only pairs.")

    # # 타임 오프셋 + extrinsic 회전 추정
    # best = grid_search_time_offset(
    #     poses,
    #     imu,
    #     search_range_s=0.2,
    #     step_s=0.01,
    #     g_cam=g_cam_vec,
    #     g_imu=g_imu_vec,
    #     gravity_weight=5
    # )
    # if best is None:
    #     print("[ERROR] 매칭 가능한 회전 쌍이 부족하거나 search_range/step이 적절하지 않습니다.")

    # print(f"[INFO] best offset: {best['offset_s']:.6f} s")
    # print(f"[INFO] mean error:  {best['err']:.6f}")
    # print(f"[INFO] used pairs:  {best['pairs']}")

    # R = best["R"]

    # extrinsics 저장
    out = {
        "time_offset_s": float(best["offset_s"]),
        "R_imu_to_cam": R.tolist(),
        "t_imu_to_cam_m": [0.0, 0.0, 0.0],  # 이 스크립트에서는 translation은 추정하지 않음
        "note": (
            "rotation estimated from delta-rotations (gyro) + gravity constraint (accel+board normal). "
            "board normal treated as world-up proxy; sign adjusted relative to IMU gravity."
        )
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, "IMU.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] Saved {out_path}")
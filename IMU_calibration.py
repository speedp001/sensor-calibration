# ...existing code...
import os, glob, json
import numpy as np
import cv2 as cv
import yaml

# 설정
FRAME_DIR = os.path.join("out", "frames")
SPEC_ROWS = 6
SPEC_COLS = 8
SPEC_SQUARE = 0.025

def load_imu_csv(path):
    # t_ns,gx,gy,gz,ax,ay,az
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    # columns: t_ns, gx,gy,gz, ax,ay,az
    return data

def integrate_gyro(imu, t0_ns, t1_ns):
    # imu: ndarray Nx7 (t_ns,gx,gy,gz,ax,ay,az)
    # 적분해서 rotation vector (axis * angle) 반환 (rad)
    sel = (imu[:,0] >= t0_ns) & (imu[:,0] <= t1_ns)
    s = imu[sel]
    if s.shape[0] == 0:
        return None
    # 시간 간격 (s) between samples
    ts = s[:,0].astype(np.float64) * 1e-9
    ws = s[:,1:4].astype(np.float64)  # rad/s (assumed)
    # integrate using simple trapezoid: accumulate small rotation vectors
    rot = np.zeros(3, dtype=np.float64)
    for i in range(len(ws)-1):
        dt = ts[i+1] - ts[i]
        w_avg = 0.5*(ws[i]+ws[i+1])
        dtheta = w_avg * dt  # small-angle vector (rad)
        rot += dtheta  # approximate additive (valid for small increments)
    # last sample ignored for dt; acceptable
    # Return rotation vector magnitude and axis encoded as vector (axis*angle)
    return rot  # 3-vector

def rvec_to_rotmat(rvec):
    R,_ = cv.Rodrigues(rvec.reshape(3,1))
    return R

def rotmat_to_rotvec(R):
    # convert to axis-angle vector (axis * angle)
    theta = np.arccos(np.clip((np.trace(R)-1)/2.0, -1.0, 1.0))
    if abs(theta) < 1e-8:
        return np.zeros(3)
    rx = (1/(2*np.sin(theta))) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    return rx * theta

def estimate_rotation_extrinsic(camera_pairs, imu_pairs):
    # camera_pairs, imu_pairs: lists of 3-vectors (axis*angle) aligned pairwise
    A = np.stack(imu_pairs, axis=0)  # Nx3
    B = np.stack(camera_pairs, axis=0)  # Nx3
    # remove centroid
    A0 = A - A.mean(axis=0)
    B0 = B - B.mean(axis=0)
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    return R

def load_camera_poses_from_frames(frame_dir):
    # Returns list of dict {idx, t_ns, rvec, tvec}
    poses = []
    objp = np.zeros((SPEC_ROWS*SPEC_COLS, 3), np.float32)
    grid = np.mgrid[0:SPEC_COLS, 0:SPEC_ROWS].T.reshape(-1,2)
    objp[:,:2] = grid * SPEC_SQUARE
    K, dist = None, None
    # compute poses using solvePnP for each color frame
    for p in sorted(glob.glob(os.path.join(frame_dir, "color_*.png"))):
        idx = int(os.path.splitext(os.path.basename(p))[0].split('_')[-1])
        img = cv.imread(p)
        meta_p = os.path.join(frame_dir, f"meta_{idx:06d}.json")
        if not os.path.exists(meta_p): continue
        meta = json.load(open(meta_p,'r'))
        t_ns = int(meta['t_ns'])
        ok, corners = None, None
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (SPEC_COLS, SPEC_ROWS))
        if not ret: continue
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), term)
        if K is None:
            # try read factory intrinsics if exists
            import yaml as _y
            f = os.path.join("out","factory_intrinsics.yaml")
            if os.path.exists(f):
                d = _y.safe_load(open(f,'r'))
                K = np.array([[d['color']['fx'],0,d['color']['cx']],[0,d['color']['fy'],d['color']['cy']],[0,0,1]], dtype=np.float64)
                dist = np.array(d['color']['dist'], dtype=np.float64).reshape(-1,1)
            else:
                # fallback: assume focal ~fx, cx from image shape
                K = np.array([[600,0,img.shape[1]/2],[0,600,img.shape[0]/2],[0,0,1]], dtype=np.float64)
                dist = np.zeros((5,1))
        okp, rvec, tvec = cv.solvePnP(objp, corners, K, dist, flags=cv.SOLVEPNP_ITERATIVE)
        if not okp: continue
        poses.append({'idx': idx, 't_ns': t_ns, 'rvec': rvec.flatten(), 'tvec': tvec.flatten()})
    return poses

def build_delta_pairs(poses, imu, max_dt_sec=0.5, ts_offset_ns=0):
    # Build matched delta rotations between successive camera poses and IMU-integrated rotations.
    cam_pairs = []
    imu_pairs = []
    for i in range(len(poses)-1):
        p0, p1 = poses[i], poses[i+1]
        t0, t1 = p0['t_ns'] + ts_offset_ns, p1['t_ns'] + ts_offset_ns
        if (t1 - t0) * 1e-9 > max_dt_sec: 
            continue
        R0 = rvec_to_rotmat(p0['rvec'])
        R1 = rvec_to_rotmat(p1['rvec'])
        Rc = R0.T @ R1
        rc_vec = rotmat_to_rotvec(Rc)  # camera delta as axis-angle vector
        imu_rot = integrate_gyro(imu, t0, t1)
        if imu_rot is None: continue
        cam_pairs.append(rc_vec)
        imu_pairs.append(imu_rot)
    return cam_pairs, imu_pairs

def grid_search_time_offset(poses, imu, search_range_s=0.2, step_s=0.01):
    best = None
    best_err = 1e9
    rng = np.arange(-search_range_s, search_range_s+1e-9, step_s)
    for off in rng:
        ts_off_ns = int(off * 1e9)
        cam_pairs, imu_pairs = build_delta_pairs(poses, imu, ts_offset_ns=ts_off_ns)
        if len(cam_pairs) < 8: continue
        R = estimate_rotation_extrinsic(cam_pairs, imu_pairs)
        # compute residual error after rotating imu_pairs by R
        A = np.stack(imu_pairs)
        B = np.stack(cam_pairs)
        A_rot = (R @ A.T).T
        err = np.mean(np.linalg.norm(A_rot - B, axis=1))
        if err < best_err:
            best_err = err
            best = {'offset_s': off, 'R': R, 'err': err, 'pairs': len(A)}
    return best

def main():
    imu_csv = os.path.join(FRAME_DIR, "imu_log.csv")
    if not os.path.exists(imu_csv):
        print("[ERROR] imu_log.csv 없음. 녹화시 IMU 로그를 out/frames/imu_log.csv 로 저장하세요.")
        return
    imu = load_imu_csv(imu_csv)
    poses = load_camera_poses_from_frames(FRAME_DIR)
    print(f"[INFO] loaded {len(poses)} camera poses, {imu.shape[0]} imu samples")
    best = grid_search_time_offset(poses, imu, search_range_s=0.2, step_s=0.01)
    if best is None:
        print("[ERROR] 적절한 파라미터로 매칭 실패. 녹화 데이터 더 확보하거나 search 범위 조정.")
        return
    print(f"[INFO] best offset {best['offset_s']} s, err {best['err']:.6f}, pairs {best['pairs']}")
    R = best['R']
    save = {
        'time_offset_s': float(best['offset_s']),
        'R_imu_to_cam': R.tolist(),
        't_imu_to_cam_m': [0.0, 0.0, 0.0],
        'note': 'rotation estimated from delta-rotations. translation not estimated here.'
    }
    with open(os.path.join("out","imu_to_camera_extrinsics.yaml"), "w") as f:
        yaml.safe_dump(save, f, sort_keys=False)
    print("[OK] Saved out/imu_to_camera_extrinsics.yaml")

if __name__ == "__main__":
    main()
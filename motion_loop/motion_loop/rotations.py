"""All rotation math in one place (numpy only).

Motions are unit quaternions in (x, y, z, w) order, the order three.js and the player JSON use.
Euler angles remain only as a way to WRITE poses by hand (test motions, mock generator, keyframe
head/torso angles): R = Rx(a) @ Ry(b) @ Rz(c), three.js order 'XYZ'. They are never stored, because
Euler values jump near |b| = 90 deg (gimbal lock) even when the rotation itself moves smoothly.
"""
import numpy as np


# ---------------- conversions ----------------
def euler_to_mat(e):
    """e [...,3] radians -> [...,3,3], R = Rx @ Ry @ Rz."""
    e = np.asarray(e, float)
    c, s = np.cos(e), np.sin(e)
    cx, cy, cz, sx, sy, sz = c[..., 0], c[..., 1], c[..., 2], s[..., 0], s[..., 1], s[..., 2]
    R = np.empty(e.shape[:-1] + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 0, 2] = cy * cz, -cy * sz, sy
    R[..., 1, 0], R[..., 1, 1], R[..., 1, 2] = cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy
    R[..., 2, 0], R[..., 2, 1], R[..., 2, 2] = sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy
    return R


def aa_to_mat(aa):
    """Axis-angle (rotation vector) [...,3] -> [...,3,3]."""
    aa = np.asarray(aa, float)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-12)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def mat_to_quat(R):
    """[...,3,3] -> [...,4] (x,y,z,w) with w >= 0."""
    R = np.asarray(R, float)
    r = lambda i, j: R[..., i, j]
    t = np.stack([1 + r(0, 0) + r(1, 1) + r(2, 2), 1 + r(0, 0) - r(1, 1) - r(2, 2),
                  1 - r(0, 0) + r(1, 1) - r(2, 2), 1 - r(0, 0) - r(1, 1) + r(2, 2)], -1)
    with np.errstate(all="ignore"):
        s = 2 * np.sqrt(np.maximum(t, 1e-12))
        # four ways to recover (w,x,y,z); pick the numerically safest per element
        q = np.stack([
            np.stack([s[..., 0] / 4, (r(2, 1) - r(1, 2)) / s[..., 0], (r(0, 2) - r(2, 0)) / s[..., 0], (r(1, 0) - r(0, 1)) / s[..., 0]], -1),
            np.stack([(r(2, 1) - r(1, 2)) / s[..., 1], s[..., 1] / 4, (r(0, 1) + r(1, 0)) / s[..., 1], (r(0, 2) + r(2, 0)) / s[..., 1]], -1),
            np.stack([(r(0, 2) - r(2, 0)) / s[..., 2], (r(0, 1) + r(1, 0)) / s[..., 2], s[..., 2] / 4, (r(1, 2) + r(2, 1)) / s[..., 2]], -1),
            np.stack([(r(1, 0) - r(0, 1)) / s[..., 3], (r(0, 2) + r(2, 0)) / s[..., 3], (r(1, 2) + r(2, 1)) / s[..., 3], s[..., 3] / 4], -1),
        ], -2)
    q = np.take_along_axis(q, np.argmax(t, -1)[..., None, None], -2)[..., 0, :]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., :1] < 0, -q, q)
    return q[..., [1, 2, 3, 0]]


def quat_to_mat(q):
    q = np.asarray(q, float)
    x, y, z, w = (q[..., i] for i in range(4))
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 0, 2] = 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)
    R[..., 1, 0], R[..., 1, 1], R[..., 1, 2] = 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)
    R[..., 2, 0], R[..., 2, 1], R[..., 2, 2] = 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
    return R


def to_6d(q):
    """[...,4] -> [...,6]: the first two columns of the rotation matrix. Unlike quaternions it has
    no q / -q ambiguity and no jumps, so it is what networks get as input."""
    R = quat_to_mat(q)
    return np.concatenate([R[..., 0], R[..., 1]], -1)


# ---------------- quaternion algebra ----------------
def quat_mul(a, b):
    """Rotation b first, then a (same order as matrices: R(a) @ R(b))."""
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], -1)


def quat_conj(q):
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def normalize(q):
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def rotvec(q):
    """Log map: [...,4] -> [...,3] rotation vector (axis * angle), angle in [0, pi]."""
    q = np.where(q[..., 3:] < 0, -q, q)
    n = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    ang = 2 * np.arctan2(n, q[..., 3:])
    return q[..., :3] * np.where(n > 1e-9, ang / np.maximum(n, 1e-12), 2.0)


def from_rotvec(v):
    v = np.asarray(v, float)
    ang = np.linalg.norm(v, axis=-1, keepdims=True)
    k = np.where(ang > 1e-9, np.sin(ang / 2) / np.maximum(ang, 1e-12), 0.5)
    return np.concatenate([v * k, np.cos(ang / 2)], -1)


def angle(q):
    """Rotation angle from rest, [...,4] -> [...] in [0, pi]."""
    return np.linalg.norm(rotvec(q), axis=-1)


def slerp(q0, q1, t):
    d = np.sum(q0 * q1, -1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.abs(d).clip(0, 1)
    th = np.arccos(d)
    sin = np.sin(th)
    near = sin < 1e-6
    a = np.where(near, 1 - t, np.sin((1 - t) * th) / np.where(near, 1, sin))
    b = np.where(near, t, np.sin(t * th) / np.where(near, 1, sin))
    return normalize(a * q0 + b * q1)


# ---------------- motions: q [T, J, 4] ----------------
def continuous(q):
    """Remove q / -q sign flips along time (both mean the same rotation); first frame gets w >= 0."""
    q = np.array(q, float)
    q[0] = np.where(q[0, ..., 3:] < 0, -q[0], q[0])
    for t in range(1, len(q)):
        q[t] = np.where(np.sum(q[t] * q[t - 1], -1, keepdims=True) < 0, -q[t], q[t])
    return q


def from_euler(e):
    """Hand-written Euler motion [T,J,3] -> quaternion motion [T,J,4] float32."""
    return continuous(mat_to_quat(euler_to_mat(e))).astype(np.float32)


def resample(q, new_T):
    """Slerp a motion to new_T frames covering the same duration."""
    q = continuous(q)
    if len(q) == 1:
        return np.repeat(q, new_T, 0)
    pos = np.linspace(0, len(q) - 1, new_T)
    i0 = np.minimum(np.floor(pos).astype(int), len(q) - 2)
    return slerp(q[i0], q[i0 + 1], (pos - i0)[:, None, None])


def resample_fps(q, fps_in, fps_out):
    return resample(q, int(round((len(q) - 1) / fps_in * fps_out)) + 1)


def angular_velocity(q, fps):
    """[T,J,4] -> [T-1,J,3] rad/s, in each joint's own frame. Its norm is the true rotation speed."""
    return rotvec(quat_mul(quat_conj(q[:-1]), q[1:])) * fps

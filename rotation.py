import numpy as np

# Use a single complex dtype for numpy everywhere.
DTYPE = np.complex128

INV_SQRT2 = 1.0 / np.sqrt(2.0)
H = INV_SQRT2 * np.array([[1, 1], [1, -1]], dtype=DTYPE)

X = np.array([[0, 1], [1, 0]], dtype=DTYPE)
Y = np.array([[0, -1j], [1j, 0]], dtype=DTYPE)
Z = np.array([[1, 0], [0, -1]], dtype=DTYPE)

# LAMBDA_PI is the base rotation angle realized by the H/T building blocks:
# cos(LAMBDA_PI) = cos^2(pi/8) = (1 + 1/sqrt2)/2. Because LAMBDA_PI / (2 pi) is
# irrational, the multiples {k * LAMBDA_PI mod 2 pi} densely fill [0, 2 pi).
LAMBDA_PI = np.arccos((1.0 + INV_SQRT2) / 2.0)
TWO_PI = 2.0 * np.pi


class Bloch:
    """Axis-angle (Bloch) form of a 2x2 unitary G:

        G = e^{i alpha} (cos(theta/2) I - i sin(theta/2) (n . sigma))

    i.e. a global phase e^{i alpha} times a rotation by angle `theta` about the
    Bloch-sphere axis `n`. Here (n . sigma) = n_x X + n_y Y + n_z Z.
    """

    alpha: float  # global phase
    n: np.ndarray  # unit rotation axis, shape (3,): [n_x, n_y, n_z]
    theta: float  # rotation angle


def to_bloch(g: np.ndarray) -> Bloch:
    """Recover the Bloch form (alpha, n, theta) of a 2x2 unitary `g`."""
    alpha = 0.5 * np.angle(np.linalg.det(g))
    u = np.exp(-1j * alpha) * g

    c = np.clip(np.real(np.trace(u)) / 2.0, -1.0, 1.0)
    theta = 2.0 * np.arccos(c)

    s = np.sin(theta / 2.0)

    if abs(s) < 1e-12:
        n = np.array([1.0, 0.0, 0.0])
    else:
        nx = np.real((1j * np.trace(X @ u)) / (2.0 * s))
        ny = np.real((1j * np.trace(Y @ u)) / (2.0 * s))
        nz = np.real((1j * np.trace(Z @ u)) / (2.0 * s))

        n = np.array([nx, ny, nz])
        n /= np.linalg.norm(n)

    b = Bloch()
    b.alpha = alpha
    b.n = n
    b.theta = theta

    return b


# n1, n2 are two orthogonal Bloch-sphere axes (n1 . n2 == 0)
# TODO: fill in the two orthogonal rotation axes (each a length-3
# unit vector [x, y, z])
c = np.sqrt(2.0) / np.tan(np.pi / 8.0)

n1 = np.array([-c / np.sqrt(2), 1.0, c / np.sqrt(2)])
n1 /= np.linalg.norm(n1)

n2 = np.array([1.0 / np.sqrt(2), c, -1.0 / np.sqrt(2)])
n2 /= np.linalg.norm(n2)

# frame derived from the axes (given)
# take the dot product of the Bloch axis with these
# the minus sign arises from the double cover issue
a1 = -n1
a2 = -n2
a3 = np.cross(a1, a2)


def n1n2n1_angles(b: Bloch) -> tuple[float, float, float, float]:
    """Factor the rotation part of a unitary (given as its Bloch form `b`) as
        u = e^{i global_phase} * Rn1(alpha) * Rn2(beta) * Rn1(gamma)

    where Ra(angle) is a rotation by `angle` about axis a, and {a1, a2, a3} is
    the orthonormal frame defined above. Returns (alpha, beta, gamma, global_phase).
    """
    def n1n2n1_angles(b: Bloch) -> tuple[float, float, float, float]:
    """Factor the rotation part of a unitary (given as its Bloch form `b`) as
        u = e^{i global_phase} * Rn1(alpha) * Rn2(beta) * Rn1(gamma)

    where Ra(angle) is a rotation by `angle` about axis a, and {a1, a2, a3} is
    the orthonormal frame defined above. Returns (alpha, beta, gamma, global_phase).
    """
    phi = b.theta / 2.0
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    v = sin_phi * b.n

    v_a1 = np.dot(v, a1)
    v_a2 = np.dot(v, a2)
    v_a3 = np.dot(v, a3)

    cos_beta = cos_phi
    sin_beta = np.sqrt(v_a2**2 + v_a3**2)
    beta = np.arctan2(sin_beta, cos_beta)

    if np.abs(sin_beta) < 1e-12:
        alpha = b.theta
        gamma = 0.0
    else:
        sum_angles = np.arctan2(v_a1, cos_phi)
        diff_angles = np.arctan2(v_a3, v_a2)

        alpha = 0.5 * (sum_angles - diff_angles)
        gamma = 0.5 * (sum_angles + diff_angles)

    return 2.0 * alpha, 2.0 * beta, 2.0 * gamma, b.alpha


def approx_angle_with_tolerance(angle: float, tolerance: float) -> int:
    """Find an integer multiple k such that
        (k * LAMBDA_PI) mod 2*pi  ~=  angle   (within `tolerance`)
    Since LAMBDA_PI / (2 pi) is irrational, such a k always exists; search
    k = 1, 2, 3, ... and return the first one whose wrapped multiple lands within
    `tolerance` of `angle` (compare both as angles in [0, 2 pi)).

    Hint:
      * wrap an angle into [0, 2 pi)
      * the angular distance between two wrapped angles a, b is
        min(|a - b|, TWO_PI - |a - b|) (so 0.01 and 2*pi - 0.01 count as close).
    """
    angle %= TWO_PI

    k = 1
    while True:
        current = (k * LAMBDA_PI) % TWO_PI

        diff = abs(current - angle)
        diff = min(diff, TWO_PI - diff)

        if diff <= tolerance:
            return k

        k += 1


def decompose_2x2(u: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    """Approximate a 2x2 unitary `u` as a product of powers of M1 and M2:

        u  ~=  M1^k * M2^l * M1^m     (up to a global phase)

    where M1 is a rotation about axis a1 and M2 a rotation about axis a2, each by
    the base angle realized by the H/T building blocks. Returns the powers
    (k, l, m).

    Steps (combine the two functions above):

      1. Get the Bloch form of u (to_bloch), then factor its rotation into the
         three frame angles with n1n2n1_angles:
             alpha, beta, gamma, _global_phase = n1n2n1_angles(to_bloch(u))
         alpha and gamma are rotations about a1 (realized by powers of M1);
         beta is a rotation about a2 (realized by powers of M2).

      2. Convert each angle to an integer power with approx_angle_with_tolerance:
             k = approx_angle_with_tolerance(alpha, tolerance)   # power of M1
             l = approx_angle_with_tolerance(beta,  tolerance)   # power of M2
             m = approx_angle_with_tolerance(gamma, tolerance)   # power of M1
         (Mind the relationship between a target rotation angle and the base
         angle each application of M1/M2 adds.)

      3. Return (k, l, m).
    """
    b = to_bloch(u)

    alpha, beta, gamma, global_phase = n1n2n1_angles(b)

    k = approx_angle_with_tolerance(alpha, tolerance)
    l = approx_angle_with_tolerance(beta, tolerance)
    m = approx_angle_with_tolerance(gamma, tolerance)

    return k, l, m

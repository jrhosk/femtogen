import math

import tensorflow as tf

from finn.elastics import kelly, galster
from finn.four_vector import *
from finn import graphtrace

cos = tf.math.cos
sin = tf.math.sin
sqrt = tf.math.sqrt
T = tf.transpose
pi = tf.constant(math.pi, dtype=tf.float64)

# mass of the proton in GeV
M = tf.constant(0.93828, dtype=tf.float64)

# electromagnetic fine structure constant
alpha = tf.constant(0.0072973525693, dtype=tf.float64)


@graphtrace.trace_graph
def ff_to_xsx_old(reH, imH, reE, imE, reHt, imHt, reEt, imEt, phi, xbj, t, Q2, L, k0):
    """ Calculation of cross sections from form factor predictions.

    Args:
        reH (tf.Tensor) : Tensor of shape (batch_size,). 0 index of model output. dtype tf.float32
        imH (tf.Tensor) : Tensor of shape (batch_size,). 1 index of model output. dtype tf.float32
        reE (tf.Tensor) : Tensor of shape (batch_size,). 2 index of model output. dtype tf.float32
        imE (tf.Tensor) : Tensor of shape (batch_size,). 3 index of model output. dtype tf.float32
        reHt (tf.Tensor) : Tensor of shape (batch_size,). 4 index of model output. dtype tf.float32
        imHt (tf.Tensor) : Tensor of shape (batch_size,). 5 index of model output. dtype tf.float32
        reEt (tf.Tensor) : Tensor of shape (batch_size,). 6 index of model output. dtype tf.float32
        imEt (tf.Tensor) : Tensor of shape (batch_size,). 7 index of model output. dtype tf.float32
        xbj (tf.Tensor) : Tensor of shape (batch_size,). 0 index of kinematic input. dtype tf.float32
        t (tf.Tensor) : Tensor of shape (batch_size,). 1 index of kinematic input. dtype tf.float32
        Q2 (tf.Tensor) : Tensor of shape (batch_size,). 2 index of kinematic input. dtype tf.float32
        phi (tf.Tensor) : Tensor of shape (batch_size,). 3 index of kinematic input. dtype tf.float32
        L (tf.Tensor) : Tensor of shape (batch_size,). 0 index of sigma_true label. dtype *tf.int32*
    
    Returns:
        Calculated cross section tf.Tensor of shape (batch_size,)
    """
    depth_vector = tf.ones((reH.shape[0],), dtype=tf.float64)

    ###################################
    ## Secondary Kinematic Variables ##
    ###################################

    # energy of the virtual photon
    nu = Q2 / (2.0 * M * xbj)

    # skewness parameter set by xbj, t, and Q^2
    xi = xbj * ((1.0 + (t / (2.0 * Q2))) / (2.0 - xbj + ((xbj * t) / Q2)))

    # gamma variable ratio of virtuality to energy of virtual photon
    gamma = sqrt(Q2) / nu

    # fractional energy of virtual photon
    y = sqrt(Q2) / (gamma * k0)

    # final lepton energy
    k0p = k0 * (1.0 - y)

    # minimum t value
    t0 = -(4.0 * xi * xi * M * M) / (1.0 - (xi * xi))

    # Lepton Angle Kinematics of initial lepton
    costl = -(1.0 / (sqrt(1.0 + gamma * gamma))) * (1.0 + (y * gamma * gamma / 2.0))
    sintl = (gamma / (sqrt(1.0 + gamma * gamma))) * sqrt(
        1.0 - y - (y * y * gamma * gamma / 4.0)
    )

    # Lepton Angle Kinematics of final lepton
    sintlp = sintl / (1.0 - y)
    costlp = (costl + y * sqrt(1.0 + gamma * gamma)) / (1.0 - y)

    # final proton energy
    p0p = M - (t / (2.0 * M))

    # ratio of longitudinal to transverse virtual photon flux
    eps = (1.0 - y - 0.25 * y * y * gamma * gamma) / (
        1.0 - y + 0.5 * y * y + 0.25 * y * y * gamma * gamma
    )

    # angular kinematics of outgoing photon
    cost = -(1 / (sqrt(1 + gamma * gamma))) * (
        1 + (0.5 * gamma * gamma) * ((1 + (t / Q2)) / (1 + ((xbj * t) / (Q2))))
    )
    cost = tf.math.maximum(cost, -1.)
    sint = sqrt(1.0 - cost * cost)

    # outgoing photon energy
    q0p = (sqrt(Q2) / gamma) * (1 + ((xbj * t) / Q2))

    # conversion from GeV to NanoBarn
    jacobian = (1.0 / (2.0 * M * xbj * k0)) * 4.0 * pi
    conversion = (0.1973 * 0.1973) * 10000000 * jacobian

    # ratio of momentum transfer to proton mass
    tau = -t / (4.0 * M * M)

    ###############################################################################
    ## Creates arrays of 4-vector kinematics uses in Bethe Heitler Cross Section ##
    ###############################################################################

    # initial proton 4-momentum
    p = T(
        tf.convert_to_tensor(
            [
                M * depth_vector,
                0.0 * depth_vector,
                0.0 * depth_vector,
                0.0 * depth_vector,
            ]
        )
    )

    # initial lepton 4-momentum
    k = T(
        tf.convert_to_tensor(
            [k0 * depth_vector, k0 * sintl, 0.0 * depth_vector, k0 * costl]
        )
    )

    # final lepton 4-momentum
    kp = T(
        tf.convert_to_tensor(
            [k0p * depth_vector, k0p * sintlp, 0.0 * depth_vector, k0p * costlp]
        )
    )

    # virtual photon 4-momentum
    q = k - kp

    ##################################
    ## Creates four vector products ##
    ##################################
    plp = product(p, p)  # pp
    qq = product(q, q)  # qq
    kk = product(k, k)  # kk
    kkp = product(k, kp)  # kk'
    kq = product(k, q)  # kq
    pk = product(k, p)  # pk
    pkp = product(kp, p)  # pk'

    # sets the Mandelstam variables s which is the center of mass energy
    s = kk + (2 * pk) + plp

    # the Gamma factor in front of the cross section
    Gamma = (alpha ** 3) / (
        16.0 * (pi ** 2) * ((s - M * M) ** 2) * sqrt(1.0 + gamma ** 2) * xbj
    )

    phi = phi * 0.0174532951  # radian conversion

    # final real photon 4-momentum
    qp = T(
        tf.convert_to_tensor(
            [
                q0p * depth_vector,
                q0p * sint * T(cos(phi)),
                q0p * sint * T(sin(phi)),
                q0p * cost * depth_vector,
            ]
        )
    )

    # momentum transfer Δ from the initial proton to the final proton
    d = q - qp

    # final proton momentum
    pp = p + d

    # average initial proton momentum
    P = 0.5 * (p + pp)

    # 4-vector products of variables multiplied by spin vectors
    ppSL = ((M) / (sqrt(1.0 + gamma ** 2))) * (
        xbj * (1.0 - (t / Q2)) - (t / (2.0 * M ** 2))
    )
    kSL = (
        ((Q2) / (sqrt(1.0 + gamma ** 2)))
        * (1.0 + 0.5 * y * gamma ** 2)
        * (1.0 / (2.0 * M * xbj * y))
    )
    kpSL = (
        ((Q2) / (sqrt(1 + gamma ** 2)))
        * (1 - y - 0.5 * y * gamma ** 2)
        * (1.0 / (2.0 * M * xbj * y))
    )

    # 4-vector products denoted in the paper by the commented symbols
    kd = product(k, d)  # dΔ
    kpd = product(kp, d)  # k'Δ
    kP = product(k, P)  # kP
    kpP = product(kp, P)  # k'P
    kqp = product(k, qp)  # kq'
    kpqp = product(kp, qp)  # k'q'
    dd = product(d, d)  # ΔΔ
    Pq = product(P, q)  # Pq
    Pqp = product(P, qp)  # Pq'
    qd = product(q, d)  # qΔ
    qpd = product(qp, d)  # q'Δ

    # transverse vector products
    kkT = tproduct(k, k)
    kqpT = tproduct(k, qp)
    kkpT = tproduct(k, kp)
    ddT = tproduct(d, d)
    kdT = tproduct(k, d)
    kpqpT = tproduct(kp, qp)
    qpdT = tproduct(qp, d)
    kPT = tproduct(k, P)
    kpPT = tproduct(kp, P)
    qpPT = tproduct(qp, P)
    kpdT = tproduct(kp, d)

    # light cone variables expressed as A^{+-} = 1/sqrt(2)(A^{0} +- A^{3})
    inv_root_2 = 1 / sqrt(2.0)
    inv_root_2 = tf.dtypes.cast(inv_root_2, tf.float64)
    kplus = T(inv_root_2 * (k[..., 0] + k[..., 3]))
    kpplus = T(inv_root_2 * (kp[..., 0] + kp[..., 3]))
    kminus = T(inv_root_2 * (k[..., 0] - k[..., 3]))
    kpminus = T(inv_root_2 * (kp[..., 0] - kp[..., 3]))
    qplus = T(inv_root_2 * (q[..., 0] + q[..., 3]))
    qpplus = T(inv_root_2 * (qp[..., 0] + qp[..., 3]))
    qminus = T(inv_root_2 * (q[..., 0] - q[..., 3]))
    qpminus = T(inv_root_2 * (qp[..., 0] - qp[..., 3]))
    Pplus = T(inv_root_2 * (P[..., 0] + P[..., 3]))
    Pminus = T(inv_root_2 * (P[..., 0] - P[..., 3]))
    dplus = T(inv_root_2 * (d[..., 0] + d[..., 3]))
    dminus = T(inv_root_2 * (d[..., 0] - d[..., 3]))

    # expresssions used that appear in coefficient calculations
    Dplus = (1 / (2 * kpqp)) - (1 / (2 * kqp))
    Dminus = (1 / (2 * kpqp)) + (1 / (2 * kqp))

    # calculates BH
    AUUBH = ((16.0 * M * M) / (kqp * kpqp)) * (
        (4.0 * tau * (kP * kP + kpP * kpP)) - ((tau + 1.0) * (kd * kd + kpd * kpd))
    )
    BUUBH = ((32.0 * M * M) / (kqp * kpqp)) * (kd * kd + kpd * kpd)

    # calculates BHLL
    ALLBH = -((16.0 * M * M) / (kqp * kpqp)) * (
        (ppSL / M) * ((kpd * kpd - kd * kd) - 2.0 * tau * (kpd * pkp - kd * pk))
        + t * (kSL / M) * (1.0 + tau) * kd
        - t * (kpSL / M) * (1.0 + tau) * kpd
    )
    BLLBH = ((16.0 * M * M) / (kqp * kpqp)) * (
        (ppSL / M) * (kpd * kpd - kd * kd) + t * (kSL / M) * kd - t * (kpSL / M) * kpd
    )

    # converted Unpolarized Coefficients with the Gamma factor and in nano-barn
    con_AUUBH = (Gamma / t ** 2) * AUUBH * conversion
    con_BUUBH = (Gamma / t ** 2) * BUUBH * conversion

    # converted Longitudinally Polarized Coefficients with the Gamma Factor and in nano-barn
    con_ALLBH = (Gamma / t ** 2) * ALLBH * conversion
    con_BLLBH = (Gamma / t ** 2) * BLLBH * conversion

    ffF1, ffF2, ffGM = kelly(-t)

    # unpolarized Coefficients multiplied by the Form Factors calculated in form_factor.py
    # we use the Galster Form Factors as approximations
    bhAUU = con_AUUBH * ((ffF1 * ffF1) + (tau * ffF2 * ffF2))
    bhBUU = con_BUUBH * (tau * ffGM * ffGM)

    # polarized Coefficients multiplied by the Form Factors calculated in form_factor.py
    # using the Galster Form Factor Model
    bhALL = con_ALLBH * (ffF2 * ffGM)
    bhBLL = con_BLLBH * (ffGM * ffGM)

    # Calculation of the Total Unpolarized Bethe-Heitler Cross Section
    XSXUUBH = bhAUU + bhBUU
    XSXLLBH = bhALL + bhBLL

    # Calculates the Unpolarized Coefficients in front of the Elastic Form Factors and
    # Compton Form Factors
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    AUUBHDVCS = -16 * Dplus * (
        (kqpT - 2 * kkT - 2 * kqp) * kpP
        + (2 * kpqp - 2 * kkpT - kpqpT) * kP
        + kpqp * kPT
        + kqp * kpPT
        - 2 * kkp * kPT
    ) * cos(phi) - 16 * Dminus * (
        (2 * kkp - kpqpT - kkpT) * Pqp + 2 * kkp * qpPT - kpqp * kPT - kqp * kpPT
    ) * cos(
        phi
    )
    BUUBHDVCS = -8 * xi * Dplus * (
        (kqpT - 2 * kkT - 2 * kqp) * kpd
        + (2 * kpqp - 2 * kkpT - kpqpT) * kd
        + kpqp * kdT
        + kqp * kpdT
        - 2 * kkp * kdT
    ) * cos(phi) - 8 * xi * Dminus * (
        (2 * kkp - kpqpT - kkpT) * qpd + 2 * kkp * qpdT - kpqp * kdT - kqp * kpdT
    ) * cos(
        phi
    )
    CUUBHDVCS = -8 * Dplus * (
        (2 * kkp * kdT - kpqp * kdT - kqp * kpdT)
        + 2 * xi * (2 * kkp * kPT - kpqp * kPT - kqp * kpPT)
    ) * cos(phi) - 8 * Dminus * (
        (kkp * qpdT - kpqp * kdT - kqp * kpdT)
        + 2 * xi * (kkp * qpPT - kpqp * kPT - kqp * kpPT)
    ) * cos(
        phi
    )

    # Calculates the Unpolarized Beam Polarized Target Coefficients in front of the
    # Elastic Form Factors and Compton Form Factors
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    AULBHDVCS = -16 * Dplus * (
        kpP * (2 * kkT - kqpT + 2 * kqp)
        + kP * (2 * kkpT - kpqpT + 2 * kpqp)
        + 2 * kkp * kPT
        - kpqp * kPT
        - kqp * kpPT
    ) * sin(phi) - 16 * Dminus * (
        Pqp * (kkpT + kpqpT - 2 * kkp) - (2 * kkp * qpPT - kpqp * kPT - kqp * kpPT)
    ) * sin(
        phi
    )
    BULBHDVCS = -8 * xi * Dplus * (
        kpd * (2 * kkT - kqpT + 2 * kqp)
        + kd * (2 * kkpT - kpqpT + 2 * kpqp)
        + 2 * kkp * kdT
        - kpqp * kdT
        - kqp * kpdT
    ) * sin(phi) - 8 * xi * Dminus * (
        qpd * (kkpT + kpqpT - 2 * kkp) - (2 * kkp * qpdT - kpqp * kdT - kqp * kpdT)
    ) * sin(
        phi
    )
    CULBHDVCS = -4 * Dplus * (
        2 * (2 * kkp * kdT - kpqp * kdT - kqp * kpdT)
        + 4 * xi * (2 * kkp * kPT - kpqp * kPT - kqp * kpPT)
    ) * sin(phi) - 4 * Dminus * (
        -2 * (kkp * qpdT - kpqp * kdT - kqp * kpdT)
        - 4 * xi * (kkp * qpPT - kpqp * kPT - kqp * kpPT)
    ) * sin(
        phi
    )

    # Calculates the Polarized Beam Unpolarized Target Coefficients in front of the
    # Elastic Form Factors and Compton Form Factors
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    ALUBHDVCS = -16 * Dplus * (
        2
        * (
            k[:, 1] * Pplus * kp[:, 1] * kminus
            - k[:, 1] * Pplus * kpminus * k[:, 1]
            + k[:, 1] * Pminus * kpplus * k[:, 1]
            - k[:, 1] * Pminus * kp[:, 1] * kplus
            + k[:, 1] * P[:, 1] * kpminus * kplus
            - k[:, 1] * P[:, 1] * kpplus * kminus
        )
        + 2
        * (
            kp[:, 1] * Pplus * qpminus * k[:, 1]
            - kp[:, 1] * Pplus * qp[:, 1] * kminus
            + kp[:, 1] * Pminus * qp[:, 1] * kplus
            - kp[:, 1] * Pminus * qpplus * k[:, 1]
            + kp[:, 1] * P[:, 1] * qpplus * kminus
            - kp[:, 1] * P[:, 1] * qpminus * kplus
            + k[:, 1] * Pplus * qpminus * kp[:, 1]
            - k[:, 1] * Pplus * qp[:, 1] * kpminus
            + k[:, 1] * Pminus * qp[:, 1] * kpplus
            - k[:, 1] * Pminus * qpplus * kp[:, 1]
            + k[:, 1] * P[:, 1] * qpplus * kpminus
            - k[:, 1] * P[:, 1] * qpminus * kpplus
            + 2 * (qpminus * Pplus - qpplus * Pminus) * kkp
        )
    ) * sin(phi) - 16 * Dminus * (-2) * (
        2 * (kminus * kpplus - kplus * kpminus) * Pqp
        + kpminus * kplus * qp[:, 1] * P[:, 1]
        + kpplus * k[:, 1] * qpminus * P[:, 1]
        + kp[:, 1] * kminus * qpplus * P[:, 1]
        - kpplus * kminus * qp[:, 1] * P[:, 1]
        - kp[:, 1] * kplus * qpminus * P[:, 1]
        - kpminus * k[:, 1] * qpplus * P[:, 1]
        + kpminus * kplus * qp[:, 2] * P[:, 2]
        - kpplus * kminus * qp[:, 2] * P[:, 2]
    ) * sin(
        phi
    )
    BLUBHDVCS = -8 * xi * Dplus * (
        2
        * (
            k[:, 1] * dplus * kp[:, 1] * kminus
            - k[:, 1] * dplus * kpminus * k[:, 1]
            + k[:, 1] * dminus * kpplus * k[:, 1]
            - k[:, 1] * dminus * kp[:, 1] * kplus
            + k[:, 1] * d[:, 1] * kpminus * kplus
            - k[:, 1] * d[:, 1] * kpplus * kminus
        )
        + 2
        * (
            kp[:, 1] * dplus * qpminus * k[:, 1]
            - kp[:, 1] * dplus * qp[:, 1] * kminus
            + kp[:, 1] * dminus * qp[:, 1] * kplus
            - kp[:, 1] * dminus * qpplus * k[:, 1]
            + kp[:, 1] * d[:, 1] * qpplus * kminus
            - kp[:, 1] * d[:, 1] * qpminus * kplus
            + k[:, 1] * dplus * qpminus * kp[:, 1]
            - k[:, 1] * dplus * qp[:, 1] * kpminus
            + k[:, 1] * dminus * qp[:, 1] * kpplus
            - k[:, 1] * dminus * qpplus * kp[:, 1]
            + k[:, 1] * d[:, 1] * qpplus * kpminus
            - k[:, 1] * d[:, 1] * qpminus * kpplus
            + 2 * (qpminus * dplus - qpplus * dminus) * kkp
        )
    ) * sin(phi) - 8 * xi * Dminus * (-2) * (
        2 * (kminus * kpplus - kplus * kpminus) * qpd
        + kpminus * kplus * qp[:, 1] * d[:, 1]
        + kpplus * k[:, 1] * qpminus * d[:, 1]
        + kp[:, 1] * kminus * qpplus * d[:, 1]
        - kpplus * kminus * qp[:, 1] * d[:, 1]
        - kp[:, 1] * kplus * qpminus * d[:, 1]
        - kpminus * k[:, 1] * qpplus * d[:, 1]
        + kpminus * kplus * qp[:, 2] * d[:, 2]
        - kpplus * kminus * qp[:, 2] * d[:, 2]
    ) * sin(
        phi
    )
    CLUBHDVCS = 8 * Dplus * (
        2
        * (kp[:, 1] * kpminus * kplus * d[:, 1] - kp[:, 1] * kpplus * kminus * d[:, 1])
        + 2
        * (
            kp[:, 1] * qpminus * kplus * d[:, 1]
            - kp[:, 1] * qpplus * kminus * d[:, 1]
            + k[:, 1] * qpminus * kpplus * d[:, 1]
            - k[:, 1] * qpplus * kpminus * d[:, 1]
        )
    ) * sin(phi) + 8 * Dminus * (-2) * (
        -kpminus * k[:, 1] * qpplus * d[:, 1]
        + kpminus * kplus * qp[:, 1] * d[:, 1]
        + kpplus * k[:, 1] * qpminus * d[:, 1]
        - kpplus * kminus * qp[:, 1] * d[:, 1]
        + kp[:, 1] * kminus * qpplus * d[:, 1]
        - kp[:, 1] * kplus * qpminus * d[:, 1]
        - qp[:, 2] * d[:, 2] * (kpplus * kminus - kpminus * kplus)
    ) * sin(
        phi
    )

    # Calculates the Longitudinally Polarized Coefficients in front of the EFFs
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    ALLBHDVCS = -16 * Dplus * (
        2 * kp[:, 1] * (kp[:, 1] * kminus - kpminus * k[:, 1]) * Pplus
        + 2 * kp[:, 1] * (kpplus * k[:, 1] - kp[:, 1] * kplus) * Pminus
        + 2 * kp[:, 1] * (kpminus * kplus - kpplus * kminus) * P[:, 1]
        + 2
        * (
            kp[:, 1] * (qpminus * k[:, 1] - qp[:, 1] * kminus) * Pplus
            + kp[:, 1] * (qp[:, 1] * kplus - qpplus * k[:, 1]) * Pminus
            + kp[:, 1] * (qpplus * kminus - qpminus * kplus) * P[:, 1]
            + k[:, 1] * (qpminus * kp[:, 1] - qp[:, 1] * kpminus) * Pplus
            + k[:, 1] * (qp[:, 1] * kpplus - qpplus * kp[:, 1]) * Pminus
            + k[:, 1] * (qpplus * kpminus - qpminus * kpplus) * P[:, 1]
            - 2 * kkp * (qpplus * Pminus - qpminus * Pplus)
        )
    ) * cos(phi) - 16 * Dminus * (-2) * (
        2 * Pqp * (kpplus * kminus - kpminus * kplus)
        + P[:, 2] * qp[:, 2] * (kpplus * kminus - kpminus * kplus)
        + P[:, 1]
        * (
            kp[:, 1] * kplus * qpminus
            - kp[:, 1] * kminus * qpplus
            + kpminus * k[:, 1] * qpplus
            - kpminus * kplus * qp[:, 1]
            + kpplus * kminus * qp[:, 1]
            - kpplus * k[:, 1] * qpminus
        )
    ) * cos(
        phi
    )
    BLLBHDVCS = -8 * xi * Dplus * (
        2 * kp[:, 1] * (kp[:, 1] * kminus - kpminus * k[:, 1]) * dplus
        + 2 * kp[:, 1] * (kpplus * k[:, 1] - kp[:, 1] * kplus) * dminus
        + 2 * kp[:, 1] * (kpminus * kplus - kpplus * kminus) * d[:, 1]
        + 2
        * (
            kp[:, 1] * (qpminus * k[:, 1] - qp[:, 1] * kminus) * dplus
            + kp[:, 1] * (qp[:, 1] * kplus - qpplus * k[:, 1]) * dminus
            + kp[:, 1] * d[:, 1] * (qpplus * kminus - qpminus * kplus)
            + k[:, 1] * (qpminus * kp[:, 1] - qp[:, 1] * kpminus) * dplus
            + k[:, 1] * (qp[:, 1] * kpplus - qpplus * kp[:, 1]) * dminus
            + k[:, 1] * (qpplus * kpminus - qpminus * kpplus) * d[:, 1]
            - 2 * kkp * (qpplus * dminus - qpminus * dplus)
        )
    ) * cos(phi) - 8 * xi * Dminus * (-2) * (
        2 * qpd * (kpplus * kminus - kpminus * kplus)
        + d[:, 2] * qp[:, 2] * (kpplus * kminus - kpminus * kplus)
        + d[:, 1]
        * (
            kp[:, 1] * kplus * qpminus
            - kp[:, 1] * kminus * qpplus
            + kpminus * k[:, 1] * qpplus
            - kpminus * kplus * qp[:, 1]
            + kpplus * kminus * qp[:, 1]
            - kpplus * k[:, 1] * qpminus
        )
    ) * cos(
        phi
    )
    CLLBHDVCS = 16 * Dplus * (
        2 * (k[:, 1] * kminus * kpplus * d[:, 1] - k[:, 1] * kpminus * kplus * d[:, 1])
        + 2
        * (
            kp[:, 1] * qpplus * kminus * d[:, 1]
            - kp[:, 1] * qpminus * kplus * d[:, 1]
            + k[:, 1] * qpplus * kpminus * d[:, 1]
            - k[:, 1] * qpminus * kpplus * d[:, 1]
        )
    ) * cos(phi) - 16 * Dminus * (-2) * (
        -d[:, 1]
        * (
            kpminus * kplus * qp[:, 1]
            - kpminus * k[:, 1] * qpplus
            + kpplus * k[:, 1] * qpminus
            - kpplus * kminus * qp[:, 1]
            + kp[:, 1] * kminus * qpplus
            - kp[:, 1] * kplus * qpminus
        )
        + qp[:, 2] * kpplus * kminus * d[:, 2]
        - qp[:, 2] * kpminus * kplus * d[:, 2]
    ) * cos(
        phi
    )

    # Converted Unpolarized Coefficients with the Gamma factor and in nano-barn
    con_AUUBHDVCS = 2 * (Gamma / (Q2 * -t)) * AUUBHDVCS * conversion
    con_BUUBHDVCS = 2 * (Gamma / (Q2 * -t)) * BUUBHDVCS * conversion
    con_CUUBHDVCS = 2 * (Gamma / (Q2 * -t)) * CUUBHDVCS * conversion

    # Converted Longitudinally Polarized Coefficients with the Gamma Factor and in nano-barn
    con_ALLBHDVCS = (Gamma / (Q2 * -t)) * ALLBHDVCS * conversion
    con_BLLBHDVCS = (Gamma / (Q2 * -t)) * BLLBHDVCS * conversion
    con_CLLBHDVCS = (Gamma / (Q2 * -t)) * CLLBHDVCS * conversion

    # Converted Longitudinally Polarized Beam Unpolarized Target Coefficients with
    # the Gamma Factor and in nano-barn
    con_ALUBHDVCS = (Gamma / (Q2 * -t)) * ALUBHDVCS * conversion
    con_BLUBHDVCS = (Gamma / (Q2 * -t)) * BLUBHDVCS * conversion
    con_CLUBHDVCS = (Gamma / (Q2 * -t)) * CLUBHDVCS * conversion

    # Converted Longitudinally Polarized Target Unpolarized Beam Coefficients with
    # the Gamma Factor and in nano-barn
    con_AULBHDVCS = (Gamma / (Q2 * -t)) * AULBHDVCS * conversion
    con_BULBHDVCS = (Gamma / (Q2 * -t)) * BULBHDVCS * conversion
    con_CULBHDVCS = (Gamma / (Q2 * -t)) * CULBHDVCS * conversion

    # Unpolarized Coefficients multiplied by the Form Factors
    bhdvcsAUU = con_AUUBHDVCS * (ffF1 * reH + tau * ffF2 * reE)
    bhdvcsBUU = con_BUUBHDVCS * (ffGM * (reH + reE))
    bhdvcsCUU = con_CUUBHDVCS * (ffGM * reHt)

    # Polarized Coefficients multiplied by the Form Factors
    bhdvcsALU = con_ALUBHDVCS * (ffF1 * imH + tau * ffF2 * imE)
    bhdvcsBLU = con_BLUBHDVCS * (ffGM * (imH + imE))
    bhdvcsCLU = con_CLUBHDVCS * (ffGM * imHt)

    # Unpolarized Beam Polarized Target Coefficients multiplied by the Form Factors
    bhdvcsAUL = con_AULBHDVCS * (ffF1 * imHt - xi * ffF1 * imEt + tau * ffF2 * imEt)
    bhdvcsBUL = con_BULBHDVCS * (ffGM * imHt)
    bhdvcsCUL = con_CULBHDVCS * (ffGM * (imH + imE))

    # Polarized Beam Unpolarized Target Coefficients multiplied by the Form Factors
    bhdvcsALL = con_ALLBHDVCS * (ffF1 * reHt - xi * ffF1 * reEt + tau * ffF2 * reEt)
    bhdvcsBLL = con_BLLBHDVCS * (ffGM * reHt)
    bhdvcsCLL = con_CLLBHDVCS * (ffGM * (reH + reE))

    # Calculation of the Total Unpolarized Bethe-Heitler Cross Section
    XSXUUBHDVCS = bhdvcsAUU + bhdvcsBUU + bhdvcsCUU
    XSXLLBHDVCS = bhdvcsALL + bhdvcsBLL + bhdvcsCLL
    XSXULBHDVCS = bhdvcsAUL + bhdvcsBUL + bhdvcsCUL
    XSXLUBHDVCS = bhdvcsALU + bhdvcsBLU + bhdvcsCLU
    FUUT = (
        (Gamma / (Q2))
        / (1 - eps)
        * conversion
        * (
            4
            * (
                (1 - xi * xi) 
                * (reH * reH + imH * imH + reHt * reHt + imHt * imHt)
                + ((t0 - t) / (2 * M * M))
                * (
                    reE * reE
                    + imE * imE
                    + xi * xi * reEt * reEt
                    + xi * xi * imEt * imEt
                )
                - ((2 * xi * xi) / (1 - xi * xi))
                * (reH * reE + imH * imE + reHt * reEt + imHt * imEt)
            )
        )
    )

    #XSXUUBHDVCS = tf.math.maximum(XSXUUBHDVCS, -XSXUUBH+.02)
    XSXUU = XSXUUBHDVCS + XSXUUBH + FUUT * (1 / (1 - eps))
    XSXLL = XSXLLBHDVCS + XSXLLBH
    XSXUL = XSXULBHDVCS
    XSXLU = XSXLUBHDVCS
    XSXALU = XSXLU / XSXUU
    XSXAUL = XSXUL / XSXUU
    XSXALL = XSXLL / XSXUU
    sigmas = T(tf.stack([XSXUU, XSXLU, XSXUL, XSXLL, XSXALU, XSXAUL, XSXALL]))
    gather_nd_idxs = tf.stack(
        [tf.range(sigmas.shape[0], dtype=tf.int32), L - 1], axis=1
    )
    return tf.gather_nd(sigmas, gather_nd_idxs)


@graphtrace.trace_graph
def ff_to_xsx_simplified(reH, imH, reE, imE, reHt, imHt, reEt, imEt, phi, xbj, t, Q2, L, k0):
    depth_vector = tf.ones((reH.shape[0],), dtype=tf.float64)
    nu = Q2 / (2.0 * M * xbj)
    xi = xbj * ((1.0 + (t / (2.0 * Q2))) / (2.0 - xbj + ((xbj * t) / Q2)))
    gamma = sqrt(Q2) / nu
    y = sqrt(Q2) / (gamma * k0)
    k0p = k0 * (1.0 - y)
    t0 = -(4.0 * xi * xi * M * M) / (1.0 - (xi * xi))
    costl = -(1.0 / (sqrt(1.0 + gamma * gamma))) * (1.0 + (y * gamma * gamma / 2.0))
    sintl = (gamma / (sqrt(1.0 + gamma * gamma))) * sqrt(
        1.0 - y - (y * y * gamma * gamma / 4.0)
    )
    sintlp = sintl / (1.0 - y)
    costlp = (costl + y * sqrt(1.0 + gamma * gamma)) / (1.0 - y)

    p0p = M - (t / (2.0 * M))

    # ratio of longitudinal to transverse virtual photon flux
    eps = (1.0 - y - 0.25 * y * y * gamma * gamma) / (
        1.0 - y + 0.5 * y * y + 0.25 * y * y * gamma * gamma
    )

    # angular kinematics of outgoing photon
    cost = -(1 / (sqrt(1 + gamma * gamma))) * (
        1 + (0.5 * gamma * gamma) * ((1 + (t / Q2)) / (1 + ((xbj * t) / (Q2))))
    )
    cost = tf.math.maximum(cost, -1.)
    sint = sqrt(1.0 - cost * cost)

    # outgoing photon energy
    q0p = (sqrt(Q2) / gamma) * (1 + ((xbj * t) / Q2))

    # conversion from GeV to NanoBarn
    jacobian = (1.0 / (2.0 * M * xbj * k0)) * 4.0 * pi
    conversion = (0.1973 * 0.1973) * 10000000 * jacobian

    # ratio of momentum transfer to proton mass
    tau = -t / (4.0 * M * M)

    ###############################################################################
    ## Creates arrays of 4-vector kinematics uses in Bethe Heitler Cross Section ##
    ###############################################################################

    # initial proton 4-momentum
    p = T(
        tf.convert_to_tensor(
            [
                M * depth_vector,
                0.0 * depth_vector,
                0.0 * depth_vector,
                0.0 * depth_vector,
            ]
        )
    )

    # initial lepton 4-momentum
    k = T(
        tf.convert_to_tensor(
            [k0 * depth_vector, k0 * sintl, 0.0 * depth_vector, k0 * costl]
        )
    )

    # final lepton 4-momentum
    kp = T(
        tf.convert_to_tensor(
            [k0p * depth_vector, k0p * sintlp, 0.0 * depth_vector, k0p * costlp]
        )
    )

    # virtual photon 4-momentum
    q = k - kp

    ##################################
    ## Creates four vector products ##
    ##################################
    plp = product(p, p)  # pp
    qq = product(q, q)  # qq
    kk = product(k, k)  # kk
    kkp = product(k, kp)  # kk'
    kq = product(k, q)  # kq
    pk = product(k, p)  # pk
    pkp = product(kp, p)  # pk'

    # sets the Mandelstam variables s which is the center of mass energy
    s = kk + (2 * pk) + plp


    Gamma = (alpha ** 3) / (16.0 * (pi ** 2) * ((s - M * M) ** 2) * sqrt(1.0 + gamma ** 2)* xbj)

    phi = phi * 0.0174532951  # radian conversion

    ffF1, ffF2, ffGM = kelly(-t)

    # Calculation of the Total Unpolarized Bethe-Heitler Cross Section
    XSXUUBHDVCS = Gamma/(Q2*(-t))*cos(phi)*conversion*ffF1*reH
    FUUT = (Gamma / Q2)/(1 - eps)*conversion*(4*((1 - xi * xi) * (reH * reH)))

    XSXUU = XSXUUBHDVCS + FUUT
    return XSXUU


    # Calculates the Polarized Beam Unpolarized Target Coefficients in front of the
    # Elastic Form Factors and Compton Form Factors
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    ALUBHDVCS = -16 * Dplus * (
        2
        * (
            k[:, 1] * Pplus * kp[:, 1] * kminus
            - k[:, 1] * Pplus * kpminus * k[:, 1]
            + k[:, 1] * Pminus * kpplus * k[:, 1]
            - k[:, 1] * Pminus * kp[:, 1] * kplus
            + k[:, 1] * P[:, 1] * kpminus * kplus
            - k[:, 1] * P[:, 1] * kpplus * kminus
        )
        + 2
        * (
            kp[:, 1] * Pplus * qpminus * k[:, 1]
            - kp[:, 1] * Pplus * qp[:, 1] * kminus
            + kp[:, 1] * Pminus * qp[:, 1] * kplus
            - kp[:, 1] * Pminus * qpplus * k[:, 1]
            + kp[:, 1] * P[:, 1] * qpplus * kminus
            - kp[:, 1] * P[:, 1] * qpminus * kplus
            + k[:, 1] * Pplus * qpminus * kp[:, 1]
            - k[:, 1] * Pplus * qp[:, 1] * kpminus
            + k[:, 1] * Pminus * qp[:, 1] * kpplus
            - k[:, 1] * Pminus * qpplus * kp[:, 1]
            + k[:, 1] * P[:, 1] * qpplus * kpminus
            - k[:, 1] * P[:, 1] * qpminus * kpplus
            + 2 * (qpminus * Pplus - qpplus * Pminus) * kkp
        )
    ) * sin(phi) - 16 * Dminus * (-2) * (
        2 * (kminus * kpplus - kplus * kpminus) * Pqp
        + kpminus * kplus * qp[:, 1] * P[:, 1]
        + kpplus * k[:, 1] * qpminus * P[:, 1]
        + kp[:, 1] * kminus * qpplus * P[:, 1]
        - kpplus * kminus * qp[:, 1] * P[:, 1]
        - kp[:, 1] * kplus * qpminus * P[:, 1]
        - kpminus * k[:, 1] * qpplus * P[:, 1]
        + kpminus * kplus * qp[:, 2] * P[:, 2]
        - kpplus * kminus * qp[:, 2] * P[:, 2]
    ) * sin(
        phi
    )
    BLUBHDVCS = -8 * xi * Dplus * (
        2
        * (
            k[:, 1] * dplus * kp[:, 1] * kminus
            - k[:, 1] * dplus * kpminus * k[:, 1]
            + k[:, 1] * dminus * kpplus * k[:, 1]
            - k[:, 1] * dminus * kp[:, 1] * kplus
            + k[:, 1] * d[:, 1] * kpminus * kplus
            - k[:, 1] * d[:, 1] * kpplus * kminus
        )
        + 2
        * (
            kp[:, 1] * dplus * qpminus * k[:, 1]
            - kp[:, 1] * dplus * qp[:, 1] * kminus
            + kp[:, 1] * dminus * qp[:, 1] * kplus
            - kp[:, 1] * dminus * qpplus * k[:, 1]
            + kp[:, 1] * d[:, 1] * qpplus * kminus
            - kp[:, 1] * d[:, 1] * qpminus * kplus
            + k[:, 1] * dplus * qpminus * kp[:, 1]
            - k[:, 1] * dplus * qp[:, 1] * kpminus
            + k[:, 1] * dminus * qp[:, 1] * kpplus
            - k[:, 1] * dminus * qpplus * kp[:, 1]
            + k[:, 1] * d[:, 1] * qpplus * kpminus
            - k[:, 1] * d[:, 1] * qpminus * kpplus
            + 2 * (qpminus * dplus - qpplus * dminus) * kkp
        )
    ) * sin(phi) - 8 * xi * Dminus * (-2) * (
        2 * (kminus * kpplus - kplus * kpminus) * qpd
        + kpminus * kplus * qp[:, 1] * d[:, 1]
        + kpplus * k[:, 1] * qpminus * d[:, 1]
        + kp[:, 1] * kminus * qpplus * d[:, 1]
        - kpplus * kminus * qp[:, 1] * d[:, 1]
        - kp[:, 1] * kplus * qpminus * d[:, 1]
        - kpminus * k[:, 1] * qpplus * d[:, 1]
        + kpminus * kplus * qp[:, 2] * d[:, 2]
        - kpplus * kminus * qp[:, 2] * d[:, 2]
    ) * sin(
        phi
    )
    CLUBHDVCS = 8 * Dplus * (
        2
        * (kp[:, 1] * kpminus * kplus * d[:, 1] - kp[:, 1] * kpplus * kminus * d[:, 1])
        + 2
        * (
            kp[:, 1] * qpminus * kplus * d[:, 1]
            - kp[:, 1] * qpplus * kminus * d[:, 1]
            + k[:, 1] * qpminus * kpplus * d[:, 1]
            - k[:, 1] * qpplus * kpminus * d[:, 1]
        )
    ) * sin(phi) + 8 * Dminus * (-2) * (
        -kpminus * k[:, 1] * qpplus * d[:, 1]
        + kpminus * kplus * qp[:, 1] * d[:, 1]
        + kpplus * k[:, 1] * qpminus * d[:, 1]
        - kpplus * kminus * qp[:, 1] * d[:, 1]
        + kp[:, 1] * kminus * qpplus * d[:, 1]
        - kp[:, 1] * kplus * qpminus * d[:, 1]
        - qp[:, 2] * d[:, 2] * (kpplus * kminus - kpminus * kplus)
    ) * sin(
        phi
    )

    # Calculates the Longitudinally Polarized Coefficients in front of the EFFs
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    ALLBHDVCS = -16 * Dplus * (
        2 * kp[:, 1] * (kp[:, 1] * kminus - kpminus * k[:, 1]) * Pplus
        + 2 * kp[:, 1] * (kpplus * k[:, 1] - kp[:, 1] * kplus) * Pminus
        + 2 * kp[:, 1] * (kpminus * kplus - kpplus * kminus) * P[:, 1]
        + 2
        * (
            kp[:, 1] * (qpminus * k[:, 1] - qp[:, 1] * kminus) * Pplus
            + kp[:, 1] * (qp[:, 1] * kplus - qpplus * k[:, 1]) * Pminus
            + kp[:, 1] * (qpplus * kminus - qpminus * kplus) * P[:, 1]
            + k[:, 1] * (qpminus * kp[:, 1] - qp[:, 1] * kpminus) * Pplus
            + k[:, 1] * (qp[:, 1] * kpplus - qpplus * kp[:, 1]) * Pminus
            + k[:, 1] * (qpplus * kpminus - qpminus * kpplus) * P[:, 1]
            - 2 * kkp * (qpplus * Pminus - qpminus * Pplus)
        )
    ) * cos(phi) - 16 * Dminus * (-2) * (
        2 * Pqp * (kpplus * kminus - kpminus * kplus)
        + P[:, 2] * qp[:, 2] * (kpplus * kminus - kpminus * kplus)
        + P[:, 1]
        * (
            kp[:, 1] * kplus * qpminus
            - kp[:, 1] * kminus * qpplus
            + kpminus * k[:, 1] * qpplus
            - kpminus * kplus * qp[:, 1]
            + kpplus * kminus * qp[:, 1]
            - kpplus * k[:, 1] * qpminus
        )
    ) * cos(
        phi
    )
    BLLBHDVCS = -8 * xi * Dplus * (
        2 * kp[:, 1] * (kp[:, 1] * kminus - kpminus * k[:, 1]) * dplus
        + 2 * kp[:, 1] * (kpplus * k[:, 1] - kp[:, 1] * kplus) * dminus
        + 2 * kp[:, 1] * (kpminus * kplus - kpplus * kminus) * d[:, 1]
        + 2
        * (
            kp[:, 1] * (qpminus * k[:, 1] - qp[:, 1] * kminus) * dplus
            + kp[:, 1] * (qp[:, 1] * kplus - qpplus * k[:, 1]) * dminus
            + kp[:, 1] * d[:, 1] * (qpplus * kminus - qpminus * kplus)
            + k[:, 1] * (qpminus * kp[:, 1] - qp[:, 1] * kpminus) * dplus
            + k[:, 1] * (qp[:, 1] * kpplus - qpplus * kp[:, 1]) * dminus
            + k[:, 1] * (qpplus * kpminus - qpminus * kpplus) * d[:, 1]
            - 2 * kkp * (qpplus * dminus - qpminus * dplus)
        )
    ) * cos(phi) - 8 * xi * Dminus * (-2) * (
        2 * qpd * (kpplus * kminus - kpminus * kplus)
        + d[:, 2] * qp[:, 2] * (kpplus * kminus - kpminus * kplus)
        + d[:, 1]
        * (
            kp[:, 1] * kplus * qpminus
            - kp[:, 1] * kminus * qpplus
            + kpminus * k[:, 1] * qpplus
            - kpminus * kplus * qp[:, 1]
            + kpplus * kminus * qp[:, 1]
            - kpplus * k[:, 1] * qpminus
        )
    ) * cos(
        phi
    )
    CLLBHDVCS = 16 * Dplus * (
        2 * (k[:, 1] * kminus * kpplus * d[:, 1] - k[:, 1] * kpminus * kplus * d[:, 1])
        + 2
        * (
            kp[:, 1] * qpplus * kminus * d[:, 1]
            - kp[:, 1] * qpminus * kplus * d[:, 1]
            + k[:, 1] * qpplus * kpminus * d[:, 1]
            - k[:, 1] * qpminus * kpplus * d[:, 1]
        )
    ) * cos(phi) - 16 * Dminus * (-2) * (
        -d[:, 1]
        * (
            kpminus * kplus * qp[:, 1]
            - kpminus * k[:, 1] * qpplus
            + kpplus * k[:, 1] * qpminus
            - kpplus * kminus * qp[:, 1]
            + kp[:, 1] * kminus * qpplus
            - kp[:, 1] * kplus * qpminus
        )
        + qp[:, 2] * kpplus * kminus * d[:, 2]
        - qp[:, 2] * kpminus * kplus * d[:, 2]
    ) * cos(
        phi
    )

    # Converted Unpolarized Coefficients with the Gamma factor and in nano-barn
    con_AUUBHDVCS = 2 * (Gamma / (Q2 * -t)) * AUUBHDVCS * conversion
    con_BUUBHDVCS = 2 * (Gamma / (Q2 * -t)) * BUUBHDVCS * conversion
    con_CUUBHDVCS = 2 * (Gamma / (Q2 * -t)) * CUUBHDVCS * conversion

    # Converted Longitudinally Polarized Coefficients with the Gamma Factor and in nano-barn
    con_ALLBHDVCS = (Gamma / (Q2 * -t)) * ALLBHDVCS * conversion
    con_BLLBHDVCS = (Gamma / (Q2 * -t)) * BLLBHDVCS * conversion
    con_CLLBHDVCS = (Gamma / (Q2 * -t)) * CLLBHDVCS * conversion

    # Converted Longitudinally Polarized Beam Unpolarized Target Coefficients with
    # the Gamma Factor and in nano-barn
    con_ALUBHDVCS = (Gamma / (Q2 * -t)) * ALUBHDVCS * conversion
    con_BLUBHDVCS = (Gamma / (Q2 * -t)) * BLUBHDVCS * conversion
    con_CLUBHDVCS = (Gamma / (Q2 * -t)) * CLUBHDVCS * conversion

    # Converted Longitudinally Polarized Target Unpolarized Beam Coefficients with
    # the Gamma Factor and in nano-barn
    con_AULBHDVCS = (Gamma / (Q2 * -t)) * AULBHDVCS * conversion
    con_BULBHDVCS = (Gamma / (Q2 * -t)) * BULBHDVCS * conversion
    con_CULBHDVCS = (Gamma / (Q2 * -t)) * CULBHDVCS * conversion

    # Unpolarized Coefficients multiplied by the Form Factors
    bhdvcsAUU = con_AUUBHDVCS * (ffF1 * reH)
    bhdvcsBUU = con_BUUBHDVCS * (ffGM * reH)
    bhdvcsCUU = con_CUUBHDVCS * (ffGM * reH)

    # Polarized Coefficients multiplied by the Form Factors
    bhdvcsALU = con_ALUBHDVCS * (ffF1 * imH)
    bhdvcsBLU = con_BLUBHDVCS * (ffGM * imH)
    bhdvcsCLU = con_CLUBHDVCS * (ffGM * imH)

    # Unpolarized Beam Polarized Target Coefficients multiplied by the Form Factors
    bhdvcsAUL = con_AULBHDVCS * (ffF1 * imHt - xi * ffF1 * imEt + tau * ffF2 * imEt)
    bhdvcsBUL = con_BULBHDVCS * (ffGM * imHt)
    bhdvcsCUL = con_CULBHDVCS * (ffGM * (imH + imE))

    # Polarized Beam Unpolarized Target Coefficients multiplied by the Form Factors
    bhdvcsALL = con_ALLBHDVCS * (ffF1 * reHt - xi * ffF1 * reEt + tau * ffF2 * reEt)
    bhdvcsBLL = con_BLLBHDVCS * (ffGM * reHt)
    bhdvcsCLL = con_CLLBHDVCS * (ffGM * (reH + reE))

    # Calculation of the Total Unpolarized Bethe-Heitler Cross Section
    XSXUUBHDVCS = bhdvcsAUU + bhdvcsBUU + bhdvcsCUU
    XSXLLBHDVCS = bhdvcsALL + bhdvcsBLL + bhdvcsCLL
    XSXULBHDVCS = bhdvcsAUL + bhdvcsBUL + bhdvcsCUL
    XSXLUBHDVCS = bhdvcsALU + bhdvcsBLU + bhdvcsCLU
    FUUT = (
        (Gamma / (Q2))
        / (1 - eps)
        * conversion
        * (
            4
            * (
                (1 - xi * xi) 
                * (reH * reH + imH * imH + reHt * reHt + imHt * imHt)
                + ((t0 - t) / (2 * M * M))
                * (
                    reE * reE
                    + imE * imE
                    + xi * xi * reEt * reEt
                    + xi * xi * imEt * imEt
                )
                - ((2 * xi * xi) / (1 - xi * xi))
                * (reH * reE + imH * imE + reHt * reEt + imHt * imEt)
            )
        )
    )

    XSXUU = XSXUUBHDVCS
    XSXLL = XSXLLBHDVCS + XSXLLBH
    XSXUL = XSXULBHDVCS
    XSXLU = XSXLUBHDVCS
    XSXALU = XSXLU / XSXUU
    XSXAUL = XSXUL / XSXUU
    XSXALL = XSXLL / XSXUU
    sigmas = T(tf.stack([XSXUU, XSXLU, XSXUL, XSXLL, XSXALU, XSXAUL, XSXALL]))
    gather_nd_idxs = tf.stack(
        [tf.range(sigmas.shape[0], dtype=tf.int32), L - 1], axis=1
    )
    return tf.gather_nd(sigmas, gather_nd_idxs)

@graphtrace.trace_graph
def ff_to_xsx_new(reH, imH, reE, imE, reHt, imHt, reEt, imEt, phi, xbj, t, Q2, L, k0):
    """ Calculation of cross sections from form factor predictions.
    Args:
        reH (tf.Tensor) : Tensor of shape (batch_size,). 0 index of model output. dtype tf.float32
        imH (tf.Tensor) : Tensor of shape (batch_size,). 1 index of model output. dtype tf.float32
        reE (tf.Tensor) : Tensor of shape (batch_size,). 2 index of model output. dtype tf.float32
        imE (tf.Tensor) : Tensor of shape (batch_size,). 3 index of model output. dtype tf.float32
        reHt (tf.Tensor) : Tensor of shape (batch_size,). 4 index of model output. dtype tf.float32
        imHt (tf.Tensor) : Tensor of shape (batch_size,). 5 index of model output. dtype tf.float32
        reEt (tf.Tensor) : Tensor of shape (batch_size,). 6 index of model output. dtype tf.float32
        imEt (tf.Tensor) : Tensor of shape (batch_size,). 7 index of model output. dtype tf.float32
        xbj (tf.Tensor) : Tensor of shape (batch_size,). 0 index of kinematic input. dtype tf.float32
        t (tf.Tensor) : Tensor of shape (batch_size,). 1 index of kinematic input. dtype tf.float32
        Q2 (tf.Tensor) : Tensor of shape (batch_size,). 2 index of kinematic input. dtype tf.float32
        phi (tf.Tensor) : Tensor of shape (batch_size,). 3 index of kinematic input. dtype tf.float32
        L (tf.Tensor) : Tensor of shape (batch_size,). 0 index of sigma_true label. dtype *tf.int32*
    
    Returns:
        Calculated cross section tf.Tensor of shape (batch_size,)
    """
    depth_vector = tf.ones((reH.shape[0],), dtype=tf.float64)

    ###################################
    ## Secondary Kinematic Variables ##
    ###################################

    # energy of the virtual photon
    nu = Q2 / (2.0 * M * xbj)

    # skewness parameter set by xbj, t, and Q^2
    xi = xbj * ((1.0 + (t / (2.0 * Q2))) / (2.0 - xbj + ((xbj * t) / Q2)))

    # gamma variable ratio of virtuality to energy of virtual photon
    gamma = sqrt(Q2) / nu

    # fractional energy of virtual photon
    y = sqrt(Q2) / (gamma * k0)

    # final lepton energy
    k0p = k0 * (1.0 - y)

    # minimum t value
    t0 = -(4.0 * xi * xi * M * M) / (1.0 - (xi * xi))

    # Lepton Angle Kinematics of initial lepton
    costl = -(1.0 / (sqrt(1.0 + gamma * gamma))) * (1.0 + (y * gamma * gamma / 2.0))
    sintl = (gamma / (sqrt(1.0 + gamma * gamma))) * sqrt(
        1.0 - y - (y * y * gamma * gamma / 4.0)
    )

    # Lepton Angle Kinematics of final lepton
    sintlp = sintl / (1.0 - y)
    costlp = (costl + y * sqrt(1.0 + gamma * gamma)) / (1.0 - y)

    # final proton energy
    p0p = M - (t / (2.0 * M))

    # ratio of longitudinal to transverse virtual photon flux
    eps = (1.0 - y - 0.25 * y * y * gamma * gamma) / (
        1.0 - y + 0.5 * y * y + 0.25 * y * y * gamma * gamma
    )

    # angular kinematics of outgoing photon
    cost = -(1 / (sqrt(1 + gamma * gamma))) * (
        1 + (0.5 * gamma * gamma) * ((1 + (t / Q2)) / (1 + ((xbj * t) / (Q2))))
    )
    cost = tf.math.maximum(cost, -1.)
    sint = sqrt(1.0 - cost * cost)

    # outgoing photon energy
    q0p = (sqrt(Q2) / gamma) * (1 + ((xbj * t) / Q2))

    # conversion from GeV to NanoBarn
    jacobian = 2.*pi
    conversion = (0.1973 * 0.1973) * 10000000 * jacobian / 4

    # ratio of momentum transfer to proton mass
    tau = -t / (4.0 * M * M)

    ###############################################################################
    ## Creates arrays of 4-vector kinematics uses in Bethe Heitler Cross Section ##
    ###############################################################################

    # initial proton 4-momentum
    p = T(
        tf.convert_to_tensor(
            [
                M * depth_vector,
                0.0 * depth_vector,
                0.0 * depth_vector,
                0.0 * depth_vector,
            ]
        )
    )

    # initial lepton 4-momentum
    k = T(
        tf.convert_to_tensor(
            [k0 * depth_vector, k0 * sintl * depth_vector, 0.0 * depth_vector, k0 * costl * depth_vector]
        )
    )

    # final lepton 4-momentum
    kp = T(
        tf.convert_to_tensor(
            [k0p * depth_vector, k0p * sintlp * depth_vector, 0.0 * depth_vector, k0p * costlp* depth_vector]
        )
    )

    # virtual photon 4-momentum
    q = k - kp

    ##################################
    ## Creates four vector products ##
    ##################################
    plp = product(p, p)  # pp
    qq = product(q, q)  # qq
    kk = product(k, k)  # kk
    kkp = product(k, kp)  # kk'
    kq = product(k, q)  # kq
    pk = product(k, p)  # pk
    pkp = product(kp, p)  # pk'

    # sets the Mandelstam variables s which is the center of mass energy
    s = kk + (2 * pk) + plp

    # the Gamma factor in front of the cross section
    Gamma = (alpha ** 3) / (
        16.0 * (pi ** 2) * ((s - M * M) ** 2) * sqrt(1.0 + gamma ** 2) * xbj
    )

    phi = phi * 0.0174532951  # radian conversion

    # final real photon 4-momentum
    qp = T(
        tf.convert_to_tensor(
            [
                q0p * depth_vector,
                q0p * sint * T(cos(phi)),
                q0p * sint * T(sin(phi)),
                q0p * cost * depth_vector,
            ]
        )
    )

    # momentum transfer Δ from the initial proton to the final proton
    d = q - qp

    # final proton momentum
    pp = p + d

    # average initial proton momentum
    P = 0.5 * (p + pp)

    # 4-vector products of variables multiplied by spin vectors
    ppSL = ((M) / (sqrt(1.0 + gamma ** 2))) * (
        xbj * (1.0 - (t / Q2)) - (t / (2.0 * M ** 2))
    )
    kSL = (
        ((Q2) / (sqrt(1.0 + gamma ** 2)))
        * (1.0 + 0.5 * y * gamma ** 2)
        * (1.0 / (2.0 * M * xbj * y))
    )
    kpSL = (
        ((Q2) / (sqrt(1 + gamma ** 2)))
        * (1 - y - 0.5 * y * gamma ** 2)
        * (1.0 / (2.0 * M * xbj * y))
    )

    # 4-vector products denoted in the paper by the commented symbols
    kd = product(k, d)  # dΔ
    kpd = product(kp, d)  # k'Δ
    kP = product(k, P)  # kP
    kpP = product(kp, P)  # k'P
    kqp = product(k, qp)  # kq'
    kpqp = product(kp, qp)  # k'q'
    dd = product(d, d)  # ΔΔ
    Pq = product(P, q)  # Pq
    Pqp = product(P, qp)  # Pq'
    qd = product(q, d)  # qΔ
    qpd = product(qp, d)  # q'Δ

    # transverse vector products
    kkT = tproduct(k, k)
    kqpT = tproduct(k, qp)
    kkpT = tproduct(k, kp)
    ddT = tproduct(d, d)
    kdT = tproduct(k, d)
    kpqpT = tproduct(kp, qp)
    qpdT = tproduct(qp, d)
    kPT = tproduct(k, P)
    kpPT = tproduct(kp, P)
    qpPT = tproduct(qp, P)
    kpdT = tproduct(kp, d)

    # light cone variables expressed as A^{+-} = 1/sqrt(2)(A^{0} +- A^{3})
    inv_root_2 = 1 / sqrt(2.0)
    inv_root_2 = tf.dtypes.cast(inv_root_2, tf.float64)
    kplus = T(inv_root_2 * (k[..., 0] + k[..., 3]))
    kpplus = T(inv_root_2 * (kp[..., 0] + kp[..., 3]))
    kminus = T(inv_root_2 * (k[..., 0] - k[..., 3]))
    kpminus = T(inv_root_2 * (kp[..., 0] - kp[..., 3]))
    qplus = T(inv_root_2 * (q[..., 0] + q[..., 3]))
    qpplus = T(inv_root_2 * (qp[..., 0] + qp[..., 3]))
    qminus = T(inv_root_2 * (q[..., 0] - q[..., 3]))
    qpminus = T(inv_root_2 * (qp[..., 0] - qp[..., 3]))
    Pplus = T(inv_root_2 * (P[..., 0] + P[..., 3]))
    Pminus = T(inv_root_2 * (P[..., 0] - P[..., 3]))
    dplus = T(inv_root_2 * (d[..., 0] + d[..., 3]))
    dminus = T(inv_root_2 * (d[..., 0] - d[..., 3]))

    # expresssions used that appear in coefficient calculations
    Dplus = (1 / (2 * kpqp)) - (1 / (2 * kqp))
    Dminus = (1 / (2 * kpqp)) + (1 / (2 * kqp))

    # calculates BH
    AUUBH = ((16.0 * M * M) / (kqp * kpqp)) * (
        (4.0 * tau * (kP * kP + kpP * kpP)) - ((tau + 1.0) * (kd * kd + kpd * kpd))
    )
    BUUBH = ((32.0 * M * M) / (kqp * kpqp)) * (kd * kd + kpd * kpd)

    # calculates BHLL
    ALLBH = -((16.0 * M * M) / (kqp * kpqp)) * (
        (ppSL / M) * ((kpd * kpd - kd * kd) - 2.0 * tau * (kpd * pkp - kd * pk))
        + t * (kSL / M) * (1.0 + tau) * kd
        - t * (kpSL / M) * (1.0 + tau) * kpd
    )
    BLLBH = ((16.0 * M * M) / (kqp * kpqp)) * (
        (ppSL / M) * (kpd * kpd - kd * kd) + t * (kSL / M) * kd - t * (kpSL / M) * kpd
    )

    # converted Unpolarized Coefficients with the Gamma factor and in nano-barn
    con_AUUBH = (Gamma / t ** 2) * AUUBH * conversion * 2
    con_BUUBH = (Gamma / t ** 2) * BUUBH * conversion * 2

    # converted Longitudinally Polarized Coefficients with the Gamma Factor and in nano-barn
    con_ALLBH = (Gamma / t ** 2) * ALLBH * conversion * 2
    con_BLLBH = (Gamma / t ** 2) * BLLBH * conversion * 2

    ffF1, ffF2, ffGM = kelly(-t)

    # unpolarized Coefficients multiplied by the Form Factors calculated in form_factor.py
    # we use the Galster Form Factors as approximations
    bhAUU = con_AUUBH * ((ffF1 * ffF1) + (tau * ffF2 * ffF2))
    bhBUU = con_BUUBH * (tau * ffGM * ffGM)

    # polarized Coefficients multiplied by the Form Factors calculated in form_factor.py
    # using the Galster Form Factor Model
    bhALL = con_ALLBH * (ffF2 * ffGM)
    bhBLL = con_BLLBH * (ffGM * ffGM)

    # Calculation of the Total Unpolarized Bethe-Heitler Cross Section
    XSXUUBH = bhAUU + bhBUU
    XSXLLBH = bhALL + bhBLL

    # Calculates the Unpolarized Coefficients in front of the Elastic Form Factors and
    # Compton Form Factors
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    AUUBHDVCS = 16 * Dplus * (
        (kqpT - 2 * kkT - 2 * kqp) * kpP
        + (2 * kpqp - 2 * kkpT - kpqpT) * kP
        + kpqp * kPT
        + kqp * kpPT
        - 2 * kkp * kPT
    ) * cos(phi) - 16 * Dminus * (
        (2 * kkp - kpqpT - kkpT) * Pqp + 2 * kkp * qpPT - kpqp * kPT - kqp * kpPT
    ) * cos(
        phi
    )
    BUUBHDVCS = 8 * xi * Dplus * (
        (kqpT - 2 * kkT - 2 * kqp) * kpd
        + (2 * kpqp - 2 * kkpT - kpqpT) * kd
        + kpqp * kdT
        + kqp * kpdT
        - 2 * kkp * kdT
    ) * cos(phi) - 8 * xi * Dminus * (
        (2 * kkp - kpqpT - kkpT) * qpd + 2 * kkp * qpdT - kpqp * kdT - kqp * kpdT
    ) * cos(
        phi
    )
    CUUBHDVCS = 8 * Dplus * (
        (2 * kkp * kdT - kpqp * kdT - kqp * kpdT)
        + 2 * xi * (2 * kkp * kPT - kpqp * kPT - kqp * kpPT)
    ) * cos(phi) - 8 * Dminus * (
        (kkp * qpdT - kpqp * kdT - kqp * kpdT)
        + 2 * xi * (kkp * qpPT - kpqp * kPT - kqp * kpPT)
    ) * cos(
        phi
    )

    # Calculates the Unpolarized Beam Polarized Target Coefficients in front of the
    # Elastic Form Factors and Compton Form Factors
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    AULBHDVCS = -16 * Dplus * (
        kpP * (2 * kkT - kqpT + 2 * kqp)
        + kP * (2 * kkpT - kpqpT + 2 * kpqp)
        + 2 * kkp * kPT
        - kpqp * kPT
        - kqp * kpPT
    ) * sin(phi) - 16 * Dminus * (
        Pqp * (kkpT + kpqpT - 2 * kkp) - (2 * kkp * qpPT - kpqp * kPT - kqp * kpPT)
    ) * sin(
        phi
    )
    BULBHDVCS = -8 * xi * Dplus * (
        kpd * (2 * kkT - kqpT + 2 * kqp)
        + kd * (2 * kkpT - kpqpT + 2 * kpqp)
        + 2 * kkp * kdT
        - kpqp * kdT
        - kqp * kpdT
    ) * sin(phi) - 8 * xi * Dminus * (
        qpd * (kkpT + kpqpT - 2 * kkp) - (2 * kkp * qpdT - kpqp * kdT - kqp * kpdT)
    ) * sin(
        phi
    )
    CULBHDVCS = -4 * Dplus * (
        2 * (2 * kkp * kdT - kpqp * kdT - kqp * kpdT)
        + 4 * xi * (2 * kkp * kPT - kpqp * kPT - kqp * kpPT)
    ) * sin(phi) - 4 * Dminus * (
        -2 * (kkp * qpdT - kpqp * kdT - kqp * kpdT)
        - 4 * xi * (kkp * qpPT - kpqp * kPT - kqp * kpPT)
    ) * sin(
        phi
    )

    # Calculates the Polarized Beam Unpolarized Target Coefficients in front of the
    # Elastic Form Factors and Compton Form Factors
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    ALUBHDVCS = -16 * Dplus * (
        2
        * (
            k[:, 1] * Pplus * kp[:, 1] * kminus
            - k[:, 1] * Pplus * kpminus * k[:, 1]
            + k[:, 1] * Pminus * kpplus * k[:, 1]
            - k[:, 1] * Pminus * kp[:, 1] * kplus
            + k[:, 1] * P[:, 1] * kpminus * kplus
            - k[:, 1] * P[:, 1] * kpplus * kminus
        )
        + 2
        * (
            kp[:, 1] * Pplus * qpminus * k[:, 1]
            - kp[:, 1] * Pplus * qp[:, 1] * kminus
            + kp[:, 1] * Pminus * qp[:, 1] * kplus
            - kp[:, 1] * Pminus * qpplus * k[:, 1]
            + kp[:, 1] * P[:, 1] * qpplus * kminus
            - kp[:, 1] * P[:, 1] * qpminus * kplus
            + k[:, 1] * Pplus * qpminus * kp[:, 1]
            - k[:, 1] * Pplus * qp[:, 1] * kpminus
            + k[:, 1] * Pminus * qp[:, 1] * kpplus
            - k[:, 1] * Pminus * qpplus * kp[:, 1]
            + k[:, 1] * P[:, 1] * qpplus * kpminus
            - k[:, 1] * P[:, 1] * qpminus * kpplus
            + 2 * (qpminus * Pplus - qpplus * Pminus) * kkp
        )
    ) * sin(phi) - 16 * Dminus * (-2) * (
        2 * (kminus * kpplus - kplus * kpminus) * Pqp
        + kpminus * kplus * qp[:, 1] * P[:, 1]
        + kpplus * k[:, 1] * qpminus * P[:, 1]
        + kp[:, 1] * kminus * qpplus * P[:, 1]
        - kpplus * kminus * qp[:, 1] * P[:, 1]
        - kp[:, 1] * kplus * qpminus * P[:, 1]
        - kpminus * k[:, 1] * qpplus * P[:, 1]
        + kpminus * kplus * qp[:, 2] * P[:, 2]
        - kpplus * kminus * qp[:, 2] * P[:, 2]
    ) * sin(
        phi
    )
    BLUBHDVCS = -8 * xi * Dplus * (
        2
        * (
            k[:, 1] * dplus * kp[:, 1] * kminus
            - k[:, 1] * dplus * kpminus * k[:, 1]
            + k[:, 1] * dminus * kpplus * k[:, 1]
            - k[:, 1] * dminus * kp[:, 1] * kplus
            + k[:, 1] * d[:, 1] * kpminus * kplus
            - k[:, 1] * d[:, 1] * kpplus * kminus
        )
        + 2
        * (
            kp[:, 1] * dplus * qpminus * k[:, 1]
            - kp[:, 1] * dplus * qp[:, 1] * kminus
            + kp[:, 1] * dminus * qp[:, 1] * kplus
            - kp[:, 1] * dminus * qpplus * k[:, 1]
            + kp[:, 1] * d[:, 1] * qpplus * kminus
            - kp[:, 1] * d[:, 1] * qpminus * kplus
            + k[:, 1] * dplus * qpminus * kp[:, 1]
            - k[:, 1] * dplus * qp[:, 1] * kpminus
            + k[:, 1] * dminus * qp[:, 1] * kpplus
            - k[:, 1] * dminus * qpplus * kp[:, 1]
            + k[:, 1] * d[:, 1] * qpplus * kpminus
            - k[:, 1] * d[:, 1] * qpminus * kpplus
            + 2 * (qpminus * dplus - qpplus * dminus) * kkp
        )
    ) * sin(phi) + 8 * xi * Dminus * (
        2 * (kminus * kpplus - kplus * kpminus) * qpd
        + kpminus * kplus * qp[:, 1] * d[:, 1]
        + kpplus * k[:, 1] * qpminus * d[:, 1]
        + kp[:, 1] * kminus * qpplus * d[:, 1]
        - kpplus * kminus * qp[:, 1] * d[:, 1]
        - kp[:, 1] * kplus * qpminus * d[:, 1]
        - kpminus * k[:, 1] * qpplus * d[:, 1]
        + kpminus * kplus * qp[:, 2] * d[:, 2]
        - kpplus * kminus * qp[:, 2] * d[:, 2]
    ) * sin(
        phi
    )
    CLUBHDVCS = -8 * Dplus * (
        2
        * (kp[:, 1] * kpminus * kplus * d[:, 1] - kp[:, 1] * kpplus * kminus * d[:, 1])
        + 2
        * (
            kp[:, 1] * qpminus * kplus * d[:, 1]
            - kp[:, 1] * qpplus * kminus * d[:, 1]
            + k[:, 1] * qpminus * kpplus * d[:, 1]
            - k[:, 1] * qpplus * kpminus * d[:, 1]
        )
    ) * sin(phi) - 8 * Dminus * (-2) * (
        -kpminus * k[:, 1] * qpplus * d[:, 1]
        + kpminus * kplus * qp[:, 1] * d[:, 1]
        + kpplus * k[:, 1] * qpminus * d[:, 1]
        - kpplus * kminus * qp[:, 1] * d[:, 1]
        + kp[:, 1] * kminus * qpplus * d[:, 1]
        - kp[:, 1] * kplus * qpminus * d[:, 1]
        - qp[:, 2] * d[:, 2] * (kpplus * kminus - kpminus * kplus)
    ) * sin(
        phi
    )

    # Calculates the Longitudinally Polarized Coefficients in front of the EFFs
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    ALLBHDVCS = -16 * Dplus * (
        2 * kp[:, 1] * (kp[:, 1] * kminus - kpminus * k[:, 1]) * Pplus
        + 2 * kp[:, 1] * (kpplus * k[:, 1] - kp[:, 1] * kplus) * Pminus
        + 2 * kp[:, 1] * (kpminus * kplus - kpplus * kminus) * P[:, 1]
        + 2
        * (
            kp[:, 1] * (qpminus * k[:, 1] - qp[:, 1] * kminus) * Pplus
            + kp[:, 1] * (qp[:, 1] * kplus - qpplus * k[:, 1]) * Pminus
            + kp[:, 1] * (qpplus * kminus - qpminus * kplus) * P[:, 1]
            + k[:, 1] * (qpminus * kp[:, 1] - qp[:, 1] * kpminus) * Pplus
            + k[:, 1] * (qp[:, 1] * kpplus - qpplus * kp[:, 1]) * Pminus
            + k[:, 1] * (qpplus * kpminus - qpminus * kpplus) * P[:, 1]
            - 2 * kkp * (qpplus * Pminus - qpminus * Pplus)
        )
    ) * cos(phi) - 16 * Dminus * (-2) * (
        2 * Pqp * (kpplus * kminus - kpminus * kplus)
        + P[:, 2] * qp[:, 2] * (kpplus * kminus - kpminus * kplus)
        + P[:, 1]
        * (
            kp[:, 1] * kplus * qpminus
            - kp[:, 1] * kminus * qpplus
            + kpminus * k[:, 1] * qpplus
            - kpminus * kplus * qp[:, 1]
            + kpplus * kminus * qp[:, 1]
            - kpplus * k[:, 1] * qpminus
        )
    ) * cos(
        phi
    )
    BLLBHDVCS = -8 * xi * Dplus * (
        2 * kp[:, 1] * (kp[:, 1] * kminus - kpminus * k[:, 1]) * dplus
        + 2 * kp[:, 1] * (kpplus * k[:, 1] - kp[:, 1] * kplus) * dminus
        + 2 * kp[:, 1] * (kpminus * kplus - kpplus * kminus) * d[:, 1]
        + 2
        * (
            kp[:, 1] * (qpminus * k[:, 1] - qp[:, 1] * kminus) * dplus
            + kp[:, 1] * (qp[:, 1] * kplus - qpplus * k[:, 1]) * dminus
            + kp[:, 1] * d[:, 1] * (qpplus * kminus - qpminus * kplus)
            + k[:, 1] * (qpminus * kp[:, 1] - qp[:, 1] * kpminus) * dplus
            + k[:, 1] * (qp[:, 1] * kpplus - qpplus * kp[:, 1]) * dminus
            + k[:, 1] * (qpplus * kpminus - qpminus * kpplus) * d[:, 1]
            - 2 * kkp * (qpplus * dminus - qpminus * dplus)
        )
    ) * cos(phi) - 8 * xi * Dminus * (-2) * (
        2 * qpd * (kpplus * kminus - kpminus * kplus)
        + d[:, 2] * qp[:, 2] * (kpplus * kminus - kpminus * kplus)
        + d[:, 1]
        * (
            kp[:, 1] * kplus * qpminus
            - kp[:, 1] * kminus * qpplus
            + kpminus * k[:, 1] * qpplus
            - kpminus * kplus * qp[:, 1]
            + kpplus * kminus * qp[:, 1]
            - kpplus * k[:, 1] * qpminus
        )
    ) * cos(
        phi
    )
    CLLBHDVCS = 16 * Dplus * (
        2 * (k[:, 1] * kminus * kpplus * d[:, 1] - k[:, 1] * kpminus * kplus * d[:, 1])
        + 2
        * (
            kp[:, 1] * qpplus * kminus * d[:, 1]
            - kp[:, 1] * qpminus * kplus * d[:, 1]
            + k[:, 1] * qpplus * kpminus * d[:, 1]
            - k[:, 1] * qpminus * kpplus * d[:, 1]
        )
    ) * cos(phi) - 16 * Dminus * (-2) * (
        -d[:, 1]
        * (
            kpminus * kplus * qp[:, 1]
            - kpminus * k[:, 1] * qpplus
            + kpplus * k[:, 1] * qpminus
            - kpplus * kminus * qp[:, 1]
            + kp[:, 1] * kminus * qpplus
            - kp[:, 1] * kplus * qpminus
        )
        + qp[:, 2] * kpplus * kminus * d[:, 2]
        - qp[:, 2] * kpminus * kplus * d[:, 2]
    ) * cos(
        phi
    )

    # Converted Unpolarized Coefficients with the Gamma factor and in nano-barn
    con_AUUBHDVCS = (Gamma / (Q2 * t)) * AUUBHDVCS * conversion
    con_BUUBHDVCS = (Gamma / (Q2 * t)) * BUUBHDVCS * conversion
    con_CUUBHDVCS = (Gamma / (Q2 * t)) * CUUBHDVCS * conversion

    # Converted Longitudinally Polarized Coefficients with the Gamma Factor and in nano-barn
    con_ALLBHDVCS = (Gamma / (Q2 * -t)) * ALLBHDVCS * conversion
    con_BLLBHDVCS = (Gamma / (Q2 * -t)) * BLLBHDVCS * conversion
    con_CLLBHDVCS = (Gamma / (Q2 * -t)) * CLLBHDVCS * conversion

    # Converted Longitudinally Polarized Beam Unpolarized Target Coefficients with
    # the Gamma Factor and in nano-barn
    con_ALUBHDVCS = (Gamma / (Q2 * -t)) * ALUBHDVCS * conversion
    con_BLUBHDVCS = (Gamma / (Q2 * -t)) * BLUBHDVCS * conversion
    con_CLUBHDVCS = (Gamma / (Q2 * -t)) * CLUBHDVCS * conversion

    # Converted Longitudinally Polarized Target Unpolarized Beam Coefficients with
    # the Gamma Factor and in nano-barn
    con_AULBHDVCS = (Gamma / (Q2 * -t)) * AULBHDVCS * conversion
    con_BULBHDVCS = (Gamma / (Q2 * -t)) * BULBHDVCS * conversion
    con_CULBHDVCS = (Gamma / (Q2 * -t)) * CULBHDVCS * conversion

    # Unpolarized Coefficients multiplied by the Form Factors
    bhdvcsAUU = con_AUUBHDVCS * (ffF1 * reH + tau * ffF2 * reE)
    bhdvcsBUU = con_BUUBHDVCS * (ffGM * (reH + reE))
    bhdvcsCUU = con_CUUBHDVCS * (ffGM * reHt)

    # Polarized Coefficients multiplied by the Form Factors
    bhdvcsALU = con_ALUBHDVCS * (ffF1 * imH + tau * ffF2 * imE)
    bhdvcsBLU = con_BLUBHDVCS * (ffGM * (imH + imE))
    bhdvcsCLU = con_CLUBHDVCS * (ffGM * imHt)

    # Unpolarized Beam Polarized Target Coefficients multiplied by the Form Factors
    bhdvcsAUL = con_AULBHDVCS * (ffF1 * imHt - xi * ffF1 * imEt + tau * ffF2 * imEt)
    bhdvcsBUL = con_BULBHDVCS * (ffGM * imHt)
    bhdvcsCUL = con_CULBHDVCS * (ffGM * (imH + imE))

    # Polarized Beam Unpolarized Target Coefficients multiplied by the Form Factors
    bhdvcsALL = con_ALLBHDVCS * (ffF1 * reHt - xi * ffF1 * reEt + tau * ffF2 * reEt)
    bhdvcsBLL = con_BLLBHDVCS * (ffGM * reHt)
    bhdvcsCLL = con_CLLBHDVCS * (ffGM * (reH + reE))

    # Calculation of the Total Unpolarized Bethe-Heitler Cross Section
    XSXUUBHDVCS = bhdvcsAUU + bhdvcsBUU + bhdvcsCUU
    XSXLLBHDVCS = bhdvcsALL + bhdvcsBLL + bhdvcsCLL
    XSXULBHDVCS = bhdvcsAUL + bhdvcsBUL + bhdvcsCUL
    XSXLUBHDVCS = bhdvcsALU + bhdvcsBLU + bhdvcsCLU
    FUUT = (
        (Gamma / (Q2))
        / (1 - eps)
        * conversion
        * (
            4
            * (
                (1 - xi * xi) 
                * (reH * reH + imH * imH + reHt * reHt + imHt * imHt)
                + ((t0 - t) / (2 * M * M))
                * (
                    reE * reE
                    + imE * imE
                    + xi * xi * reEt * reEt
                    + xi * xi * imEt * imEt
                )
                - ((2 * xi * xi) / (1 - xi * xi))
                * (reH * reE + imH * imE + reHt * reEt + imHt * imEt)
            )
        )
    ) * 4

    XSXUU = XSXUUBHDVCS + FUUT + XSXUUBH
    XSXLL = XSXLLBHDVCS + XSXLLBH
    XSXUL = XSXULBHDVCS
    XSXLU = XSXLUBHDVCS
    XSXALU = XSXLU / XSXUU
    XSXAUL = XSXUL / XSXUU
    XSXALL = XSXLL / XSXUU
    sigmas = T(tf.stack([XSXUU, XSXLU, XSXUL, XSXLL, XSXALU, XSXAUL, XSXALL]))
    gather_nd_idxs = tf.stack(
        [tf.range(sigmas.shape[0], dtype=tf.int32), L - 1], axis=1
    )
    return tf.gather_nd(sigmas, gather_nd_idxs)

@graphtrace.trace_graph
def ff_to_xsx_new_no_bh(reH, imH, reE, imE, reHt, imHt, reEt, imEt, phi, xbj, t, Q2, L, k0):
    """ Calculation of cross sections from form factor predictions.
    Args:
        reH (tf.Tensor) : Tensor of shape (batch_size,). 0 index of model output. dtype tf.float32
        imH (tf.Tensor) : Tensor of shape (batch_size,). 1 index of model output. dtype tf.float32
        reE (tf.Tensor) : Tensor of shape (batch_size,). 2 index of model output. dtype tf.float32
        imE (tf.Tensor) : Tensor of shape (batch_size,). 3 index of model output. dtype tf.float32
        reHt (tf.Tensor) : Tensor of shape (batch_size,). 4 index of model output. dtype tf.float32
        imHt (tf.Tensor) : Tensor of shape (batch_size,). 5 index of model output. dtype tf.float32
        reEt (tf.Tensor) : Tensor of shape (batch_size,). 6 index of model output. dtype tf.float32
        imEt (tf.Tensor) : Tensor of shape (batch_size,). 7 index of model output. dtype tf.float32
        xbj (tf.Tensor) : Tensor of shape (batch_size,). 0 index of kinematic input. dtype tf.float32
        t (tf.Tensor) : Tensor of shape (batch_size,). 1 index of kinematic input. dtype tf.float32
        Q2 (tf.Tensor) : Tensor of shape (batch_size,). 2 index of kinematic input. dtype tf.float32
        phi (tf.Tensor) : Tensor of shape (batch_size,). 3 index of kinematic input. dtype tf.float32
        L (tf.Tensor) : Tensor of shape (batch_size,). 0 index of sigma_true label. dtype *tf.int32*
    
    Returns:
        Calculated cross section tf.Tensor of shape (batch_size,)
    """
    depth_vector = tf.ones((reH.shape[0],), dtype=tf.float64)

    ###################################
    ## Secondary Kinematic Variables ##
    ###################################

    # energy of the virtual photon
    nu = Q2 / (2.0 * M * xbj)

    # skewness parameter set by xbj, t, and Q^2
    xi = xbj * ((1.0 + (t / (2.0 * Q2))) / (2.0 - xbj + ((xbj * t) / Q2)))

    # gamma variable ratio of virtuality to energy of virtual photon
    gamma = sqrt(Q2) / nu

    # fractional energy of virtual photon
    y = sqrt(Q2) / (gamma * k0)

    # final lepton energy
    k0p = k0 * (1.0 - y)

    # minimum t value
    t0 = -(4.0 * xi * xi * M * M) / (1.0 - (xi * xi))

    # Lepton Angle Kinematics of initial lepton
    costl = -(1.0 / (sqrt(1.0 + gamma * gamma))) * (1.0 + (y * gamma * gamma / 2.0))
    sintl = (gamma / (sqrt(1.0 + gamma * gamma))) * sqrt(
        1.0 - y - (y * y * gamma * gamma / 4.0)
    )

    # Lepton Angle Kinematics of final lepton
    sintlp = sintl / (1.0 - y)
    costlp = (costl + y * sqrt(1.0 + gamma * gamma)) / (1.0 - y)

    # final proton energy
    p0p = M - (t / (2.0 * M))

    # ratio of longitudinal to transverse virtual photon flux
    eps = (1.0 - y - 0.25 * y * y * gamma * gamma) / (
        1.0 - y + 0.5 * y * y + 0.25 * y * y * gamma * gamma
    )

    # angular kinematics of outgoing photon
    cost = -(1 / (sqrt(1 + gamma * gamma))) * (
        1 + (0.5 * gamma * gamma) * ((1 + (t / Q2)) / (1 + ((xbj * t) / (Q2))))
    )
    cost = tf.math.maximum(cost, -1.)
    sint = sqrt(1.0 - cost * cost)

    # outgoing photon energy
    q0p = (sqrt(Q2) / gamma) * (1 + ((xbj * t) / Q2))

    # conversion from GeV to NanoBarn
    jacobian = (1.0 / (2.0 * M * xbj * k0)) * 4.0 * pi
    conversion = (0.1973 * 0.1973) * 10000000 * jacobian

    # ratio of momentum transfer to proton mass
    tau = -t / (4.0 * M * M)

    ###############################################################################
    ## Creates arrays of 4-vector kinematics uses in Bethe Heitler Cross Section ##
    ###############################################################################

    # initial proton 4-momentum
    p = T(
        tf.convert_to_tensor(
            [
                M * depth_vector,
                0.0 * depth_vector,
                0.0 * depth_vector,
                0.0 * depth_vector,
            ]
        )
    )

    # initial lepton 4-momentum
    k = T(
        tf.convert_to_tensor(
            [k0 * depth_vector, k0 * sintl * depth_vector, 0.0 * depth_vector, k0 * costl * depth_vector]
        )
    )

    # final lepton 4-momentum
    kp = T(
        tf.convert_to_tensor(
            [k0p * depth_vector, k0p * sintlp * depth_vector, 0.0 * depth_vector, k0p * costlp* depth_vector]
        )
    )

    # virtual photon 4-momentum
    q = k - kp

    ##################################
    ## Creates four vector products ##
    ##################################
    plp = product(p, p)  # pp
    qq = product(q, q)  # qq
    kk = product(k, k)  # kk
    kkp = product(k, kp)  # kk'
    kq = product(k, q)  # kq
    pk = product(k, p)  # pk
    pkp = product(kp, p)  # pk'

    # sets the Mandelstam variables s which is the center of mass energy
    s = kk + (2 * pk) + plp

    # the Gamma factor in front of the cross section
    Gamma = (alpha ** 3) / (
        16.0 * (pi ** 2) * ((s - M * M) ** 2) * sqrt(1.0 + gamma ** 2) * xbj
    )

    phi = phi * 0.0174532951  # radian conversion

    # final real photon 4-momentum
    qp = T(
        tf.convert_to_tensor(
            [
                q0p * depth_vector,
                q0p * sint * T(cos(phi)),
                q0p * sint * T(sin(phi)),
                q0p * cost * depth_vector,
            ]
        )
    )

    # momentum transfer Δ from the initial proton to the final proton
    d = q - qp

    # final proton momentum
    pp = p + d

    # average initial proton momentum
    P = 0.5 * (p + pp)

    # 4-vector products of variables multiplied by spin vectors
    ppSL = ((M) / (sqrt(1.0 + gamma ** 2))) * (
        xbj * (1.0 - (t / Q2)) - (t / (2.0 * M ** 2))
    )
    kSL = (
        ((Q2) / (sqrt(1.0 + gamma ** 2)))
        * (1.0 + 0.5 * y * gamma ** 2)
        * (1.0 / (2.0 * M * xbj * y))
    )
    kpSL = (
        ((Q2) / (sqrt(1 + gamma ** 2)))
        * (1 - y - 0.5 * y * gamma ** 2)
        * (1.0 / (2.0 * M * xbj * y))
    )

    # 4-vector products denoted in the paper by the commented symbols
    kd = product(k, d)  # dΔ
    kpd = product(kp, d)  # k'Δ
    kP = product(k, P)  # kP
    kpP = product(kp, P)  # k'P
    kqp = product(k, qp)  # kq'
    kpqp = product(kp, qp)  # k'q'
    dd = product(d, d)  # ΔΔ
    Pq = product(P, q)  # Pq
    Pqp = product(P, qp)  # Pq'
    qd = product(q, d)  # qΔ
    qpd = product(qp, d)  # q'Δ

    # transverse vector products
    kkT = tproduct(k, k)
    kqpT = tproduct(k, qp)
    kkpT = tproduct(k, kp)
    ddT = tproduct(d, d)
    kdT = tproduct(k, d)
    kpqpT = tproduct(kp, qp)
    qpdT = tproduct(qp, d)
    kPT = tproduct(k, P)
    kpPT = tproduct(kp, P)
    qpPT = tproduct(qp, P)
    kpdT = tproduct(kp, d)

    # light cone variables expressed as A^{+-} = 1/sqrt(2)(A^{0} +- A^{3})
    inv_root_2 = 1 / sqrt(2.0)
    inv_root_2 = tf.dtypes.cast(inv_root_2, tf.float64)
    kplus = T(inv_root_2 * (k[..., 0] + k[..., 3]))
    kpplus = T(inv_root_2 * (kp[..., 0] + kp[..., 3]))
    kminus = T(inv_root_2 * (k[..., 0] - k[..., 3]))
    kpminus = T(inv_root_2 * (kp[..., 0] - kp[..., 3]))
    qplus = T(inv_root_2 * (q[..., 0] + q[..., 3]))
    qpplus = T(inv_root_2 * (qp[..., 0] + qp[..., 3]))
    qminus = T(inv_root_2 * (q[..., 0] - q[..., 3]))
    qpminus = T(inv_root_2 * (qp[..., 0] - qp[..., 3]))
    Pplus = T(inv_root_2 * (P[..., 0] + P[..., 3]))
    Pminus = T(inv_root_2 * (P[..., 0] - P[..., 3]))
    dplus = T(inv_root_2 * (d[..., 0] + d[..., 3]))
    dminus = T(inv_root_2 * (d[..., 0] - d[..., 3]))

    # expresssions used that appear in coefficient calculations
    Dplus = (1 / (2 * kpqp)) - (1 / (2 * kqp))
    Dminus = (1 / (2 * kpqp)) + (1 / (2 * kqp))

    # calculates BH
    AUUBH = ((16.0 * M * M) / (kqp * kpqp)) * (
        (4.0 * tau * (kP * kP + kpP * kpP)) - ((tau + 1.0) * (kd * kd + kpd * kpd))
    )
    BUUBH = ((32.0 * M * M) / (kqp * kpqp)) * (kd * kd + kpd * kpd)

    # calculates BHLL
    ALLBH = -((16.0 * M * M) / (kqp * kpqp)) * (
        (ppSL / M) * ((kpd * kpd - kd * kd) - 2.0 * tau * (kpd * pkp - kd * pk))
        + t * (kSL / M) * (1.0 + tau) * kd
        - t * (kpSL / M) * (1.0 + tau) * kpd
    )
    BLLBH = ((16.0 * M * M) / (kqp * kpqp)) * (
        (ppSL / M) * (kpd * kpd - kd * kd) + t * (kSL / M) * kd - t * (kpSL / M) * kpd
    )

    # converted Unpolarized Coefficients with the Gamma factor and in nano-barn
    con_AUUBH = (Gamma / t ** 2) * AUUBH * conversion
    con_BUUBH = (Gamma / t ** 2) * BUUBH * conversion

    # converted Longitudinally Polarized Coefficients with the Gamma Factor and in nano-barn
    con_ALLBH = (Gamma / t ** 2) * ALLBH * conversion
    con_BLLBH = (Gamma / t ** 2) * BLLBH * conversion

    ffF1, ffF2, ffGM = kelly(-t)

    # unpolarized Coefficients multiplied by the Form Factors calculated in form_factor.py
    # we use the Galster Form Factors as approximations
    bhAUU = con_AUUBH * ((ffF1 * ffF1) + (tau * ffF2 * ffF2))
    bhBUU = con_BUUBH * (tau * ffGM * ffGM)

    # polarized Coefficients multiplied by the Form Factors calculated in form_factor.py
    # using the Galster Form Factor Model
    bhALL = con_ALLBH * (ffF2 * ffGM)
    bhBLL = con_BLLBH * (ffGM * ffGM)

    # Calculation of the Total Unpolarized Bethe-Heitler Cross Section
    XSXUUBH = bhAUU + bhBUU
    XSXLLBH = bhALL + bhBLL

    # Calculates the Unpolarized Coefficients in front of the Elastic Form Factors and
    # Compton Form Factors
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    AUUBHDVCS = 16 * Dplus * (
        (kqpT - 2 * kkT - 2 * kqp) * kpP
        + (2 * kpqp - 2 * kkpT - kpqpT) * kP
        + kpqp * kPT
        + kqp * kpPT
        - 2 * kkp * kPT
    ) * cos(phi) - 16 * Dminus * (
        (2 * kkp - kpqpT - kkpT) * Pqp + 2 * kkp * qpPT - kpqp * kPT - kqp * kpPT
    ) * cos(
        phi
    )
    BUUBHDVCS = 8 * xi * Dplus * (
        (kqpT - 2 * kkT - 2 * kqp) * kpd
        + (2 * kpqp - 2 * kkpT - kpqpT) * kd
        + kpqp * kdT
        + kqp * kpdT
        - 2 * kkp * kdT
    ) * cos(phi) - 8 * xi * Dminus * (
        (2 * kkp - kpqpT - kkpT) * qpd + 2 * kkp * qpdT - kpqp * kdT - kqp * kpdT
    ) * cos(
        phi
    )
    CUUBHDVCS = 8 * Dplus * (
        (2 * kkp * kdT - kpqp * kdT - kqp * kpdT)
        + 2 * xi * (2 * kkp * kPT - kpqp * kPT - kqp * kpPT)
    ) * cos(phi) - 8 * Dminus * (
        (kkp * qpdT - kpqp * kdT - kqp * kpdT)
        + 2 * xi * (kkp * qpPT - kpqp * kPT - kqp * kpPT)
    ) * cos(
        phi
    )

    # Calculates the Unpolarized Beam Polarized Target Coefficients in front of the
    # Elastic Form Factors and Compton Form Factors
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    AULBHDVCS = -16 * Dplus * (
        kpP * (2 * kkT - kqpT + 2 * kqp)
        + kP * (2 * kkpT - kpqpT + 2 * kpqp)
        + 2 * kkp * kPT
        - kpqp * kPT
        - kqp * kpPT
    ) * sin(phi) - 16 * Dminus * (
        Pqp * (kkpT + kpqpT - 2 * kkp) - (2 * kkp * qpPT - kpqp * kPT - kqp * kpPT)
    ) * sin(
        phi
    )
    BULBHDVCS = -8 * xi * Dplus * (
        kpd * (2 * kkT - kqpT + 2 * kqp)
        + kd * (2 * kkpT - kpqpT + 2 * kpqp)
        + 2 * kkp * kdT
        - kpqp * kdT
        - kqp * kpdT
    ) * sin(phi) - 8 * xi * Dminus * (
        qpd * (kkpT + kpqpT - 2 * kkp) - (2 * kkp * qpdT - kpqp * kdT - kqp * kpdT)
    ) * sin(
        phi
    )
    CULBHDVCS = -4 * Dplus * (
        2 * (2 * kkp * kdT - kpqp * kdT - kqp * kpdT)
        + 4 * xi * (2 * kkp * kPT - kpqp * kPT - kqp * kpPT)
    ) * sin(phi) - 4 * Dminus * (
        -2 * (kkp * qpdT - kpqp * kdT - kqp * kpdT)
        - 4 * xi * (kkp * qpPT - kpqp * kPT - kqp * kpPT)
    ) * sin(
        phi
    )

    # Calculates the Polarized Beam Unpolarized Target Coefficients in front of the
    # Elastic Form Factors and Compton Form Factors
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    ALUBHDVCS = -16 * Dplus * (
        2
        * (
            k[:, 1] * Pplus * kp[:, 1] * kminus
            - k[:, 1] * Pplus * kpminus * k[:, 1]
            + k[:, 1] * Pminus * kpplus * k[:, 1]
            - k[:, 1] * Pminus * kp[:, 1] * kplus
            + k[:, 1] * P[:, 1] * kpminus * kplus
            - k[:, 1] * P[:, 1] * kpplus * kminus
        )
        + 2
        * (
            kp[:, 1] * Pplus * qpminus * k[:, 1]
            - kp[:, 1] * Pplus * qp[:, 1] * kminus
            + kp[:, 1] * Pminus * qp[:, 1] * kplus
            - kp[:, 1] * Pminus * qpplus * k[:, 1]
            + kp[:, 1] * P[:, 1] * qpplus * kminus
            - kp[:, 1] * P[:, 1] * qpminus * kplus
            + k[:, 1] * Pplus * qpminus * kp[:, 1]
            - k[:, 1] * Pplus * qp[:, 1] * kpminus
            + k[:, 1] * Pminus * qp[:, 1] * kpplus
            - k[:, 1] * Pminus * qpplus * kp[:, 1]
            + k[:, 1] * P[:, 1] * qpplus * kpminus
            - k[:, 1] * P[:, 1] * qpminus * kpplus
            + 2 * (qpminus * Pplus - qpplus * Pminus) * kkp
        )
    ) * sin(phi) - 16 * Dminus * (-2) * (
        2 * (kminus * kpplus - kplus * kpminus) * Pqp
        + kpminus * kplus * qp[:, 1] * P[:, 1]
        + kpplus * k[:, 1] * qpminus * P[:, 1]
        + kp[:, 1] * kminus * qpplus * P[:, 1]
        - kpplus * kminus * qp[:, 1] * P[:, 1]
        - kp[:, 1] * kplus * qpminus * P[:, 1]
        - kpminus * k[:, 1] * qpplus * P[:, 1]
        + kpminus * kplus * qp[:, 2] * P[:, 2]
        - kpplus * kminus * qp[:, 2] * P[:, 2]
    ) * sin(
        phi
    )
    BLUBHDVCS = -8 * xi * Dplus * (
        2
        * (
            k[:, 1] * dplus * kp[:, 1] * kminus
            - k[:, 1] * dplus * kpminus * k[:, 1]
            + k[:, 1] * dminus * kpplus * k[:, 1]
            - k[:, 1] * dminus * kp[:, 1] * kplus
            + k[:, 1] * d[:, 1] * kpminus * kplus
            - k[:, 1] * d[:, 1] * kpplus * kminus
        )
        + 2
        * (
            kp[:, 1] * dplus * qpminus * k[:, 1]
            - kp[:, 1] * dplus * qp[:, 1] * kminus
            + kp[:, 1] * dminus * qp[:, 1] * kplus
            - kp[:, 1] * dminus * qpplus * k[:, 1]
            + kp[:, 1] * d[:, 1] * qpplus * kminus
            - kp[:, 1] * d[:, 1] * qpminus * kplus
            + k[:, 1] * dplus * qpminus * kp[:, 1]
            - k[:, 1] * dplus * qp[:, 1] * kpminus
            + k[:, 1] * dminus * qp[:, 1] * kpplus
            - k[:, 1] * dminus * qpplus * kp[:, 1]
            + k[:, 1] * d[:, 1] * qpplus * kpminus
            - k[:, 1] * d[:, 1] * qpminus * kpplus
            + 2 * (qpminus * dplus - qpplus * dminus) * kkp
        )
    ) * sin(phi) - 8 * xi * Dminus * (-2) * (
        2 * (kminus * kpplus - kplus * kpminus) * qpd
        + kpminus * kplus * qp[:, 1] * d[:, 1]
        + kpplus * k[:, 1] * qpminus * d[:, 1]
        + kp[:, 1] * kminus * qpplus * d[:, 1]
        - kpplus * kminus * qp[:, 1] * d[:, 1]
        - kp[:, 1] * kplus * qpminus * d[:, 1]
        - kpminus * k[:, 1] * qpplus * d[:, 1]
        + kpminus * kplus * qp[:, 2] * d[:, 2]
        - kpplus * kminus * qp[:, 2] * d[:, 2]
    ) * sin(
        phi
    )
    CLUBHDVCS = 8 * Dplus * (
        2
        * (kp[:, 1] * kpminus * kplus * d[:, 1] - kp[:, 1] * kpplus * kminus * d[:, 1])
        + 2
        * (
            kp[:, 1] * qpminus * kplus * d[:, 1]
            - kp[:, 1] * qpplus * kminus * d[:, 1]
            + k[:, 1] * qpminus * kpplus * d[:, 1]
            - k[:, 1] * qpplus * kpminus * d[:, 1]
        )
    ) * sin(phi) + 8 * Dminus * (-2) * (
        -kpminus * k[:, 1] * qpplus * d[:, 1]
        + kpminus * kplus * qp[:, 1] * d[:, 1]
        + kpplus * k[:, 1] * qpminus * d[:, 1]
        - kpplus * kminus * qp[:, 1] * d[:, 1]
        + kp[:, 1] * kminus * qpplus * d[:, 1]
        - kp[:, 1] * kplus * qpminus * d[:, 1]
        - qp[:, 2] * d[:, 2] * (kpplus * kminus - kpminus * kplus)
    ) * sin(
        phi
    )

    # Calculates the Longitudinally Polarized Coefficients in front of the EFFs
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    ALLBHDVCS = -16 * Dplus * (
        2 * kp[:, 1] * (kp[:, 1] * kminus - kpminus * k[:, 1]) * Pplus
        + 2 * kp[:, 1] * (kpplus * k[:, 1] - kp[:, 1] * kplus) * Pminus
        + 2 * kp[:, 1] * (kpminus * kplus - kpplus * kminus) * P[:, 1]
        + 2
        * (
            kp[:, 1] * (qpminus * k[:, 1] - qp[:, 1] * kminus) * Pplus
            + kp[:, 1] * (qp[:, 1] * kplus - qpplus * k[:, 1]) * Pminus
            + kp[:, 1] * (qpplus * kminus - qpminus * kplus) * P[:, 1]
            + k[:, 1] * (qpminus * kp[:, 1] - qp[:, 1] * kpminus) * Pplus
            + k[:, 1] * (qp[:, 1] * kpplus - qpplus * kp[:, 1]) * Pminus
            + k[:, 1] * (qpplus * kpminus - qpminus * kpplus) * P[:, 1]
            - 2 * kkp * (qpplus * Pminus - qpminus * Pplus)
        )
    ) * cos(phi) - 16 * Dminus * (-2) * (
        2 * Pqp * (kpplus * kminus - kpminus * kplus)
        + P[:, 2] * qp[:, 2] * (kpplus * kminus - kpminus * kplus)
        + P[:, 1]
        * (
            kp[:, 1] * kplus * qpminus
            - kp[:, 1] * kminus * qpplus
            + kpminus * k[:, 1] * qpplus
            - kpminus * kplus * qp[:, 1]
            + kpplus * kminus * qp[:, 1]
            - kpplus * k[:, 1] * qpminus
        )
    ) * cos(
        phi
    )
    BLLBHDVCS = -8 * xi * Dplus * (
        2 * kp[:, 1] * (kp[:, 1] * kminus - kpminus * k[:, 1]) * dplus
        + 2 * kp[:, 1] * (kpplus * k[:, 1] - kp[:, 1] * kplus) * dminus
        + 2 * kp[:, 1] * (kpminus * kplus - kpplus * kminus) * d[:, 1]
        + 2
        * (
            kp[:, 1] * (qpminus * k[:, 1] - qp[:, 1] * kminus) * dplus
            + kp[:, 1] * (qp[:, 1] * kplus - qpplus * k[:, 1]) * dminus
            + kp[:, 1] * d[:, 1] * (qpplus * kminus - qpminus * kplus)
            + k[:, 1] * (qpminus * kp[:, 1] - qp[:, 1] * kpminus) * dplus
            + k[:, 1] * (qp[:, 1] * kpplus - qpplus * kp[:, 1]) * dminus
            + k[:, 1] * (qpplus * kpminus - qpminus * kpplus) * d[:, 1]
            - 2 * kkp * (qpplus * dminus - qpminus * dplus)
        )
    ) * cos(phi) - 8 * xi * Dminus * (-2) * (
        2 * qpd * (kpplus * kminus - kpminus * kplus)
        + d[:, 2] * qp[:, 2] * (kpplus * kminus - kpminus * kplus)
        + d[:, 1]
        * (
            kp[:, 1] * kplus * qpminus
            - kp[:, 1] * kminus * qpplus
            + kpminus * k[:, 1] * qpplus
            - kpminus * kplus * qp[:, 1]
            + kpplus * kminus * qp[:, 1]
            - kpplus * k[:, 1] * qpminus
        )
    ) * cos(
        phi
    )
    CLLBHDVCS = 16 * Dplus * (
        2 * (k[:, 1] * kminus * kpplus * d[:, 1] - k[:, 1] * kpminus * kplus * d[:, 1])
        + 2
        * (
            kp[:, 1] * qpplus * kminus * d[:, 1]
            - kp[:, 1] * qpminus * kplus * d[:, 1]
            + k[:, 1] * qpplus * kpminus * d[:, 1]
            - k[:, 1] * qpminus * kpplus * d[:, 1]
        )
    ) * cos(phi) - 16 * Dminus * (-2) * (
        -d[:, 1]
        * (
            kpminus * kplus * qp[:, 1]
            - kpminus * k[:, 1] * qpplus
            + kpplus * k[:, 1] * qpminus
            - kpplus * kminus * qp[:, 1]
            + kp[:, 1] * kminus * qpplus
            - kp[:, 1] * kplus * qpminus
        )
        + qp[:, 2] * kpplus * kminus * d[:, 2]
        - qp[:, 2] * kpminus * kplus * d[:, 2]
    ) * cos(
        phi
    )

    # Converted Unpolarized Coefficients with the Gamma factor and in nano-barn
    con_AUUBHDVCS = (Gamma / (Q2 * -t)) * AUUBHDVCS * conversion
    con_BUUBHDVCS = (Gamma / (Q2 * -t)) * BUUBHDVCS * conversion
    con_CUUBHDVCS = (Gamma / (Q2 * -t)) * CUUBHDVCS * conversion

    # Converted Longitudinally Polarized Coefficients with the Gamma Factor and in nano-barn
    con_ALLBHDVCS = (Gamma / (Q2 * -t)) * ALLBHDVCS * conversion
    con_BLLBHDVCS = (Gamma / (Q2 * -t)) * BLLBHDVCS * conversion
    con_CLLBHDVCS = (Gamma / (Q2 * -t)) * CLLBHDVCS * conversion

    # Converted Longitudinally Polarized Beam Unpolarized Target Coefficients with
    # the Gamma Factor and in nano-barn
    con_ALUBHDVCS = (Gamma / (Q2 * -t)) * ALUBHDVCS * conversion
    con_BLUBHDVCS = (Gamma / (Q2 * -t)) * BLUBHDVCS * conversion
    con_CLUBHDVCS = (Gamma / (Q2 * -t)) * CLUBHDVCS * conversion

    # Converted Longitudinally Polarized Target Unpolarized Beam Coefficients with
    # the Gamma Factor and in nano-barn
    con_AULBHDVCS = (Gamma / (Q2 * -t)) * AULBHDVCS * conversion
    con_BULBHDVCS = (Gamma / (Q2 * -t)) * BULBHDVCS * conversion
    con_CULBHDVCS = (Gamma / (Q2 * -t)) * CULBHDVCS * conversion

    # Unpolarized Coefficients multiplied by the Form Factors
    bhdvcsAUU = con_AUUBHDVCS * (ffF1 * reH + tau * ffF2 * reE)
    bhdvcsBUU = con_BUUBHDVCS * (ffGM * (reH + reE))
    bhdvcsCUU = con_CUUBHDVCS * (ffGM * reHt)

    # Polarized Coefficients multiplied by the Form Factors
    bhdvcsALU = con_ALUBHDVCS * (ffF1 * imH + tau * ffF2 * imE)
    bhdvcsBLU = con_BLUBHDVCS * (ffGM * (imH + imE))
    bhdvcsCLU = con_CLUBHDVCS * (ffGM * imHt)

    # Unpolarized Beam Polarized Target Coefficients multiplied by the Form Factors
    bhdvcsAUL = con_AULBHDVCS * (ffF1 * imHt - xi * ffF1 * imEt + tau * ffF2 * imEt)
    bhdvcsBUL = con_BULBHDVCS * (ffGM * imHt)
    bhdvcsCUL = con_CULBHDVCS * (ffGM * (imH + imE))

    # Polarized Beam Unpolarized Target Coefficients multiplied by the Form Factors
    bhdvcsALL = con_ALLBHDVCS * (ffF1 * reHt - xi * ffF1 * reEt + tau * ffF2 * reEt)
    bhdvcsBLL = con_BLLBHDVCS * (ffGM * reHt)
    bhdvcsCLL = con_CLLBHDVCS * (ffGM * (reH + reE))

    # Calculation of the Total Unpolarized Bethe-Heitler Cross Section
    XSXUUBHDVCS = bhdvcsAUU + bhdvcsBUU + bhdvcsCUU
    XSXLLBHDVCS = bhdvcsALL + bhdvcsBLL + bhdvcsCLL
    XSXULBHDVCS = bhdvcsAUL + bhdvcsBUL + bhdvcsCUL
    XSXLUBHDVCS = bhdvcsALU + bhdvcsBLU + bhdvcsCLU
    FUUT = (
        (Gamma / (Q2))
        / (1 - eps)
        * conversion
        * (
            4
            * (
                (1 - xi * xi) 
                * (reH * reH + imH * imH + reHt * reHt + imHt * imHt)
                + ((t0 - t) / (2 * M * M))
                * (
                    reE * reE
                    + imE * imE
                    + xi * xi * reEt * reEt
                    + xi * xi * imEt * imEt
                )
                - ((2 * xi * xi) / (1 - xi * xi))
                * (reH * reE + imH * imE + reHt * reEt + imHt * imEt)
            )
        )
    )

    XSXUU = XSXUUBHDVCS + XSXUUBH + FUUT
    XSXLL = XSXLLBHDVCS + XSXLLBH
    XSXUL = XSXULBHDVCS
    XSXLU = XSXLUBHDVCS
    XSXALU = XSXLU / XSXUU
    XSXAUL = XSXUL / XSXUU
    XSXALL = XSXLL / XSXUU
    sigmas = T(tf.stack([XSXUU, XSXLU, XSXUL, XSXLL, XSXALU, XSXAUL, XSXALL]))
    gather_nd_idxs = tf.stack(
        [tf.range(sigmas.shape[0], dtype=tf.int32), L - 1], axis=1
    )
    return tf.gather_nd(sigmas, gather_nd_idxs)

def bkm_ff_to_xsx(reH, imH, reE, imE, reHt, imHt, reEt, imEt, phi, xbj, t, Q2, L, k0):
    depth_vector = tf.ones((reH.shape[0],), dtype=tf.float64)
    pi2 = pi*pi
    M2 = M*M
    sign = -1
    jacobian = (1/(2*M*xbj*k0))*2*pi
    conversion =(.1973*.1973)*jacobian*10000000
    
    Q = sqrt(Q2)
    
    nu = Q2/(2*M*xbj)
    gamma = Q/nu
    y = Q/(gamma*k0)

    eps = 2*xbj*M/sqrt(Q2)
    eps2 = eps**2

    y2 = y**2
    y3 = y**3
    x2 = xbj**2
    tmx = 2-xbj
    tmy = 2-y
    tmy2 = tmy**2
    omx = 1-xbj
    omy = 1-y
    tmx2 = tmx**2
    seps = sqrt(1+eps2)
    ope = 1+eps2
    xi = xbj/(2-xbj)
    
    xsx_coef_num = (alpha**3)*xbj*y
    xsx_coef_denom = 16*pi2*Q2*seps
    xsx_coef = xsx_coef_num/xsx_coef_denom

    tmin_num = -Q2*(2*omx*(1-seps)+eps2)
    tmin_denom = 4*xbj*omx+eps2
    tmin = tmin_num/tmin_denom
    
    tq = t/Q2
    optq = 1+tq
    omtq = 1-tq
    
    gam = 1-y-0.25*y2*eps2
    tpt = 1-(tmin/t)
    tp = t-tmin

    j = (1-y-0.5*y*eps2)*optq-omx*tmy*tq
    k2 = -tq*omx*gam*tpt*(seps+((4*xbj*omx+eps2)/(4*omx))*(tp/Q2))
    k = sqrt(k2)
    
    tau = -t/(4*M2)

    ffF1, ffF2, ffGM = kelly(-t)

    phi = pi - phi *.0174532951

    p1 = (-1/(y*ope))*(j+2*k*cos(phi))
    p2 = 1+tq+(1/(y*ope))*(j+2*k*cos(phi))

    bhdenom_coef = x2*y2*ope**2*t*p1*p2
    T_BH_coef = 1/(bhdenom_coef)

    dvcs_denom = y2*Q2
    T_DVCS_coef = 1/dvcs_denom

    int_denom = xbj*y3*t*p1*p2
    T_INT_coef = 1/int_denom

    c0_bh_unp_l1 = 8*K2*((2+3*eps2)*(Q2/t)*(ffF1*ffF1+tau*ffF2*ffF2)+2*x2*(ffF1+ffF2)*(ffF1+ffF2))
    c0_bh_unp_l2 = tmy2*((2+eps2)*((4*x2*M2/t)*(optq**2)+4*omx*(1+(xbj*tq)))*(ffF1*ffF1+tau*ffF2*ffF2))
    c0_bh_unp_l3 = tmy2*4*x2*(xbj+(1-xbj+eps2/2)*(omtq**2)-xbj*(1-2*xbj)*(tq**2))*(ffF1+ffF2)*(ffF1+ffF2)
    c0_bh_unp_l4 = 8*(1+eps2)*gam*(2*eps2*(1+tau)*(ffF1*ffF1+tau*ffF2*ffF2)-x2*(omtq**2)*(ffF1+ffF2)*(ffF1+ffF2))
    c0_bh_unp = c0_bh_unp_l1 + c0_bh_unp_l2 + c0_bh_unp_l3+ c0_bh_unp_l4    

    c1_bh_unp = 8*K*tmy*(((4*x2*M2/t)-2*xbj-eps2)*(ffF1*ffF1+tau*ffF2*ffF2) + 2*x2*(1-(1-2*xbj)*t/Q2)*(ffF1+ffF2)*(ffF1+ffF2))

    c2_bh_unp = 8*x2*K2*((4*M2/t)*(ffF1*ffF1+tau*ffF2*ffF2)+2*(ffF1+ffF2)*(ffF1+ffF2))

    bh_unp = T_BH_coef*(c0_bh_unp+c1_bh_unp*cos(phi)+c2_bh_unp*cos(2*phi))*conversion*xsx_coef

    dvcs_uu_cff = 4*omx*(reH*reH+imH*imH+reHt*reHt+imHt*imHt)-2*x2*(reH*imE+reE*imH+reHt*imEt+imHt*reEt)-(x2-tau*tmx2)*(reE*reE+imE*imE)+tau*x2*(reEt*reEt+imEt*imEt)
    dvcs_uu_num = 2*(2-2*y+y2)
    dvcs_uu_denom = tmx2
    dvcs_uu_coef = dvcs_uu_num/dvcs_uu_denom
    dvcs_unp = dvcs_uu_coef*dvcs_uu_cff*T_DVCS_coef*conversion*xsx_coef

    cui = ffF1*reH+xi*(ffF1+ffF2)*reHt+tau*ffF2*reE
    dui = -xi*(ffF1+ffF2)*(xi*(reH+reE)+reHt)
    c0_int_unp = -8*tmy*((tmy2)*(1/(1-y))*k2*cui+tq*omy*tmx*(cui+dui))
    c1_int_unp = -8*k*(2-2*y+y2)*cui
    int_unp = sign*xsx_coef*T_INT_coef*conversion*(c0_int_unp + c1_int_unp*cos(phi))

    unp_tot = bh_unp + dvcs_unp + int_unp
    
    return unp_tot

def bkm_alt_ff_to_xsx(reH, imH, reE, imE, reHt, imHt, reEt, imEt, phi, xbj, t, Q2, L, k0):
    depth_vector = tf.ones((reH.shape[0],), dtype=tf.float64)
    pi2 = pi*pi
    M2 = M*M
    sign = -1
    jacobian = (1/(2*M*xbj*k0))*2*pi
    conversion =(.1973*.1973)*jacobian*10000000
    
    Q = sqrt(Q2)
    
    nu = Q2/(2*M*xbj)
    gamma = Q/nu
    y = Q/(gamma*k0)

    eps = 2*xbj*M/sqrt(Q2)
    eps2 = eps**2

    y2 = y**2
    y3 = y**3
    x2 = xbj**2
    tmx = 2-xbj
    tmy = 2-y
    tmy2 = tmy**2
    tmy3 = tmy**3
    omx = 1-xbj
    omy = 1-y
    tmx2 = tmx**2
    seps = sqrt(1+eps2)
    ope = 1+eps2
    xi = xbj/(2-xbj)
    
    xsx_coef_num = (alpha**3)*xbj*y
    xsx_coef_denom = 16*pi2*Q2*seps
    xsx_coef = xsx_coef_num/xsx_coef_denom

    tmin_num = -Q2*(2*omx*(1-seps)+eps2)
    tmin_denom = 4*xbj*omx+eps2
    tmin = tmin_num/tmin_denom
    
    tq = t/Q2
    optq = 1+tq
    omtq = 1-tq
    
    gam = 1-y-0.25*y2*eps2
    tpt = 1-(tmin/t)
    tp = t-tmin

    j = (1-y-0.5*y*eps2)*optq-omx*tmy*tq
    k2 = -tq*omx*gam*tpt*(seps+((4*xbj*omx+eps2)/(4*omx))*(tp/Q2))
    k = sqrt(k2)
    
    tau = -t/(4*M2)

    ffF1, ffF2, ffGM = kelly(-t)

    phi = pi - phi *.0174532951

    p1 = (-1/(y*ope))*(j+2*k*cos(phi))
    p2 = 1+tq+(1/(y*ope))*(j+2*k*cos(phi))

    bhdenom_coef = x2*y2*ope**2*t*p1*p2
    T_BH_coef = 1/(bhdenom_coef)

    dvcs_denom = y2*Q2
    T_DVCS_coef = 1/dvcs_denom

    int_denom = xbj*y3*t*p1*p2
    T_INT_coef = 1/int_denom

    c0_bh_unp_l1 = 8*k2*((2+3*eps2)*(Q2/t)*(ffF1*ffF1+tau*ffF2*ffF2)+2*x2*(ffF1+ffF2)*(ffF1+ffF2))
    c0_bh_unp_l2 = tmy2*((2+eps2)*((4*x2*M2/t)*(optq**2)+4*omx*(1+(xbj*tq)))*(ffF1*ffF1+tau*ffF2*ffF2))
    c0_bh_unp_l3 = tmy2*4*x2*(xbj+(1-xbj+eps2/2)*(omtq**2)-xbj*(1-2*xbj)*(tq**2))*(ffF1+ffF2)*(ffF1+ffF2)
    c0_bh_unp_l4 = 8*(1+eps2)*gam*(2*eps2*(1+tau)*(ffF1*ffF1+tau*ffF2*ffF2)-x2*(omtq**2)*(ffF1+ffF2)*(ffF1+ffF2))
    c0_bh_unp = c0_bh_unp_l1 + c0_bh_unp_l2 + c0_bh_unp_l3+ c0_bh_unp_l4    

    c1_bh_unp = 8*k*tmy*(((4*x2*M2/t)-2*xbj-eps2)*(ffF1*ffF1+tau*ffF2*ffF2) + 2*x2*(1-(1-2*xbj)*t/Q2)*(ffF1+ffF2)*(ffF1+ffF2))

    c2_bh_unp = 8*x2*k2*((4*M2/t)*(ffF1*ffF1+tau*ffF2*ffF2)+2*(ffF1+ffF2)*(ffF1+ffF2))

    bh_unp = T_BH_coef*(c0_bh_unp+c1_bh_unp*cos(phi)+c2_bh_unp*cos(2*phi))*conversion*xsx_coef

    dvcs_uu_cff = 4*omx*(reH*reH+imH*imH+reHt*reHt+imHt*imHt)-2*x2*(reH*imE+reE*imH+reHt*imEt+imHt*reEt)-(x2-tau*tmx2)*(reE*reE+imE*imE)+tau*x2*(reEt*reEt+imEt*imEt)
    dvcs_uu_num = 2*(2-2*y+y2)
    dvcs_uu_denom = tmx2
    dvcs_uu_coef = dvcs_uu_num/dvcs_uu_denom
    dvcs_unp = dvcs_uu_coef*dvcs_uu_cff*T_DVCS_coef*conversion*xsx_coef

    auu_int = (-8*tmy3*(1/omy)*k2-8*tmy*tq*omy*tmx-8*k*(2-2*y+y2)*cos(phi))*(ffF1*reH+tau*ffF2*reE)
    buu_int =8*tmy*x2*(1/tmx2)*tq*omy*tmx*ffGM*(reH+reE)
    cuu_int =(-8*tmy3*(1/omy)*xi*k2-8*k*(2-2*y+y2)*xi*cos(phi))*ffGM*reHt
    int_unp = sign*xsx_coef*T_INT_coef*conversion*(auu_int + buu_int + cuu_int)

    unp_tot = bh_unp + dvcs_unp + int_unp
    
    return unp_tot

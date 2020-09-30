# 6 degree of freedom Fossen AUV model
# Christopher Iliffe Sprague

import jax.numpy as np
from jax import jit, jacfwd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from dynamics import Dynamics
from collections import OrderedDict

class Fossen(Dynamics):

    # state and control dimensionality
    state_dim = 13
    control_dim = 4

    # state bounds and scaling
    state_lb = np.array([
        *[-2000]*3, # position
        *[-1]*4,      # quaternions
        *[-10]*3,     # velocity
        *[-10]*3      # ang. velocity
    ])
    state_ub = -state_lb

    # control bounds
    control_lb = np.array([*[-200]*2, *[-0.15]*2])
    control_ub = np.array([*[1500]*2, *[0.15]*2])

    # default parameters
    params = OrderedDict(
        Nrr=150., Izz=10., Kt1=0.1, zb=0., Mqq=100., 
        ycp=0., xb=0., zcp=0., Yvv=100., yg=0., Ixx=10., 
        Kt0=0.1, Xuu=1., xg=0., Zww=100., W=15.4*9.81, 
        m=15.4, B=15.4*9.81, zg=0., Kpp=100., Qt1=-0.001, 
        Qt0=0.001, Iyy=10., yb=0., xcp=0.1
    )

    def __init__(self, **kwargs):

        # become dynamics model
        Dynamics.__init__(self, **kwargs)

    @staticmethod
    @jit
    def lagrangian(state, control, homotopy, *params):

        # sanity
        assert len(state.shape) == len(control.shape) == 1
        assert state.shape[-1] == Fossen.state_dim
        assert control.shape[-1] == Fossen.control_dim

        # constant parameters
        Nrr, Izz, Kt1, zb, Mqq, ycp, xb, zcp, Yvv, yg, Ixx, Kt0, Xuu, xg, Zww, W, m, B, zg, Kpp, Qt1, Qt0, Iyy, yb, xcp = params

        # extract state and control
        x, y, z, et0, eps1, eps2, eps3, u, v, w, p, q, r = state
        rpm0, rpm1, de, dr = control/(Fossen.control_ub - Fossen.control_lb)

        # homotopic cost: (quadratic-effort)-time
        a, b = homotopy
        J = sum([
            (1-b)*(
                (1-a)*u**2 + a*np.abs(u)
            )
            + a
        for u in [rpm0, rpm1]])
        return J

    @staticmethod
    @jit
    def state_dynamics(state, control, *params):

        # sanity
        assert len(state.shape) == len(control.shape) == 1
        assert state.shape[-1] == Fossen.state_dim
        assert control.shape[-1] == Fossen.control_dim

        # constant parameters
        Nrr, Izz, Kt1, zb, Mqq, ycp, xb, zcp, Yvv, yg, Ixx, Kt0, Xuu, xg, Zww, W, m, B, zg, Kpp, Qt1, Qt0, Iyy, yb, xcp = params

        # extract state and control
        x, y, z, et0, eps1, eps2, eps3, u, v, w, p, q, r = state
        rpm0, rpm1, de, dr = control
        # rpm0, rpm1, de, dr = control*(Fossen.control_ub - Fossen.control_lb) - Fossen.control_lb

        # common subexpression elimination
        x0 = 2*eps2
        x1 = eps1*x0
        x2 = 2*eps3
        x3 = et0*x2
        x4 = x1 - x3
        x5 = eps1*x2
        x6 = et0*x0
        x7 = x5 + x6
        x8 = -2*eps2**2
        x9 = 1 - 2*eps3**2
        x10 = x8 + x9
        x11 = x1 + x3
        x12 = 2*eps1*et0
        x13 = eps2*x2
        x14 = -x12 + x13
        x15 = -2*eps1**2
        x16 = x15 + x9
        x17 = x5 - x6
        x18 = x12 + x13
        x19 = x15 + x8 + 1
        x20 = p/2
        x21 = eps2/2
        x22 = eps3/2
        x23 = r/2
        x24 = q/2
        x25 = zg**2
        x26 = xg*yg
        x27 = xg**2
        x28 = m**6
        x29 = x27*x28
        x30 = yg*zg
        x31 = m**3
        x32 = yg**2
        x33 = m**2
        x34 = -x25*x33
        x35 = m*(Ixx*m + x34)
        x36 = -x31*x32 + x35
        x37 = x31*x36
        x38 = -x29*x30 + x30*x37
        x39 = x38**2
        x40 = -x27*x31
        x41 = -x29*x32 + x36*(m*(Iyy*m + x34) + x40)
        x42 = -x39 + x41*(-x25*x29 + x36*(m*(Izz*m - x32*x33) + x40))
        x43 = x28*x42
        x44 = x25*x26*x43
        x45 = m*zg
        x46 = m*yg
        x47 = -x38*x45 - x41*x46
        x48 = m**5*xg
        x49 = x30*x38
        x50 = x33*x36
        x51 = x50*xg
        x52 = x41*(-x25*x48 - x51) + x48*x49
        x53 = 1/m
        x54 = 1/x41
        x55 = 1/x42
        x56 = Yvv*abs(v)
        x57 = m*xg
        x58 = p*x57
        x59 = r*x45
        x60 = m*u
        x61 = r*x46
        x62 = np.sin(dr)
        x63 = Kt0*rpm0
        x64 = Kt1*rpm1
        x65 = m*w
        x66 = -x65
        x67 = p*x46
        x68 = x10*x14 - x11*x7
        x69 = -B + W
        x70 = x10*x16 - x11*x4
        x71 = 1/(-x68*(x10*x18 - x17*x4) + x70*(x10*x19 - x17*x7))
        x72 = x10*x71
        x73 = x69*x72
        x74 = x55*(-p*(x66 - x67) - q*(x58 + x59) - r*(x60 - x61) - v*x56 + x62*(-x63 - x64) + x68*x73)
        x75 = x54*x74
        x76 = x53*x75
        x77 = x32*x48 + x51
        x78 = x42*x77
        x79 = x41*x48
        x80 = x30*x79 - x38*x77
        x81 = Zww*abs(w)
        x82 = m*v
        x83 = p*x45
        x84 = q*x46
        x85 = np.sin(de)
        x86 = np.cos(dr)
        x87 = x86*(x63 + x64)
        x88 = -x60
        x89 = q*x45
        x90 = x55*(-p*(x82 - x83) - q*(x88 - x89) - r*(x58 + x84) - w*x81 - x70*x73 + x85*x87)
        x91 = x54*x90
        x92 = x53*x91
        x93 = x25*x31
        x94 = x38*x50
        x95 = x33*x41
        x96 = x95*yg
        x97 = x36*x96 + x94*zg
        x98 = Xuu*abs(u)
        x99 = q*x57
        x100 = np.cos(de)
        x101 = -x82
        x102 = r*x57
        x103 = x71*(-x4*x68 + x7*x70)
        x104 = x55*(-p*(x59 + x84) - q*(x65 - x99) - r*(x101 - x102) - u*x98 + x100*x87 + x103*x69)
        x105 = x104*x54
        x106 = x105*x53
        x107 = p*q
        x108 = Qt0*rpm0
        x109 = Qt1*rpm1
        x110 = x86*(x108 + x109)
        x111 = -x58
        x112 = -x84
        x113 = x68*x72
        x114 = x55*(B*(x103*yb - x113*xb) + Ixx*x107 - Iyy*x107 - Nrr*r*abs(r) - W*(x103*yg - x113*xg) - u*(x102 + x82 - x98*ycp) - v*(x56*xcp + x61 + x88) - w*(x111 + x112) + x110*x85)
        x115 = m*x114
        x116 = x30*x43*xg
        x117 = x48*yg
        x118 = x117*x38 - x79*zg
        x119 = q*r
        x120 = -x59
        x121 = x70*x72
        x122 = x55*(B*(x113*zb + x121*yb) + Iyy*x119 - Izz*x119 - Kpp*p*abs(p) - W*(x113*zg + x121*yg) - u*(x112 + x120) - v*(-x56*zcp + x65 + x67) - w*(x101 + x81*ycp + x83) + x100*x110)
        x123 = x122*x54
        x124 = x123*x53
        x125 = p*r
        x126 = x55*(B*(-x103*zb - x121*xb) - Ixx*x125 + Izz*x125 - Mqq*q*abs(q) - W*(-x103*zg - x121*xg) - u*(x66 + x98*zcp + x99) - v*(x111 + x120) - w*(x60 - x81*xcp + x89) + x62*(-x108 - x109))
        x127 = x126*x54
        x128 = x127*x53
        x129 = m**4*xg
        x130 = x36*x57
        x131 = -x129*x49 + x41*(x129*x25 + x130)
        x132 = 1/x36
        x133 = x106*x132
        x134 = x27*x32
        x135 = m**9*x134
        x136 = x132*x76
        x137 = x129*x30
        x138 = x31*x41
        x139 = x138*zg
        x140 = -x139*yg
        x141 = x132*x92
        x142 = x124*x132
        x143 = x128*x132
        x144 = -x129*x32 - x130
        x145 = x30*x48
        x146 = -x137*x41 - x144*x38
        x147 = x42*x50
        x148 = x144*x147
        x149 = m**8*x134
        x150 = x26*x31
        x151 = x139*xg - x150*x38
        x152 = x145*x42
        x153 = x117*x42

        # return state rate of change dx/dt
        return np.array([
            u*x10 + v*x4 + w*x7,
            u*x11 + v*x16 + w*x14,
            u*x17 + v*x18 + w*x19,
            -eps1*x20 - q*x21 - r*x22,
            et0*x20 - q*x22 + r*x21,
            -eps1*x23 + eps3*x20 + et0*x24,
            eps1*x24 - eps2*x20 + et0*x23,
            x106*(x42*(x36*x93 + x41) - x47*x97) - x115*x36*x47 + x124*(x116 - x118*x47) + x128*(-x37*x42*zg + x47*x94) + x76*(x44 - x47*x52) + x92*(-x45*x78 - x47*x80),
            -x115*x131 + x133*(-x131*x97 + x36*x44) + x136*(-x131*x52 + x42*(x135*x25 + x41*(x36 + x93))) + x141*(-x131*x80 + x42*(-x137*x77 + x140)) + x142*(-x118*x131 + x42*(x135*zg + x139)) + x143*(-x116*x36 + x131*x94),
            -x115*x146 + x133*(-x146*x97 + x148*zg) + x136*(-x146*x52 + x42*(x140 + x144*x145)) + x141*(-x146*x80 + x42*(-x144*x77 + x35*x41)) + x142*(-x118*x146 + x42*(x117*x144 - x138*yg)) + x143*(x146*x94 - x148),
            x105*x132*(-x151*x97 + x152*x36) - x114*x151*x33 + x123*x132*(-x118*x151 + x42*(x149 + x95)) + x127*x132*(x151*x94 - x153*x36) + x132*x75*(-x151*x52 + x42*(x149*zg + x95*zg)) + x132*x91*(-x151*x80 + x42*(-x150*x77 - x96)),
            x105*(-x147*zg - x38*x97) - x114*x94 + x123*(-x118*x38 - x153) + x127*(x147 + x39*x50) + x75*(-x152 - x38*x52) + x91*(-x38*x80 + x78),
            x104*x97 + x114*x36*x95 + x118*x122 - x126*x94 + x52*x74 + x80*x90
        ], dtype=np.float32)

    @staticmethod
    @jit
    def state_dynamics_jac_state_sympy(state, control, *params):

        # constant parameters
        Nrr, Izz, Kt1, zb, Mqq, ycp, xb, zcp, Yvv, yg, Ixx, Kt0, Xuu, xg, Zww, W, m, B, zg, Kpp, Qt1, Qt0, Iyy, yb, xcp = params

        # sanity
        assert len(state.shape) == len(control.shape) == 1
        assert state.shape[-1] == Fossen.state_dim
        assert control.shape[-1] == Fossen.control_dim

        # extract state and control
        x, y, z, et0, eps1, eps2, eps3, u, v, w, p, q, r = state
        rpm0, rpm1, de, dr = control
        # rpm0, rpm1, de, dr = control*(Fossen.control_ub - Fossen.control_lb) - Fossen.control_lb

        # common subexpression elimination
        x0 = 2*w
        x1 = eps2*x0
        x2 = 2*v
        x3 = eps3*x2
        x4 = eps2*x2
        x5 = eps3*x0
        x6 = eps1*x2
        x7 = 4*u
        x8 = et0*x0
        x9 = eps1*x0
        x10 = et0*x2
        x11 = -2*eps2**2
        x12 = 1 - 2*eps3**2
        x13 = x11 + x12
        x14 = 2*eps2
        x15 = eps1*x14
        x16 = 2*eps3
        x17 = et0*x16
        x18 = -x17
        x19 = x15 + x18
        x20 = eps1*x16
        x21 = et0*x14
        x22 = x20 + x21
        x23 = u*x16
        x24 = 4*v
        x25 = u*x14
        x26 = 2*u
        x27 = eps1*x26
        x28 = et0*x26
        x29 = x15 + x17
        x30 = -2*eps1**2
        x31 = x12 + x30
        x32 = 2*eps1
        x33 = et0*x32
        x34 = eps2*x16
        x35 = -x33 + x34
        x36 = 4*w
        x37 = x20 - x21
        x38 = x33 + x34
        x39 = x11 + x30 + 1
        x40 = p/2
        x41 = -x40
        x42 = q/2
        x43 = -x42
        x44 = r/2
        x45 = -x44
        x46 = eps1/2
        x47 = -x46
        x48 = eps2/2
        x49 = -x48
        x50 = eps3/2
        x51 = -x50
        x52 = et0/2
        x53 = -B + W
        x54 = -x15
        x55 = x18 + x54
        x56 = x14*x55
        x57 = x16*x22
        x58 = -x57
        x59 = x13*x32
        x60 = x13*x38 - x19*x37
        x61 = x22*x29
        x62 = x13*x35
        x63 = -x61 + x62
        x64 = x13*x31 - x19*x29
        x65 = x13*x39 - x22*x37
        x66 = -x60*x63 + x64*x65
        x67 = 1/x66
        x68 = x13*x67
        x69 = x68*(x56 + x58 - x59)
        x70 = -x20 + x21
        x71 = x14*x70
        x72 = x14*x22
        x73 = x17 + x54
        x74 = x16*x29 + x16*x73
        x75 = -x56 + x57 + x59
        x76 = x16*x70
        x77 = x61 - x62
        x78 = -x60*x75 - x64*(x71 + x72) - x65*x74 - x77*(x14*x19 + x59 - x76)
        x79 = x66**(-2)
        x80 = x13*x79
        x81 = x78*x80
        x82 = x63*x81
        x83 = x53*x69 + x53*x82
        x84 = 1/m
        x85 = xg*yg
        x86 = zg**2
        x87 = xg**2
        x88 = m**6
        x89 = x87*x88
        x90 = yg*zg
        x91 = m**3
        x92 = yg**2
        x93 = m**2
        x94 = -x86*x93
        x95 = m*(Ixx*m + x94)
        x96 = -x91*x92 + x95
        x97 = x91*x96
        x98 = -x89*x90 + x90*x97
        x99 = x98**2
        x100 = -x87*x91
        x101 = -x89*x92 + x96*(m*(Iyy*m + x94) + x100)
        x102 = x101*(-x86*x89 + x96*(m*(Izz*m - x92*x93) + x100)) - x99
        x103 = x102*x88
        x104 = x103*x86
        x105 = m*zg
        x106 = m*yg
        x107 = -x101*x106 - x105*x98
        x108 = m**5*xg
        x109 = x90*x98
        x110 = x96*xg
        x111 = x110*x93
        x112 = x101*(-x108*x86 - x111) + x108*x109
        x113 = x104*x85 - x107*x112
        x114 = 1/x102
        x115 = x114/x101
        x116 = x113*x115
        x117 = x116*x84
        x118 = x68*x74
        x119 = x64*x81
        x120 = -x118*x53 - x119*x53
        x121 = x108*x92 + x111
        x122 = x102*x121
        x123 = x101*x108
        x124 = -x121*x98 + x123*x90
        x125 = x115*(-x105*x122 - x107*x124)
        x126 = x125*x84
        x127 = x67*(x14*x64 - x16*x77 + x19*x75 + x22*x74)
        x128 = x79*(-x19*x63 + x22*x64)
        x129 = x128*x78
        x130 = x127*x53 + x129*x53
        x131 = x86*x91
        x132 = x93*x96
        x133 = x132*x98
        x134 = x101*x93
        x135 = x134*yg
        x136 = x133*zg + x135*x96
        x137 = x102*(x101 + x131*x96) - x107*x136
        x138 = x115*x137
        x139 = x138*x84
        x140 = B*(x118*yb + x119*yb + x69*zb + x82*zb) - W*(x118*yg + x119*yg + x69*zg + x82*zg)
        x141 = x103*x90
        x142 = x108*yg
        x143 = -x123*zg + x142*x98
        x144 = x115*x84
        x145 = x144*(-x107*x143 + x141*xg)
        x146 = B*(x127*yb + x129*yb - x69*xb - x82*xb) - W*(x127*yg + x129*yg - x69*xg - x82*xg)
        x147 = m*x114
        x148 = x107*x147*x96
        x149 = x64*xb
        x150 = B*(-x118*xb - x127*zb - x129*zb - x149*x81) - W*(-x118*xg - x119*xg - x127*zg - x129*zg)
        x151 = x144*(-x102*x97*zg + x107*x133)
        x152 = x16*x55
        x153 = 2*et0
        x154 = x13*x153
        x155 = x68*(x152 - x154 - x72)
        x156 = -x152 + x154 + x72
        x157 = 4*x13
        x158 = -eps1*x157
        x159 = -x14*x29 + x14*x73 + x158
        x160 = -x156*x60 - x159*x65 - x64*(x158 + x58 + x76) - x77*(x154 - x16*x19 + x71)
        x161 = x160*x80
        x162 = x161*x63
        x163 = x155*x53 + x162*x53
        x164 = x159*x68
        x165 = x161*x64
        x166 = -x164*x53 - x165*x53
        x167 = x67*(x14*x77 + x156*x19 + x159*x22 + x16*x64)
        x168 = x128*x160
        x169 = x167*x53 + x168*x53
        x170 = B*(x155*zb + x162*zb + x164*yb + x165*yb) - W*(x155*zg + x162*zg + x164*yg + x165*yg)
        x171 = B*(-x155*xb - x162*xb + x167*yb + x168*yb) - W*(-x155*xg - x162*xg + x167*yg + x168*yg)
        x172 = B*(-x149*x161 - x164*xb - x167*zb - x168*zb) - W*(-x164*xg - x165*xg - x167*zg - x168*zg)
        x173 = 4*eps2
        x174 = x173*x67
        x175 = x53*x63
        x176 = x22*x32
        x177 = -x176
        x178 = x173*x35
        x179 = x153*x55
        x180 = x13*x16
        x181 = x68*(x177 - x178 + x179 + x180)
        x182 = -x173*x31 - x29*x32 + x32*x73
        x183 = x176 + x178 - x179 - x180
        x184 = x32*x70
        x185 = x153*x70
        x186 = x153*x22
        x187 = -x182*x65 - x183*x60 - x64*(-eps2*x157 - x173*x39 + x185 + x186) - x77*(x153*x19 - x173*x38 + x180 + x184)
        x188 = x187*x80
        x189 = -x174*x175 + x175*x188 + x181*x53
        x190 = x53*x64
        x191 = x182*x68
        x192 = x174*x190 - x188*x190 - x191*x53
        x193 = x67*(x153*x64 + x182*x22 + x183*x19 + x32*x77)
        x194 = x128*x187
        x195 = x193*x53 + x194*x53
        x196 = x63*xb
        x197 = x63*xg
        x198 = B*(x174*x196 - x181*xb - x188*x196 + x193*yb + x194*yb) - W*(x174*x197 - x181*xg - x188*x197 + x193*yg + x194*yg)
        x199 = x64*xg
        x200 = B*(x149*x174 - x149*x188 - x191*xb - x193*zb - x194*zb) - W*(x174*x199 - x188*x199 - x191*xg - x193*zg - x194*zg)
        x201 = x63*zb
        x202 = x64*yb
        x203 = x63*zg
        x204 = x64*yg
        x205 = B*(-x174*x201 - x174*x202 + x181*zb + x188*x201 + x188*x202 + x191*yb) - W*(-x174*x203 - x174*x204 + x181*zg + x188*x203 + x188*x204 + x191*yg)
        x206 = 4*eps3
        x207 = x206*x67
        x208 = x32*x55
        x209 = x206*x35
        x210 = x13*x14
        x211 = x68*(-x186 + x208 - x209 + x210)
        x212 = x186 - x208 + x209 - x210
        x213 = -eps3*x157 + x153*x29 + x153*x73 - x206*x31
        x214 = -x212*x60 - x213*x65 - x64*(x177 + x184 - x206*x39) - x77*(-x185 - x19*x32 - x206*x38 + x210)
        x215 = x214*x80
        x216 = -x175*x207 + x175*x215 + x211*x53
        x217 = x213*x68
        x218 = x190*x207 - x190*x215 - x217*x53
        x219 = x67*(-x153*x77 + x19*x212 + x213*x22 + x32*x64)
        x220 = x128*x214
        x221 = x219*x53 + x220*x53
        x222 = B*(x196*x207 - x196*x215 - x211*xb + x219*yb + x220*yb) - W*(x197*x207 - x197*x215 - x211*xg + x219*yg + x220*yg)
        x223 = B*(x149*x207 - x149*x215 - x217*xb - x219*zb - x220*zb) - W*(x199*x207 - x199*x215 - x217*xg - x219*zg - x220*zg)
        x224 = B*(-x201*x207 + x201*x215 - x202*x207 + x202*x215 + x211*zb + x217*yb) - W*(-x203*x207 + x203*x215 - x204*x207 + x204*x215 + x211*zg + x217*yg)
        x225 = Xuu*abs(u)
        x226 = m*xg
        x227 = r*x226
        x228 = -x227
        x229 = Xuu*u*np.sign(u)
        x230 = x225*ycp + x228 + x229*ycp
        x231 = q*x226
        x232 = -x231
        x233 = -x225*zcp - x229*zcp + x232
        x234 = q*x106
        x235 = r*x105
        x236 = x234 + x235
        x237 = r*x115
        x238 = -x225 - x229
        x239 = r*x106
        x240 = -x239
        x241 = Yvv*abs(v)
        x242 = Yvv*v*np.sign(v)
        x243 = x240 - x241*xcp - x242*xcp
        x244 = p*x226
        x245 = x235 + x244
        x246 = p*x106
        x247 = -x246
        x248 = x241*zcp + x242*zcp + x247
        x249 = -x241 - x242
        x250 = x234 + x244
        x251 = Zww*abs(w)
        x252 = q*x105
        x253 = -x252
        x254 = Zww*w*np.sign(w)
        x255 = x251*xcp + x253 + x254*xcp
        x256 = p*x105
        x257 = -x256
        x258 = -x251*ycp - x254*ycp + x257
        x259 = -x251 - x254
        x260 = Iyy*q
        x261 = m*w
        x262 = Ixx*q - x260 + x261*xg
        x263 = Izz*r
        x264 = m*v
        x265 = -Ixx*r + x263 + x264*xg
        x266 = -x264*yg
        x267 = -x261*zg
        x268 = -Kpp*p*np.sign(p) - Kpp*abs(p) + x266 + x267
        x269 = x232 + 2*x246 + x261
        x270 = x228 + 2*x256 - x264
        x271 = -x234
        x272 = -x235
        x273 = x271 + x272
        x274 = Ixx*p
        x275 = -Iyy*p + x261*yg + x274
        x276 = m*u
        x277 = -x276*xg
        x278 = -Mqq*q*np.sign(q) - Mqq*abs(q) + x267 + x277
        x279 = Iyy*r - x263 + x276*yg
        x280 = -x244
        x281 = x272 + x280
        x282 = x240 + 2*x252 + x276
        x283 = 2*x231 + x247 - x261
        x284 = -Nrr*r*np.sign(r) - Nrr*abs(r) + x266 + x277
        x285 = Izz*p + x264*zg - x274
        x286 = -Izz*q + x260 + x276*zg
        x287 = 2*x239 + x253 - x276
        x288 = x271 + x280
        x289 = 2*x227 + x257 + x264
        x290 = m**4*xg
        x291 = x290*x90
        x292 = x101*x91
        x293 = x292*zg
        x294 = -x293*yg
        x295 = m*x110
        x296 = x101*(x290*x86 + x295) - x109*x290
        x297 = x102*(-x121*x291 + x294) - x124*x296
        x298 = 1/x96
        x299 = x144*x298
        x300 = x297*x299
        x301 = x87*x92
        x302 = m**9*x301
        x303 = x102*(x101*(x131 + x96) + x302*x86) - x112*x296
        x304 = x299*x303
        x305 = x104*x110*yg - x136*x296
        x306 = x299*x305
        x307 = x147*x296
        x308 = x299*(x102*(x293 + x302*zg) - x143*x296)
        x309 = x299*(-x110*x141 + x133*x296)
        x310 = x237*x298
        x311 = x115*x298
        x312 = x297*x311
        x313 = q*x311
        x314 = p*x311
        x315 = -x290*x92 - x295
        x316 = x108*x90
        x317 = -x101*x291 - x315*x98
        x318 = x102*(x294 + x315*x316) - x112*x317
        x319 = x299*x318
        x320 = x102*(x101*x95 - x121*x315) - x124*x317
        x321 = x299*x320
        x322 = x102*x132
        x323 = x315*x322
        x324 = -x136*x317 + x323*zg
        x325 = x299*x324
        x326 = x147*x317
        x327 = x299*(x102*(x142*x315 - x292*yg) - x143*x317)
        x328 = x299*(x133*x317 - x323)
        x329 = m**8*x301
        x330 = x85*x91
        x331 = x293*xg - x330*x98
        x332 = x102*(x134*zg + x329*zg) - x112*x331
        x333 = x311*x332
        x334 = x311*(x102*(-x121*x330 - x135) - x124*x331)
        x335 = x102*x316
        x336 = -x136*x331 + x335*x96
        x337 = x311*x336
        x338 = x114*x146
        x339 = x331*x93
        x340 = x311*(x102*(x134 + x329) - x143*x331)
        x341 = x102*x142
        x342 = x311*(x133*x331 - x341*x96)
        x343 = x114*x339
        x344 = m*x310
        x345 = m*q
        x346 = m*p
        x347 = -x112*x98 - x335
        x348 = x115*x347
        x349 = x115*(x122 - x124*x98)
        x350 = -x136*x98 - x322*zg
        x351 = x115*x350
        x352 = x115*(-x143*x98 - x341)
        x353 = x115*(x132*x99 + x322)
        x354 = x114*x133
        x355 = m*x237
        x356 = x114*x124
        x357 = x112*x114
        x358 = x114*x136
        x359 = x114*x143
        x360 = x134*x96
        x361 = x114*x360
        x362 = m*x356
        x363 = m*r

        # return Jacobian d(dx/dt)/dx
        return np.array([
            [0, 0, 0,                                                              x1 - x3,                                                               x4 + x5,                                                    -eps2*x7 + x6 + x8,                                                   -eps3*x7 - x10 + x9,                                                                    x13,                                                                    x19,                                                                    x22,                                                                      0,                                                                      0,                                                                      0],
            [0, 0, 0,                                                             x23 - x9,                                                  -eps1*x24 + x25 - x8,                                                              x27 + x5,                                                  -eps3*x24 + x1 + x28,                                                                    x29,                                                                    x31,                                                                    x35,                                                                      0,                                                                      0,                                                                      0],
            [0, 0, 0,                                                            -x25 + x6,                                                 -eps1*x36 + x10 + x23,                                                  -eps2*x36 - x28 + x3,                                                              x27 + x4,                                                                    x37,                                                                    x38,                                                                    x39,                                                                      0,                                                                      0,                                                                      0],
            [0, 0, 0,                                                                    0,                                                                   x41,                                                                   x43,                                                                   x45,                                                                      0,                                                                      0,                                                                      0,                                                                    x47,                                                                    x49,                                                                    x51],
            [0, 0, 0,                                                                  x40,                                                                     0,                                                                   x44,                                                                   x43,                                                                      0,                                                                      0,                                                                      0,                                                                    x52,                                                                    x51,                                                                    x48],
            [0, 0, 0,                                                                  x42,                                                                   x45,                                                                     0,                                                                   x40,                                                                      0,                                                                      0,                                                                      0,                                                                    x50,                                                                    x52,                                                                    x47],
            [0, 0, 0,                                                                  x44,                                                                   x42,                                                                   x41,                                                                     0,                                                                      0,                                                                      0,                                                                      0,                                                                    x49,                                                                    x46,                                                                    x52],
            [0, 0, 0, x117*x83 + x120*x126 + x130*x139 + x140*x145 - x146*x148 + x150*x151, x117*x163 + x126*x166 + x139*x169 + x145*x170 - x148*x171 + x151*x172, x117*x189 + x126*x192 + x139*x195 + x145*x205 - x148*x198 + x151*x200, x117*x216 + x126*x218 + x139*x221 + x145*x224 - x148*x222 + x151*x223,     q*x125 - x113*x237 + x139*x238 + x145*x236 - x148*x230 + x151*x233,    -p*x125 + x117*x249 + x137*x237 + x145*x248 - x148*x243 + x151*x245,        p*x116 - q*x138 + x126*x259 + x145*x258 - x148*x250 + x151*x255,  x117*x269 + x126*x270 + x139*x273 + x145*x268 - x148*x262 + x151*x265,  x117*x281 + x126*x282 + x139*x283 + x145*x279 - x148*x275 + x151*x278,  x117*x287 + x126*x288 + x139*x289 + x145*x286 - x148*x284 + x151*x285],
            [0, 0, 0, x120*x300 + x130*x306 + x140*x308 - x146*x307 + x150*x309 + x304*x83, x163*x304 + x166*x300 + x169*x306 + x170*x308 - x171*x307 + x172*x309, x189*x304 + x192*x300 + x195*x306 - x198*x307 + x200*x309 + x205*x308, x216*x304 + x218*x300 + x221*x306 - x222*x307 + x223*x309 + x224*x308,     q*x312 - x230*x307 + x233*x309 + x236*x308 + x238*x306 - x303*x310,    -p*x312 - x243*x307 + x245*x309 + x248*x308 + x249*x304 + x305*x310, -x250*x307 + x255*x309 + x258*x308 + x259*x300 + x303*x314 - x305*x313, -x262*x307 + x265*x309 + x268*x308 + x269*x304 + x270*x300 + x273*x306, -x275*x307 + x278*x309 + x279*x308 + x281*x304 + x282*x300 + x283*x306, -x284*x307 + x285*x309 + x286*x308 + x287*x304 + x288*x300 + x289*x306],
            [0, 0, 0, x120*x321 + x130*x325 + x140*x327 - x146*x326 + x150*x328 + x319*x83, x163*x319 + x166*x321 + x169*x325 + x170*x327 - x171*x326 + x172*x328, x189*x319 + x192*x321 + x195*x325 - x198*x326 + x200*x328 + x205*x327, x216*x319 + x218*x321 + x221*x325 - x222*x326 + x223*x328 + x224*x327, -x230*x326 + x233*x328 + x236*x327 + x238*x325 - x310*x318 + x313*x320, -x243*x326 + x245*x328 + x248*x327 + x249*x319 + x310*x324 - x314*x320, -x250*x326 + x255*x328 + x258*x327 + x259*x321 - x313*x324 + x314*x318, -x262*x326 + x265*x328 + x268*x327 + x269*x319 + x270*x321 + x273*x325, -x275*x326 + x278*x328 + x279*x327 + x281*x319 + x282*x321 + x283*x325, -x284*x326 + x285*x328 + x286*x327 + x287*x319 + x288*x321 + x289*x325],
            [0, 0, 0, x120*x334 + x130*x337 + x140*x340 + x150*x342 + x333*x83 - x338*x339, x163*x333 + x166*x334 + x169*x337 + x170*x340 - x171*x343 + x172*x342, x189*x333 + x192*x334 + x195*x337 - x198*x343 + x200*x342 + x205*x340, x216*x333 + x218*x334 + x221*x337 - x222*x343 + x223*x342 + x224*x340, -x230*x343 + x233*x342 + x236*x340 + x238*x337 - x332*x344 + x334*x345, -x243*x343 + x245*x342 + x248*x340 + x249*x333 - x334*x346 + x336*x344, -x250*x343 + x255*x342 + x258*x340 + x259*x334 + x333*x346 - x337*x345, -x262*x343 + x265*x342 + x268*x340 + x269*x333 + x270*x334 + x273*x337, -x275*x343 + x278*x342 + x279*x340 + x281*x333 + x282*x334 + x283*x337, -x284*x343 + x285*x342 + x286*x340 + x287*x333 + x288*x334 + x289*x337],
            [0, 0, 0, x120*x349 + x130*x351 - x133*x338 + x140*x352 + x150*x353 + x348*x83, x163*x348 + x166*x349 + x169*x351 + x170*x352 - x171*x354 + x172*x353, x189*x348 + x192*x349 + x195*x351 - x198*x354 + x200*x353 + x205*x352, x216*x348 + x218*x349 + x221*x351 - x222*x354 + x223*x353 + x224*x352, -x230*x354 + x233*x353 + x236*x352 + x238*x351 + x345*x349 - x347*x355, -x243*x354 + x245*x353 + x248*x352 + x249*x348 - x346*x349 + x350*x355, -x250*x354 + x255*x353 + x258*x352 + x259*x349 - x345*x351 + x346*x348, -x262*x354 + x265*x353 + x268*x352 + x269*x348 + x270*x349 + x273*x351, -x275*x354 + x278*x353 + x279*x352 + x281*x348 + x282*x349 + x283*x351, -x284*x354 + x285*x353 + x286*x352 + x287*x348 + x288*x349 + x289*x351],
            [0, 0, 0, x120*x356 + x130*x358 + x140*x359 - x150*x354 + x338*x360 + x357*x83, x163*x357 + x166*x356 + x169*x358 + x170*x359 + x171*x361 - x172*x354, x189*x357 + x192*x356 + x195*x358 + x198*x361 - x200*x354 + x205*x359, x216*x357 + x218*x356 + x221*x358 + x222*x361 - x223*x354 + x224*x359,     q*x362 + x230*x361 - x233*x354 + x236*x359 + x238*x358 - x357*x363,    -p*x362 + x243*x361 - x245*x354 + x248*x359 + x249*x357 + x358*x363,  x250*x361 - x255*x354 + x258*x359 + x259*x356 - x345*x358 + x346*x357,  x262*x361 - x265*x354 + x268*x359 + x269*x357 + x270*x356 + x273*x358,  x275*x361 - x278*x354 + x279*x359 + x281*x357 + x282*x356 + x283*x358,  x284*x361 - x285*x354 + x286*x359 + x287*x357 + x288*x356 + x289*x358]
        ], dtype=np.float32)

    def plot(self, fname, states, controls=None, dpi=500):

        # create figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # plot positional trajectory
        ax.plot(states[:,0], states[:,1], states[:,2], 'k.-')

        # labels
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        ax.set_zlabel('$z$ [m]')
        
        # formating
        ax.grid('False')
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # save
        fig.savefig(fname, bbox_inches='tight', dpi=500)
        plt.show()


if __name__ == '__main__':

    # instantiate Fossen model
    system = Fossen()

    # initial state
    state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    
    # controller
    controller = lambda x: np.array([1000, 1000, 0.1, 0.1])

    # # propagate system
    t, x, u = system.propagate(state, controller, 0, 50, atol=1e-4, rtol=1e-4)

    # save
    system.plot('../img/trajectory.png', x, dpi=500)
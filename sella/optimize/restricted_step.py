from typing import Optional, Union, List

import numpy as np
import inspect

from sella.peswrapper import PES, InternalPES
from .stepper import get_stepper, BaseStepper, NaiveStepper


# Classes for restricted step (e.g. trust radius, max atom displacement, etc)
class BaseRestrictedStep:
    synonyms: List[str] = []

    def __init__(
        self,
        pes: Union[PES, InternalPES],
        order: int,
        delta: float,
        method: str = 'qn',
        tol: float = 1e-15,
        maxiter: int = 1000,
        d1: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
    ):
        self.pes = pes
        self.delta = delta
        self.d1 = d1
        g0 = self.pes.get_g()

        if W is None:
            W = np.eye(len(g0))

        self.scons = self.pes.get_scons()
        # TODO: Should this be HL instead of H?
        g = g0 + self.pes.get_H() @ self.scons

        if inspect.isclass(method) and issubclass(method, BaseStepper):
            stepper = method
        else:
            stepper = get_stepper(method.lower())

        if self.cons(self.scons) - self.delta > 1e-8:
            self.P = self.pes.get_Unred().T
            dx = self.P @ self.scons
            self.stepper = NaiveStepper(dx)
            self.scons[:] *= 0
        else:
            self.P = self.pes.get_Ufree().T @ W
            d1 = self.d1
            if d1 is not None:
                d1 = np.linalg.lstsq(self.P.T, d1, rcond=None)[0]
            self.stepper = stepper(
                self.P @ g,
                self.pes.get_HL().project(self.P.T),
                order,
                d1=d1,
            )

        self.tol = tol
        self.maxiter = maxiter

    def cons(self, s, dsda=None):
        raise NotImplementedError

    def eval(self, alpha):
        s, dsda = self.stepper.get_s(alpha)
        stot = self.P.T @ s + self.scons
        val, dval = self.cons(stot, self.P.T @ dsda)
        return stot, val, dval

    def get_s(self):
        alpha = self.stepper.alpha0

        s, val, dval = self.eval(alpha)
        if val < self.delta:
            assert val > 0.
            return s, val
        err = val - self.delta

        lower = self.stepper.alphamin
        upper = self.stepper.alphamax

        for niter in range(self.maxiter):
            if abs(err) <= self.tol:
                break

            if np.nextafter(lower, upper) >= upper:
                break

            if err * self.stepper.slope > 0:
                upper = alpha
            else:
                lower = alpha

            a1 = alpha - err / dval
            if np.isnan(a1) or a1 <= lower or a1 >= upper or niter > 4:
                a2 = (lower + upper) / 2.
                if np.isinf(a2):
                    alpha = alpha + max(1, 0.5 * alpha) * np.sign(a2)
                else:
                    alpha = a2
            else:
                alpha = a1

            s, val, dval = self.eval(alpha)
            err = val - self.delta
        else:
            raise RuntimeError("Restricted step failed to converge!")

        assert val > 0
        return s, self.delta

    @classmethod
    def match(cls, name):
        return name in cls.synonyms


class TrustRegion(BaseRestrictedStep):
    synonyms = [
        'tr',
        'trust region',
        'trust-region',
        'trust radius',
        'trust-radius',
    ]

    def cons(self, s, dsda=None):
        val = np.linalg.norm(s)
        if dsda is None:
            return val

        dval = dsda @ s / val
        return val, dval


class IRCTrustRegion(TrustRegion):
    synonyms = []

    def __init__(self, *args, sqrtm=None, **kwargs):
        assert sqrtm is not None
        self.sqrtm = sqrtm
        TrustRegion.__init__(self, *args, **kwargs)
        # assert self.d1 is not None

    def cons(self, s, dsda=None):
        s = s * self.sqrtm
        if dsda is not None:
            dsda = dsda * self.sqrtm
        return TrustRegion.cons(self, s, dsda)

    # def cons(self, s, d1=None, dsda=None, dsdb=None):
    #     s = s * self.sqrtm
    #     val = np.linalg.norm(s)
    #     if d1 is None and dsda is None and dsdb is None:
    #         return val
    #     d1 = d1 * self.sqrtm
    #     dsda = dsda * self.sqrtm
    #     dsdb = dsdb * self.sqrtm
    #     proj = s @ d1
    #     dval = dsda @ s / val
    #     dproj = dsdb @ d1
    #     return val, dval, proj, dproj
    
    # def eval(self, alpha, beta):
    #     s, dsda, dsdb = self.stepper.get_s(alpha, beta)
    #     stot = self.P.T @ s + self.scons
    #     dtot = self.P.T @ self.d1
    #     val, dval, proj, dproj = self.cons(stot, dtot, self.P.T @ dsda, self.P.T @ dsdb)
    #     return stot, val, dval, proj, dproj

    # def get_s(self):
    #     # print('trying alpha = 0, beta = 0')
    #     alpha = self.stepper.alpha0
    #     beta = self.stepper.beta0
    #     s, val, dval, proj, dproj = self.eval(alpha, beta)
    #     # print('alpha', alpha, 'beta', beta, 'val', val, 'proj', proj)
    #     if val < self.delta and proj >= 0:
    #         assert val > 0.
    #         return s, val
        
    #     # print('trying alpha > 0, beta = 0')
    #     alpha = self.stepper.alpha0
    #     beta = self.stepper.beta0
    #     alphamin = self.stepper.alphamin
    #     alphamax = self.stepper.alphamax
    #     for niter in range(self.maxiter):
    #         s, val, dval, proj, dproj = self.eval(alpha, beta)
    #         # print('alpha', alpha, 'beta', beta, 'val', val, 'proj', proj)
    #         err = val - self.delta

    #         if (abs(err) <= self.tol or np.nextafter(alphamin, alphamax) >= alphamax):
    #             if proj >= 0:
    #                 assert val > 0.
    #                 return s, val
    #             else:
    #                 break
            
    #         assert alpha <= alphamax and alpha >= alphamin
    #         if err * self.stepper.alphaslope > 0:
    #             alphamax = alpha
    #         else:
    #             alphamin = alpha

    #         a1 = alpha - err / dval
    #         if np.isnan(a1) or a1 <= alphamin or a1 >= alphamax or niter > 4:
    #             a2 = (alphamin + alphamax) / 2.
    #             if np.isinf(a2):
    #                 alpha = alpha + max(1, 0.5 * alpha) * np.sign(a2)
    #             else:
    #                 alpha = a2
    #         else:
    #             alpha = a1
        
    #     # print('trying alpha = 0, beta > 0')
    #     alpha = self.stepper.alpha0
    #     beta = self.stepper.beta0
    #     betamin = self.stepper.betamin
    #     betamax = self.stepper.betamax
    #     for niter in range(self.maxiter):
    #         s, val, dval, proj, dproj = self.eval(alpha, beta)
    #         # print('alpha', alpha, 'beta', beta, 'val', val, 'proj', proj)
    #         err = val - self.delta

    #         if (abs(proj) <= self.tol or np.nextafter(betamin, betamax) >= betamax):
    #             if err <= 0:
    #                 assert val > 0.
    #                 return s, val
    #             else:
    #                 break
            
    #         assert beta <= betamax and beta >= betamin
    #         if proj * self.stepper.betaslope > 0:
    #             betamax = beta
    #         else:
    #             betamin = beta

    #         b1 = beta - proj / dproj
    #         if np.isnan(b1) or b1 <= betamin or b1 >= betamax or niter > 4:
    #             b2 = (betamin + betamax) / 2.
    #             if np.isinf(b2):
    #                 beta = beta + max(1, 0.5 * beta) * np.sign(b2)
    #             else:
    #                 beta = b2
    #         else:
    #             beta = b1

    #     # print('trying alpha > 0, beta > 0')
    #     alpha = self.stepper.alpha0
    #     beta = self.stepper.beta0
    #     alphamin = self.stepper.alphamin
    #     alphamax = self.stepper.alphamax
    #     betamin = self.stepper.betamin
    #     betamax = self.stepper.betamax
    #     for niter in range(self.maxiter):
    #         s, val, dval, proj, dproj = self.eval(alpha, beta)
    #         # print('alpha', alpha, 'beta', beta, 'val', val, 'proj', proj)
    #         err = val - self.delta

    #         if (abs(err) <= self.tol or np.nextafter(alphamin, alphamax) >= alphamax) and (abs(proj) <= self.tol or np.nextafter(betamin, betamax) >= betamax):
    #             assert val > 0.
    #             return s, val
            
    #         assert alpha <= alphamax and alpha >= alphamin
    #         if err * self.stepper.alphaslope > 0:
    #             alphamax = alpha
    #         else:
    #             alphamin = alpha
    #         assert beta <= betamax and beta >= betamin
    #         if proj * self.stepper.betaslope > 0:
    #             betamax = beta
    #         else:
    #             betamin = beta

    #         a1 = alpha - err / dval
    #         if np.isnan(a1) or a1 <= alphamin or a1 >= alphamax or niter > 4:
    #             a2 = (alphamin + alphamax) / 2.
    #             if np.isinf(a2):
    #                 alpha = alpha + max(1, 0.5 * alpha) * np.sign(a2)
    #             else:
    #                 alpha = a2
    #         else:
    #             alpha = a1
    #         b1 = beta - proj / dproj
    #         if np.isnan(b1) or b1 <= betamin or b1 >= betamax or niter > 4:
    #             b2 = (betamin + betamax) / 2.
    #             if np.isinf(b2):
    #                 beta = beta + max(1, 0.5 * beta) * np.sign(b2)
    #             else:
    #                 beta = b2
    #         else:
    #             beta = b1
    #     else:
    #         raise RuntimeError("Restricted step failed to converge!")


class RestrictedAtomicStep(BaseRestrictedStep):
    synonyms = ['ras', 'restricted atomic step']

    def __init__(self, pes, *args, **kwargs):
        if pes.int is not None:
            raise ValueError(
                "Internal coordinates are not compatible with "
                f"the {self.__class__.__name__} trust region method."
            )
        BaseRestrictedStep.__init__(self, pes, *args, **kwargs)

    def cons(self, s, dsda=None):
        s_mat = s.reshape((-1, 3))
        s_norms = np.linalg.norm(s_mat, axis=1)
        index = np.argmax(s_norms)
        val = s_norms[index]

        if dsda is None:
            return val

        dsda_mat = dsda.reshape((-1, 3))
        dval = dsda_mat[index] @ s_mat[index] / val
        return val, dval


class MaxInternalStep(BaseRestrictedStep):
    synonyms = ['mis', 'max internal step']

    def __init__(
        self, pes, *args, wx=1., wb=1., wa=1., wd=1., wo=1., **kwargs
    ):
        if pes.int is None:
            raise ValueError(
                "Internal coordinates are required for the "
                "{self.__class__.__name__} trust region method"
            )
        self.wx = wx
        self.wb = wb
        self.wa = wa
        self.wd = wd
        self.wo = wo
        BaseRestrictedStep.__init__(self, pes, *args, **kwargs)

    def cons(self, s, dsda=None):
        w = np.array(
            [self.wx] * self.pes.int.ntrans
            + [self.wb] * self.pes.int.nbonds
            + [self.wa] * self.pes.int.nangles
            + [self.wd] * self.pes.int.ndihedrals
            + [self.wo] * self.pes.int.nother
            + [self.wx] * self.pes.int.nrotations
        )
        assert len(w) == len(s)

        sw = np.abs(s * w)
        idx = np.argmax(np.abs(sw))
        val = sw[idx]

        if dsda is None:
            return val
        return val, np.sign(s[idx]) * dsda[idx] * w[idx]


_all_restricted_step = [TrustRegion, RestrictedAtomicStep, MaxInternalStep]


def get_restricted_step(name):
    for rs in _all_restricted_step:
        if rs.match(name):
            return rs
    raise ValueError("Unknown restricted step name: {}".format(name))

import numpy as np
from scipy.integrate import quad
from sft.spline.basis import B3
from sft.spline.curve_models import SplineCurve


class SimpleSplineCurve(SplineCurve):
    def __init__(self, M, basis, closed=False):
        super().__init__(M, basis, closed)

    def sample(self, sampling_rate):
        assert self.coefs is not None
        if self.closed:
            raise RuntimeError('Closed curves are not supported!')
        else:
            N = (sampling_rate * (self.M - 1)) + 1

        curve = [self.parameter_to_world(float(i) / float(sampling_rate))
                 for i in range(0, N)]
        return np.stack(curve)

    def parameter_to_world(self, t, dt=False):
        value = 0.
        for k in range(0, self.M):
            if self.closed:
                raise RuntimeError('Closed curves are not supported!')
            else:
                tval = t - k
            if (tval > -self.halfSupport and tval < self.halfSupport):
                if dt:
                    spline_value = self.basis \
                                       .firstDerivativeValue(tval)
                else:
                    spline_value = self.basis.value(tval)
                value += self.coefs[k] * spline_value
        return value

    def arc_length(self, start, end, sampling_rate=200):
        N = (sampling_rate * (self.M - 1)) + 1
        contour = [self.parameter_to_world(float(i) / float(sampling_rate))
                   for i in range(0, N)]
        contour = np.array(contour)
        src_idx = np.linalg.norm(contour - np.array(start), axis=1).argmin()
        trg_idx = np.linalg.norm(contour - np.array(end), axis=1).argmin()
        t0 = float(src_idx) / sampling_rate
        tf = float(trg_idx) / sampling_rate
        integral = quad(
            lambda t: np.linalg.norm(self.parameter_to_world(t, dt=True)),
            t0,
            tf,
            epsabs=1e-6,
            epsrel=1e-6,
            maxp1=50,
            limit=100
        )
        return integral[0], contour[src_idx: trg_idx]

    def get_coefs_from_points(self, contour_points):
        assert contour_points.ndim == 2
        assert contour_points.shape[1] == 3

        N = len(contour_points)
        phi = np.zeros((N, self.M))
        r = np.zeros((N, 3))

        if self.closed:
            raise RuntimeError('Closed curves are not supported!')
        else:
            sampling_rate = int(N / (self.M - 1))
            extra_points = N % (self.M - 1)

        for i in range(N):
            r[i] = contour_points[i]

            if i == 0:
                t = 0
            elif t < extra_points:
                t += 1. / (sampling_rate + 1.)
            else:
                t += 1. / sampling_rate

            for k in range(0, self.M):
                tval = t - k

                if (tval > -self.halfSupport and tval < self.halfSupport):
                    basis_factor = self.basis.value(tval)
                else:
                    basis_factor = 0.

                phi[i, k] += basis_factor

        c_x = np.linalg.lstsq(phi, r[:, 0], rcond=None)
        c_y = np.linalg.lstsq(phi, r[:, 1], rcond=None)
        c_z = np.linalg.lstsq(phi, r[:, 2], rcond=None)
        self._coefs = np.zeros([self.M, 3])
        for k in range(self.M):
            self._coefs[k] = np.array([c_x[0][k], c_y[0][k], c_z[0][k]])
        return


def interpolate(sampled_points, M=25, sampling_rate=200):
    np.random.seed(0)
    noise = np.random.normal(0.0, 0.5, (5, 3))
    t0 = np.array(
        [sampled_points[0] + (sampled_points[0] - sampled_points[i])
         for i in range(1, 6)]
    ) + noise
    tn = np.array(
        [sampled_points[-1] + (sampled_points[-1] - sampled_points[-i])
         for i in range(1, 6)]
    ) + noise
    sampled_points = np.concatenate([t0, sampled_points, tn])
    spline_curve = SimpleSplineCurve(M, B3(), False)
    spline_curve.get_coefs_from_points(sampled_points.astype('float32'))
    contour = spline_curve.sample(sampling_rate)
    return contour, np.concatenate([t0, tn]), spline_curve


def cut_spline(contour, u, v):
    src_idx = np.linalg.norm(contour - np.array(u), axis=1).argmin()
    trg_idx = np.linalg.norm(contour - np.array(v), axis=1).argmin()
    contour = contour[src_idx:trg_idx]
    return contour

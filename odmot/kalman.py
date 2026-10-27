"""Constant-velocity Kalman filter in (cx, cy, aspect, height) space - same as ByteTrack/SORT."""
from __future__ import annotations

import numpy as np
import scipy.linalg


class KalmanFilter:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._F = np.eye(2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt
        self._H = np.eye(ndim, 2 * ndim)
        self._wp = 1.0 / 20   # position noise, relative to box height
        self._wv = 1.0 / 160  # velocity noise, relative to box height

    def initiate(self, z: np.ndarray):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self._wp * h, 2 * self._wp * h, 1e-2, 2 * self._wp * h,
               10 * self._wv * h, 10 * self._wv * h, 1e-5, 10 * self._wv * h]
        return mean, np.diag(np.square(std))

    def multi_predict(self, means: np.ndarray, covs: np.ndarray):
        if len(means) == 0:
            return means, covs
        h = means[:, 3]
        std_p = np.stack([self._wp * h, self._wp * h, 1e-2 * np.ones_like(h), self._wp * h])
        std_v = np.stack([self._wv * h, self._wv * h, 1e-5 * np.ones_like(h), self._wv * h])
        sqr = np.square(np.r_[std_p, std_v]).T
        Q = np.zeros((len(means), 8, 8))
        Q[:, np.arange(8), np.arange(8)] = sqr
        means = means @ self._F.T
        covs = self._F @ covs @ self._F.T + Q
        return means, covs

    def update(self, mean, cov, z):
        h = mean[3]
        R = np.diag(np.square([self._wp * h, self._wp * h, 1e-1, self._wp * h]))
        proj_mean = self._H @ mean
        S = self._H @ cov @ self._H.T + R
        chol, low = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
        K = scipy.linalg.cho_solve((chol, low), (cov @ self._H.T).T, check_finite=False).T
        new_mean = mean + (z - proj_mean) @ K.T
        new_cov = cov - K @ S @ K.T
        return new_mean, new_cov

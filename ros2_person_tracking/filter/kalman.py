import numpy as np


class Kalman:
    def __init__(self, mat_f, mat_q, mat_h, mat_r, mat_f_before=None, mat_q_before=None):
        self.mat_f = mat_f
        self.mat_f_before = mat_f_before
        self.mat_h = mat_h
        self.mat_q = mat_q
        self.mat_q_before = mat_q_before
        self.mat_r = mat_r

        self.mat_i = np.eye(self.mat_f.shape[1])

    # TODO: Consider vectorized predict and update computation
    def predict(self, vec_x, mat_p):
        vec_x = self.mat_f @ vec_x
        mat_p = self.mat_f @ mat_p @ self.mat_f.T + self.mat_q

        return vec_x, mat_p

    def update(self, vec_x, mat_p, vec_z):
        mat_s = self.mat_h @ mat_p @ self.mat_h.T + self.mat_r
        mat_k = mat_p @ self.mat_h.T @ np.linalg.inv(mat_s)

        # print(mat_k)

        vec_x = vec_x + mat_k @ (vec_z - self.mat_h @ vec_x)
        mat_p = (self.mat_i - mat_k @ self.mat_h) @ mat_p

        return vec_x, mat_p

    def retrodict(self, vec_x, mat_p, mat_p_before):
        # See https://de.mathworks.com/help/fusion/ref/trackingkf.retrodict.html
        vec_x = self.mat_f @ vec_x

        mat_p_prior = self.mat_f_before @ mat_p_before @ self.mat_f_before.T + self.mat_q_before
        mat_p_prior_inv = np.linalg.inv(mat_p_prior)
        mat_s_inv = mat_p_prior_inv - mat_p_prior_inv @ mat_p @ mat_p_prior_inv
        mat_p_xv = self.mat_q - mat_p_prior @ mat_s_inv @ self.mat_q
        mat_p = self.mat_f @ (mat_p + self.mat_q - mat_p_xv - mat_p_xv.T) @ self.mat_f.T

        return vec_x, mat_p, mat_p_xv

    def retrocorrect(self, vec_x, mat_p, vec_z, vec_x_retrodicted, mat_p_retrodicted, mat_p_xv):
        mat_p_xz = (mat_p - mat_p_xv) @ self.mat_f.T @ self.mat_h.T
        mat_s = self.mat_h @ mat_p_retrodicted @ self.mat_h.T + self.mat_r
        mat_w = mat_p_xz @ np.linalg.inv(self.mat_h @ mat_p_retrodicted @ self.mat_h.T + self.mat_r)

        # print(mat_w)

        vec_x = vec_x + mat_w @ (vec_z - self.mat_h @ vec_x_retrodicted)
        mat_p = mat_p - mat_p_xz @ np.linalg.inv(mat_s) @ mat_p_xz.T

        return vec_x, mat_p

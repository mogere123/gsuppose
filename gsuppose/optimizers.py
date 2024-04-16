# -*- coding: utf-8 -*-

from typing import Sequence
import numpy as np


def get_all_subclasses(cls):
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


class GDOptimizer:

    def initialize(self, gsuppose_object):
        """Initializes all the variables needed for the optimizer to work. Method called during the start of the
        algorithm. Each optimizer must define its own method.

        Parameters
        ----------
        gsuppose_object : `gsuppose.api.GSUPPOSe`
            GSUPPOSe object that runs the algorithm.
        """
        pass

    def batch_update(self, source_indices: Sequence[int], grads: Sequence[Sequence[float]], epoch: int,
                     gsuppose_object) -> np.ndarray:
        """Computes the source displacements for the current batch. Each optimizer must define its own method.

        Parameters
        ----------
        source_indices : Sequence[int]
            Indices of the virtual sources within the current batch.
        grads : numpy.ndarray
            Gradients for the current batch.
        epoch : int
            Current epoch.
        gsuppose_object : `gsuppose.api.GSUPPOSe`
            GSUPPOSe object that runs the algorithm.

        Returns
        -------
        numpy.ndarray
            Displacements for the current batch.
        """
        pass


class VanillaGD(GDOptimizer):

    def batch_update(self, source_indices: Sequence[int], grads: Sequence[Sequence[float]], epoch: int,
                     gsuppose_object) -> np.ndarray:
        return - gsuppose_object.global_scale * grads


class Adam(GDOptimizer):

    def __init__(self, beta1: float = 0.9, beta2: float = 0.99, epsilon: float = 1e-8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None

    def initialize(self, gsuppose_object):
        self.m = np.zeros_like(gsuppose_object.positions)
        self.v = np.zeros_like(gsuppose_object.positions)

    def batch_update(self, source_indices: Sequence[int], grads: Sequence[Sequence[float]], epoch: int,
                     gsuppose_object) -> np.ndarray:
        self.m[source_indices] = self.beta1 * self.m[source_indices] + (1 - self.beta1) * grads
        self.v[source_indices] = self.beta2 * self.v[source_indices] + (1 - self.beta2) * np.power(grads, 2)

        corr_m = self.m[source_indices] / (1 - np.power(self.beta1, epoch + 1))
        corr_v = self.v[source_indices] / (1 - np.power(self.beta2, epoch + 1))

        return - gsuppose_object.global_scale * corr_m / np.sqrt(corr_v + self.epsilon)


class Nadam(Adam):

    def batch_update(self, source_indices: Sequence[int], grads: Sequence[Sequence[float]], epoch: int,
                     gsuppose_object) -> np.ndarray:
        self.m[source_indices] = self.beta1 * self.m[source_indices] + (1 - self.beta1) * grads
        self.v[source_indices] = self.beta2 * self.v[source_indices] + (1 - self.beta2) * np.power(grads, 2)
        corr_m = self.m[source_indices] / (1 - np.power(self.beta1, epoch + 1))
        corr_v = self.v[source_indices] / (1 - np.power(self.beta2, epoch + 1))
        corr_m = (1 - self.beta1) * grads + self.beta1 * corr_m

        return - gsuppose_object.global_scale * corr_m / np.sqrt(corr_v + self.epsilon)


class VectorialNadam(Adam):

    def batch_update(self, source_indices: Sequence[int], grads: Sequence[Sequence[float]], epoch: int,
                     gsuppose_object) -> np.ndarray:
        self.m[source_indices] = self.beta1 * self.m[source_indices] + (1 - self.beta1) * grads
        self.v[source_indices] = self.beta2 * self.v[source_indices] + (1 - self.beta2) * \
                                 np.power(np.linalg.norm(grads, ord=2, axis=-1, keepdims=True), 2)
        corr_m = self.m[source_indices] / (1 - np.power(self.beta1, epoch + 1))
        corr_v = self.v[source_indices] / (1 - np.power(self.beta2, epoch + 1))
        corr_m = (1 - self.beta1) * grads + self.beta1 * corr_m

        return - gsuppose_object.global_scale * corr_m / np.sqrt(corr_v + self.epsilon)


class AMSGrad(Adam):

    def batch_update(self, source_indices: Sequence[int], grads: Sequence[Sequence[float]], epoch: int,
                     gsuppose_object) -> np.ndarray:
        new_m = self.beta1 * self.m[source_indices] + (1 - self.beta1) * grads
        new_v = self.beta2 * self.v[source_indices] + (1 - self.beta2) * np.power(grads, 2)
        new_v = np.max([self.v[source_indices], new_v], axis=0)

        self.m[source_indices] = new_m
        self.v[source_indices] = new_v

        return - gsuppose_object.global_scale * new_m / np.sqrt(new_v + self.epsilon)


# Leave this at the end:
OPTIMIZERS = {cls.__name__.lower(): cls for cls in get_all_subclasses(GDOptimizer)}

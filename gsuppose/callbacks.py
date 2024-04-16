# -*- coding: utf-8 -*-

import numpy as np


class ReduceGSOnPlateau:
    MODES = ["min", "max"]

    def __init__(self,
                 factor: float = 0.1,
                 min_delta: float = 0.1,
                 relative: bool = False,
                 patience: int = 100,
                 cooldown: int = 50,
                 monitor: str = "fitness",
                 target: str = "global_scale",
                 mode: str = "min",
                 vmin: float = -np.inf,
                 vmax: float = np.inf,
                 verbose: int = 1):
        self.factor = float(factor)
        self.min_delta = float(min_delta)
        self.relative = bool(relative)
        self.patience = int(patience)
        self.cooldown = int(cooldown)
        self.monitor = str(monitor)
        self.target = str(target)
        if mode.lower() not in self.MODES:
            raise ValueError(f"El argumento 'mode' debe tomar alguno de los valores {self.MODES}. Se recibió '{mode}'.")
        else:
            self.mode = str(mode).lower()
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.verbose = int(verbose)

        self.cooldown_counter = 0
        self.wait_counter = 0
        self.best = 0.0
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode == 'min':
            if self.relative:
                self.monitor_op = lambda current, best: np.greater(1 - current / best, self.min_delta)
            else:
                self.monitor_op = lambda current, best: np.greater(best - current, self.min_delta)
            self.reduce_op = np.min
            self.best = np.inf
        elif self.mode == 'max':
            if self.relative:
                self.monitor_op = lambda current, best: np.greater(1 - current / best, self.min_delta)
            else:
                self.monitor_op = lambda current, best: np.greater(current - best, self.min_delta)
            self.reduce_op = np.max
            self.best = -np.inf
        self.cooldown_counter = 0
        self.wait_counter = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0

    def callback(self, parent_object, epoch: int):
        current = self.reduce_op(parent_object.history[self.monitor][epoch + 1])

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait_counter = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait_counter = 0
        elif not self.in_cooldown():
            self.wait_counter += 1
            if self.wait_counter >= self.patience:
                old = float(parent_object.__getattribute__(self.target))
                if old > self.vmin:
                    new = old * self.factor
                    new = np.clip(new, self.vmin, self.vmax)
                    parent_object.__setattr__(self.target, new)

                    if self.verbose > 0:
                        print(f"\nEpoch {epoch}: ReduceGSOnPlateau setting '{self.target}' to {new:.3e}")

                    self.cooldown_counter = self.cooldown
                    self.wait_counter = 0

# -*- coding: utf-8 -*-

from gsuppose.fitness import mse, dmse
from gsuppose.style import DEFAULT_PLOT_STYLE
from gsuppose.optimizers import OPTIMIZERS, GDOptimizer
from catmu.api import ConvolutionManagerCPU, ConvolutionManagerGPU, ConvolutionManager

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Union, List, Tuple, Dict, Callable, Sequence
from pathlib import Path
from queue import Empty
from multiprocessing import JoinableQueue, Process, Event


class GSUPPOSe:
    """Single image SUPPOSe deconvolution by means of gradient descent optimization.

    Main class for performing SUPPOSe deconvolution over a single image (in contrast with multiple image in parallel).
    It is initialized with the sample, the PSF and all the parameters that configures the algorithm, and then is
    started by its `run` method.

    Parameters
    ----------
    sample : numpy.ndarray
        Sample image (as a Numpy array) to be processed.
    psf : numpy.ndarray
        Point Spread Function (or PSF) of the optical system (as a Numpy array). It is assumed that the center of the
        PSF is at the position `(N / 2, M / 2)`, with `N, M` the shape of the PSF array.
    initial_positions : numpy.ndarray
        Initial positions of the virtual sources. It must be a Numpy array of shape `(nsources, 2)` so the first
        dimension is the virtual source index and the second dimension is the axis (in `y, x` order).
    optimizer : Union[str, type, gsuppose.GDOptimizer], optional
        Gradient descent optimizer used in the algorithm. It must be a string with a valid optimizer name (in lower
        case) or a `gsuppose.GDOptimizer` instance or subclass. (Default: `'vanillagd'`.)
    sample_pixel_size : Tuple[float, float], optional
        Pixel size of the sample (in arbitrary units) for each axis. Only the ratio between pixel sizes of the sample
        and PSF is used to compute the convolution. (Default: `(1, 1)`.)
    psf_pixel_size : Tuple[float, float], optional
        Pixel size of the PSF (in arbitrary units) for each axis. Only the ratio between pixel sizes of the sample and
        PSF is used to compute the convolution. (Default: `(1, 1)`.)
    fitness_function : Callable, optional
        Function that measures the pixel-wise similarity between the sample and the SUPPOSe reconstruction. It must
        have the following signature: `fitness_function(sample, reconstruction) -> numpy.array`, where the returned
        array should be of the same shape as the sample. (Default: `gsuppose.fitness.chi_squared`.)
    dfitness_function : Callable, optional
        Pixel-wise derivative of the fitness function with respect to the SUPPOSe reconstruction. It's paired with the
        value of `fitness_function` and it must have the same signature. (Default: `gsuppose.fintess.dchi_squared`.)
    normalize_input : str, optional
        String that indicates the normalization applyied to the sample before running the algorithm. It can be any of
        the following values:
            * `'mean'`: subtract the mean value of the sample.
            * `'std'`: divide the sample by its standard deviation.
            * `'both'`: subtract its mean and then divide it by its standard deviation.
            * `'none'`: do not perform any normalization.
        (Default: `'none'`.)
    allow_beta : bool, optional
        Alternative way to address images with nonzero background values. If `True`, it adds a new parameter `beta`
        (fitted by least ssquares, as `alpha`), so the SUPPOSe reconstruction becomes `reconstruction = alpha *
        convolution + beta`. (Default: `False`.)
    batch_size : int, optional
        Number of virtual sources per gradient update. Since the convolution of each source in a gradient update runs
        in parallel within the selected device (CPU or GPU), choosing its value poses a compromise between the number
        of gradient updates per epoch and the degree of parallelization of the convolution operation. (Default: `1`.)
    global_scale : float, optional
        Also called learning rate. A factor that scales all the gradient updates. (Default: `1E-1`.)
    normalize_gradient : bool, optional
        If `True`, only the directions of the gradients are calculated, so in each update the displacements of each
        virtual source are all of size `global_scale`. (Defalut: `False`.)
    device_index : int, Sequence[int] or None, optional
        Device used to compute the convolution: `-1` corresponds to CPU, higher integer values corresponds to valid GPU
        devices. A maximum of 3 devices are allowed by passing a list of integers, in which case the job is distributed
        between all the devices. (Default: `-1`.)
    *args, **kwargs
        Additional arguments passed to the initializer of the optimizer (if the argument `optimizer` is a string or a
        `gsuppose.optimizers.GDSuppose` subclass). (See `gsuppose.optimizers` documentation.)
    """

    STOP_METHODS = ["success_rate", "fitness", "mean_displacement", "custom"]

    def __init__(self,
                 sample: np.ndarray,
                 psf: np.ndarray,
                 initial_positions: np.ndarray = None,
                 optimizer: Union[str, type, GDOptimizer] = "vanillagd",
                 sample_pixel_size: Tuple[float, float] = (1, 1),
                 psf_pixel_size: Tuple[float, float] = (1, 1),
                 fitness_function: Callable = mse,
                 dfitness_function: Callable = dmse,
                 normalize_input: str = 'none',
                 allow_beta: bool = False,
                 batch_size: int = 1,
                 global_scale: float = 1E-1,
                 normalize_gradient: bool = False,
                 device_index: Union[int, Sequence[int]] = None,
                 *args, **kwargs):
        # Valores iniciales
        self.raw_sample = None
        self._sample = None
        self.sample_pixel_size = None
        self.raw_psf = None
        self._psf = None
        self._luts = {}
        self.psf_pixel_size = None
        self._subtract_mean = None
        self._normalize_input = None
        self.allow_beta = None
        self.initial_positions = None
        self._optimizer = None
        self.fitness_function = None
        self.dfitness_function = None

        # Variables de ejecución
        self.positions = None
        self.alpha = None
        self.beta = None
        self.convolution = None
        self.residue = None
        self.fitness = None
        self.history = {k: None for k in ["epoch", "positions", "fitness", "alpha", "beta", "global_scale",
                                          "mean_displacement", "max_displacement", "success_rate"]}
        self.time = {k: None for k in ["start", "end", "remaining", "iter_start", "iter_stop", "iter_total",
                                       "iter_mean"]}
        self.iter_time = {key: None for key in ["prepare_batches", "convolve_batches", "compute_gradient",
                                                "update_positions", "convolve_solution", "fit_parameters",
                                                "compute_fitness", "update_plot", "callbacks", "others"]}
        self._device_index = None
        self.convmngr: List[ConvolutionManager] = []
        self.batch_convolution = None

        # Parámetros del algoritmo
        self.epochs = None
        self.batch_size = None
        self._stop_method = None
        self._stop_limit = None
        self.global_scale = None
        self.normalize_gradient = None

        # Variables de gráfico
        self.plotter = ProcessPlotter()

        # Complementos
        self.callbacks = []

        # Configuro con los argumentos
        self.sample = sample
        self.sample_pixel_size = sample_pixel_size
        self.normalize_input = normalize_input
        self.allow_beta = allow_beta
        self.initial_positions = initial_positions
        self.psf = psf
        self.psf_pixel_size = psf_pixel_size
        self.set_optimizer(optimizer, *args, **kwargs)
        self.fitness_function = fitness_function
        self.dfitness_function = dfitness_function

        self.batch_size = min(batch_size, self.nsources)
        self.global_scale = global_scale
        self.normalize_gradient = normalize_gradient
        self.device_index = device_index

    @property
    def sample(self):
        """`numpy.ndarray`: Image to be processed. It must be a 2 dimensional Numpy array."""
        return self._sample

    @sample.setter
    def sample(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"La propiedad 'sample' debe ser un array de Numpy. Se recibió: {value}.")

        self.raw_sample = np.copy(value)
        self._sample = np.copy(self.raw_sample)

        if self.normalize_input in ["mean", "both"]:
            self._sample = self._sample - self._sample.mean()
        if self.normalize_input in ["std", "both"]:
            self._sample = self._sample / self._sample.std()

    @property
    def psf(self):
        """`numpy.ndarray`: point spread function of the optical system. Convolution kernel used in the algorithm. It
        must be a 2 dimensional array."""
        return self._psf

    @psf.setter
    def psf(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"La propiedad 'psf' debe ser tipo numpy.ndarray. Se recibió {value}.")

        self.raw_psf = np.copy(value)
        self._psf = np.copy(self.raw_psf)

        self._luts[0] = self._psf
        for dim in range(self.ndim):
            self._luts[dim + 1] = np.gradient(self._psf, axis=self.ndim - 1 - dim)

    @property
    def optimizer(self):
        """`gsuppose.optimizers.GDOptimizer`: the instance of the gradient descent optimizer used in the algorithm."""
        return self._optimizer

    def set_optimizer(self, value: Union[str, type, GDOptimizer], *args, **kwargs):
        if isinstance(value, str):
            if value.lower() in OPTIMIZERS.keys():
                self._optimizer = OPTIMIZERS[value.lower()](*args, **kwargs)
            else:
                raise ValueError(f"The optimizer must be a `GDOptimizer` class or instance or a string with a valid "
                                 f"optimizer name: '{', '.join(OPTIMIZERS.keys())}'. The value received was: "
                                 f"'{value}'.")
        elif isinstance(value, type) and issubclass(value, GDOptimizer):
            self._optimizer = value(*args, **kwargs)
        elif isinstance(value, GDOptimizer):
            self._optimizer = value
            if args or kwargs:
                print(f"Warning: `set_optimizer` method called with additinal arguments or keyword arguments, but they "
                      f"are not used since the specified optimizer was already a `GDOptimizer` instance.")
        else:
            raise ValueError(f"The optimizer must be a `GDOptimizer` class or instance or a string with a valid "
                             f"optimizer name: '{', '.join(OPTIMIZERS.keys())}'. The value received was: '{value}'.")

    @property
    def nsources(self):
        """`int`: number of virtual sources."""
        return self.initial_positions.shape[0]

    @property
    def ndim(self):
        """`int`: dimension of the system. It is determined from the shape of the initial positions."""
        return self.initial_positions.shape[1]

    @property
    def nbatches(self):
        """`int`: number of virtual sources per gradient update. Since the convolution of each source in a gradient
        update runs in parallel within the selected device (CPU or GPU), choosing its value poses a compromise between
        the number of gradient updates per epoch and the degree of parallelization of the convolution operation."""
        return self.nsources // self.batch_size

    @property
    def subtract_mean(self):
        """`bool`: wheter to subtract or not the mean value of the convolution. Its value is first determined by the
        propertie `normalize_input`."""
        return self._subtract_mean

    @subtract_mean.setter
    def subtract_mean(self, value):
        if not isinstance(value, bool):
            raise TypeError(f"La propiedad 'subtract_mean' debe ser de tipo Callable. Se recibió {value}.")
        self._subtract_mean = value
        self._normalize_input = "mean" if value else "none"

        if self._sample is not None:
            self.sample = self.raw_sample

    @property
    def normalize_input(self):
        """str: String that indicates the normalization applyied to the sample before running the algorithm. It can be
        any of the following values:
            * `'mean'`: subtract the mean value of the sample.
            * `'std'`: divide the sample by its standard deviation.
            * `'both'`: subtract its mean and then divide it by its standard deviation.
            * `'none'`: do not perform any normalization."""
        return self._normalize_input

    @normalize_input.setter
    def normalize_input(self, value):
        allowed_values = ["none", "mean", "std", "both"]

        if not isinstance(value, str):
            raise TypeError(f"La propiedad 'normalize_input' debe ser de tipo str. Se recibió {value}.")
        elif value.lower() not in allowed_values:
            raise ValueError(f"La propiedad 'normalize_input' debe tomar alguno de los siguientes valores: "
                             f"{allowed_values}.")
        self._normalize_input = value.lower()
        self._subtract_mean = self._normalize_input in ["mean", "both"]

        if self._sample is not None:
            self.sample = self.raw_sample
        if self._psf is not None:
            self.psf = self.raw_psf

    @property
    def stop_method(self):
        """`str`: name of a variable to check for raising the stop condition of the algorithm. Its value must be the
        name of any of the variables calculated during the algorithm (see class attribute `STOP_CONDITIONS`), so the
        algorithm stops either when `max_epochs` occurs or when this variable is below `stop_limit`."""
        return self._stop_method

    @stop_method.setter
    def stop_method(self, value):
        if value is None:
            pass
        elif not isinstance(value, str):
            raise TypeError(f"La propiedad 'stop_method' debe ser de tipo str. Se recibió {value}.")
        elif value.lower() not in self.history.keys():
            raise ValueError(f"La propiedad 'stop_method' debe tomar alguno de los siguientes valores: "
                             f"{[k for k in self.history.keys()]}. Se recibió {value}.")
        self._stop_method = value

    @property
    def stop_limit(self):
        """float`: if a `stop_method` is specified, the algorithm stops when the configured variable is below this
        value."""
        return self._stop_limit

    @stop_limit.setter
    def stop_limit(self, value):
        if value is None:
            self._stop_limit = None
        elif not isinstance(value, str) and not np.isscalar(value):
            raise TypeError(f"La propiedad 'stop_limit' debe ser un escalar de Numpy o un str con un bloque de código "
                            f"que devuelva un bool. Se recibió {value}.")
        elif isinstance(value, str):
            self._stop_limit = value
        else:
            self._stop_limit = float(value)

    @property
    def device_index(self):
        """`List[int]`: list of devices used to compute the convolution. `-1` corresponds to CPU, higher integer values
        corresponds to valid GPU devices. The job is distributed by sepparating the 3 convolution kernels (the PSF and
        its derivatives in each direction) as follows:
            * If no GPU is specified, all the kernels are assigned to the CPU.
            * If no CPU is specified, each kernel is assigned to the first 3 GPUs specified.
            * If both CPU and GPUs are specified, the PSF kernel is assigned to CPU while the rest of the kernels (its
              derivatives) are assigned to the first 2 GPUs specified."""
        return self._device_index

    @device_index.setter
    def device_index(self, device_index):
        if device_index is None:
            self.convmngr = []
        elif isinstance(device_index, int):
            device_index = [device_index]
        elif isinstance(device_index, Sequence):
            device_index = [int(v) for v in device_index]
        else:
            raise TypeError(f"La propiedad 'device_index' debe ser None, de tipo int o una secuencia de valores de "
                            f"tipo int. Se recibió {device_index}.")

        use_cpu = (-1 in device_index)  # El índice '-1' corresponde a utilizar la CPU. Veo si está en la lista.
        [device_index.remove(-1) for _ in range(device_index.count(-1))]  # Elimino todos los '-1' de la lista.
        n_devices = len(device_index)  # El resto son dispositivos GPU
        use_gpu = n_devices > 0  # Si hay dispositivos, se usa la GPU

        # Asigno los 3 ConvolutionManagers: uno para la PSF y dos para los gradientes en cada dirección
        for i in range(self.ndim + 1):
            # Si se usa CPU y GPUs, asigno el ConvolutionManager de la PSF a la CPU. Si no se usa GPUs, le asigno todos
            # a la CPU.
            if (i == 0 and use_cpu) or not use_gpu:
                self.convmngr += [ConvolutionManagerCPU(open_mp=True, debug=False)]
            # Caso contrario, distribuyo los ConvolutionManagers restantes entre las GPUs indicadas.
            else:
                idx = device_index[i % n_devices]  # Esta línea permite distribuirlos en hasta 3 placas
                self.convmngr += [ConvolutionManagerGPU(device=idx, n_streams=self.nbatches, block_size=8, debug=False)]

    def stop_condition(self, current_iter: int):
        """`bool`: variable that indicates if the stop condition is verified for the current epoch."""
        if self.stop_method is not None:
            return self.history[self.stop_method][current_iter] < self._stop_limit
        else:
            return False

    def batch_gradient(self, source_indices: Sequence[int]) -> np.ndarray:
        """Computes the gradients for the current batch.

        Parameters
        ----------
        source_indices : Sequence[int]
            Indices of the virtual sources within the current batch.

        Returns
        -------
        numpy.ndarray
            Gradients for the current batch.
        """
        dfitness = self.dfitness_function(self.alpha * self.convolution + self.beta, self.sample)
        grads = np.sum(- self.alpha * dfitness * self.batch_convolution[:, source_indices], axis=(-1, -2)).T

        if self.normalize_gradient:
            norm = np.linalg.norm(grads, ord=2, axis=-1, keepdims=True)
            grads = np.divide(grads, norm, where=norm != 0.0)

        return grads

    def batch_update(self, source_indices: Sequence[int], grads: Sequence[Sequence[float]], epoch: int) -> np.ndarray:
        """Computes the source displacements for the current batch. Calls the corresponding method of the optimizer.

        Parameters
        ----------
        source_indices : Sequence[int]
            Indices of the virtual sources within the current batch.
        grads : numpy.ndarray
            Gradients for the current batch.
        epoch : int
            Current epoch.

        Returns
        -------
        numpy.ndarray
            Displacements for the current batch.
        """
        return self.optimizer.batch_update(source_indices, grads, epoch, self)

    def convolve(self, positions: np.ndarray) -> np.ndarray:
        """Computes the convolution with the PSF.

        Parameters
        ----------
        positions : numpy.array
            Positions of the virtual sources to convolve.

        Returns
        -------
        numpy.ndarray
            Result of the convolution.
        """
        convolution = self.convmngr[0].sync_convolve(positions[np.newaxis, :, :])[0]

        if self.subtract_mean:
            convolution -= convolution.mean()

        return convolution

    def fit_parameters(self, convolution: np.ndarray) -> Tuple[float, float]:
        """Calculates the parameters `alpha` and `beta` for the current epoch by least-squares.

        Parameters
        ----------
        convolution : numpy.ndarray
            Convolution of the virtual sources with the PSF for the current epoch.

        Returns
        -------
        Tuple[float, float]
            The values of `alpha` and `beta` respectively.
        """
        if self.allow_beta:
            alpha = (np.sum(self._sample * convolution) - (np.sum(self._sample) * np.sum(convolution) /
                                                           self._sample.size)) / (
                            np.sum(convolution ** 2) - np.sum(convolution) ** 2 / self._sample.size)
            beta = np.sum(self._sample - alpha * convolution) / self._sample.size
        else:
            alpha = np.sum(self._sample * convolution) / np.sum(convolution ** 2)
            beta = 0
        return alpha, beta

    def run(self, epochs: int, stop_method: str = None, stop_limit: Union[float, str] = None, report_every: int = 10,
            stdout: Union[str, Path] = None, save_path: Union[str, Path, None] = None, plot: bool = False,
            save_halfway_plots: bool = False):
        """Starts the SUPPOSe algorithm. During the algorithm, the results are saved in the `history` attribute and,
        at the end, they are also saved in the disk (if a save path is specified).

        Parameters
        ----------
        epochs : int
            Maximum number of iterations of the algorithm.
        stop_method : str, optional
            Configures (along with `stop_limit`) an additional stop condition of the algorithm. Its value must be the
            name of any of the variables calculated during the algorithm (see class attribute `STOP_CONDITIONS`).
            (Default: `None`, which means that no stop condition is configured.)
        stop_limit : float, optional
            If a `stop_method` is specified, the algorithm also stops when the configured variable is below this value.
            (Default: `None`, which means that no stop condition is configured.)
        report_every : int, optional
            Update interval (in epochs) for both the report printed in console and the figure. (Default: `10`.)
        save_path : Union[str, Path, None], optional
             String or Path object that points to a valid path for saving the results and figures of the run. (Default:
             `None`, which means that no results are saved.)
        plot : bool, optional
            Wheter to show or not a figure for monitoring the run. (Default: `False`.)
        save_halfway_plots : bool, optional
            Wheter to save each update of the monitor figure in a different file. (Default: `False`, which means that
            only the first and last updates are saved.)
        """

        # Asigno las variables de la corrida
        self.epochs = epochs
        self.stop_method = stop_method
        self.stop_limit = stop_limit

        # Inicializo rutas de salida
        if save_path is not None:
            save_path = Path(save_path).resolve()
            save_folder = save_path.parent.resolve()
            save_folder.mkdir(exist_ok=True, parents=True)
            save_name = save_path.stem

            # Guardo archivo de configuración
            self.get_config(output=save_folder / (save_name + "_config.txt"))
        else:
            save_folder = None
            save_name = None

        if stdout is not None:
            fileout = open(stdout, mode="w+")
        else:
            fileout = None

        # Inicializo convoluciones
        for i, convmngr in enumerate(self.convmngr):
            convmngr.prepare_lut_psf(psf=self._luts[i],
                                     image_size=self.sample.shape,
                                     image_pixel_size=self.sample_pixel_size,
                                     psf_pixel_size=self.psf_pixel_size)

        # Inicializo variables de la corrida
        self.positions = np.copy(self.initial_positions)
        self.convolution = self.convolve(self.positions)
        self.batch_convolution = np.zeros((self.ndim, self.nsources) + self.sample.shape)
        self.alpha, self.beta = self.fit_parameters(self.convolution)
        self.residue = self.fitness_function(self.alpha * self.convolution + self.beta, self._sample)
        self.fitness = np.mean(self.residue, axis=(-1, -2))
        self.time = {key: 0.0 for key in self.time.keys()}
        self.iter_time = {key: 0.0 for key in self.iter_time.keys()}

        source_order = np.arange(self.nsources)

        # Inicializo historia de la corrida
        self.history["epoch"] = np.zeros(self.epochs + 1, dtype=int)
        self.history["positions"] = np.zeros((self.epochs + 1,) + self.positions.shape)
        self.history["mean_displacement"] = np.zeros(self.epochs)
        self.history["max_displacement"] = np.zeros(self.epochs)
        self.history["success_rate"] = np.zeros(self.epochs)
        self.history["fitness"] = np.zeros(self.epochs + 1)
        self.history["alpha"] = np.zeros(self.epochs + 1)
        self.history["beta"] = np.zeros(self.epochs + 1)
        self.history["global_scale"] = np.zeros(self.epochs + 1)

        self.history["epoch"][0] = -1
        self.history["fitness"][0] = self.fitness
        self.history["alpha"][0] = self.alpha
        self.history["beta"][0] = self.beta
        self.history["global_scale"][0] = self.global_scale

        # Inicializo optimizador
        self.optimizer.initialize(self)

        # Inicializo figuras
        if plot:
            self.plotter.start(self.sample, self.residue, self.initial_positions, save_path, save_halfway_plots,
                               save_folder, save_name)
            self.plotter.ready.wait()

        # Iteración principal (épocas)
        self.time["start"] = time.time()

        print(f"SGD-SUPPOSe started at {time.strftime('%d/%m%Y %H:%M:%S')}:\n"
              f"    Epochs: {epochs}\n"
              f"    Batch size: {self.batch_size}\n"
              f"    Image shape: {self.sample.shape}\n"
              f"    PSF shape: {self._luts[0].shape}\n"
              f"    Devices: {self.device_index}\n",
              file=fileout)

        epoch = 0
        for epoch in range(self.epochs):
            self.time["iter_start"] = time.time()

            # Actualizo el orden del lote
            tic = time.time()
            np.random.shuffle(source_order)
            batch_order = np.reshape(source_order[0:self.nbatches * self.batch_size], (self.nbatches, self.batch_size))
            toc = time.time()
            self.iter_time["prepare_batches"] = (self.iter_time["prepare_batches"] * epoch + (toc - tic)) / \
                                                (epoch + 1)

            # Convolución del gradiente
            tic = time.time()
            for dim in range(self.ndim):
                self.convmngr[dim + 1].async_convolve(self.positions[:, np.newaxis, :])
            for dim in range(self.ndim):
                self.batch_convolution[dim] = self.convmngr[dim + 1].sync_get_results()
            toc = time.time()
            self.iter_time["convolve_batches"] = (self.iter_time["convolve_batches"] * epoch + (toc - tic)) / \
                                                 (epoch + 1)

            # Actualización por lotes
            time_gradient = 0.0
            time_pos_update = 0.0
            time_convolution = 0.0
            time_parameters = 0.0
            time_fitness = 0.0

            for j in range(self.nbatches):
                # Calculo gradientes
                tic = time.time()
                grads = self.batch_gradient(source_indices=batch_order[j])
                toc = time.time()
                time_gradient += toc - tic

                # Actualizo las posiciones usando la rutina del optimizador
                tic = time.time()
                dr = self.batch_update(source_indices=batch_order[j], grads=grads, epoch=epoch)
                self.positions[batch_order[j]] += dr
                self.history["mean_displacement"][epoch] += np.mean(np.linalg.norm(dr, ord=2, axis=-1), axis=-1)
                self.history["max_displacement"][epoch] += np.max(np.linalg.norm(dr, ord=2, axis=-1), axis=-1)
                toc = time.time()
                time_pos_update += toc - tic

                # Actualizo convolución
                tic = time.time()
                self.convolution = self.convolve(self.positions)
                toc = time.time()
                time_convolution += toc - tic

                # Ajusto parámetros
                tic = time.time()
                self.alpha, self.beta = self.fit_parameters(self.convolution)
                toc = time.time()
                time_parameters += toc - tic

                # Calculo el fitness para el lote actual
                tic = time.time()
                old_fitness = self.fitness
                self.residue = self.fitness_function(self.alpha * self.convolution + self.beta, self.sample)
                self.fitness = np.mean(self.residue, axis=(-1, -2))
                self.history["success_rate"][epoch] += (self.fitness <= old_fitness).astype(int) / self.nbatches
                toc = time.time()
                time_fitness += toc - tic

            # Actualizo tiempos detallados
            self.iter_time["compute_gradient"] = \
                (self.iter_time["compute_gradient"] * epoch + time_gradient) / (epoch + 1)
            self.iter_time["update_positions"] = \
                (self.iter_time["update_positions"] * epoch + time_pos_update) / (epoch + 1)
            self.iter_time["convolve_solution"] = \
                (self.iter_time["convolve_solution"] * epoch + time_convolution) / (epoch + 1)
            self.iter_time["fit_parameters"] = \
                (self.iter_time["fit_parameters"] * epoch + time_parameters) / (epoch + 1)
            self.iter_time["compute_fitness"] = \
                (self.iter_time["compute_fitness"] * epoch + time_fitness) / (epoch + 1)

            # Actualizo historia
            self.history["epoch"][epoch + 1] = epoch
            self.history["positions"][epoch + 1] = self.positions
            self.history["fitness"][epoch + 1] = self.fitness
            self.history["alpha"][epoch + 1] = np.squeeze(self.alpha)
            self.history["beta"][epoch + 1] = np.squeeze(self.beta)
            self.history["global_scale"][epoch + 1] = self.global_scale

            # Actualizo gráfico
            if (report_every and plot and epoch % report_every == 0) or self.plotter.missed.is_set():
                tic = time.time()
                self.plotter.add_to_queue(epoch, self.residue, self.positions, self.history)
                toc = time.time()
                self.iter_time["update_plot"] = (self.iter_time["update_plot"] * epoch + (toc - tic)) / \
                                                (epoch + 1)

            # Ejecuto complementos
            tic = time.time()
            for callback in self.callbacks:
                callback.callback(self, epoch)
            toc = time.time()
            self.iter_time["callbacks"] = (self.iter_time["callbacks"] * epoch + (toc - tic)) / (epoch + 1)

            # Medidas de tiempos
            self.time["iter_stop"] = time.time()
            self.time["iter_total"] = self.time["iter_stop"] - self.time["iter_start"]
            self.time["iter_mean"] = (self.time["iter_mean"] * epoch + self.time["iter_total"]) / (epoch + 1)
            self.time["elapsed"] = self.time["iter_stop"] - self.time["start"]
            self.time["remaining"] = (self.epochs - (epoch + 1)) * self.time["iter_mean"]

            self.iter_time["others"] = \
                (self.iter_time["others"] * epoch +
                 (self.time["iter_mean"] - np.sum([v for v in self.iter_time.values()][0:-1]))) / (epoch + 1)

            # Salida de texto
            if report_every and epoch % report_every == 0:
                str_a = f"\nEpoch {epoch}\n" \
                        f"    Fitness: {np.mean(self.fitness):.2e} +- {np.std(self.fitness):.2e}\n" \
                        f"    Elapsed time: {self.time['elapsed']:.1f} s\n" \
                        f"    Mean iter time: {self.time['iter_mean']:.2e} s\n"
                str_b = "\n".join([f"      ├ {k.replace('_', ' ').capitalize()}: {v:.2e} s "
                                   f"({v / self.time['iter_mean'] * 100:.2f} %)"
                                   for k, v in self.iter_time.items()])
                print(str_a + str_b, file=fileout)

            # Condición de frenado
            if self.stop_condition(epoch):
                print(f"Stop condition '{self._stop_method}' ocurred at epoch {epoch}.", file=fileout)
                break

        self.time["end"] = time.time()
        self.time["elapsed"] = self.time["end"] - self.time["start"]

        # Salida de texto
        str_a = f"\nFinished at epoch {epoch}\n" \
                f"    Fitness: {np.mean(self.fitness):.2e} +- {np.std(self.fitness):.2e}\n" \
                f"    Elapsed time: {self.time['elapsed']:.1f} s\n" \
                f"    Mean iter time: {self.time['iter_mean']:.2e} s\n"
        str_b = "\n".join([f"      ├ {k.replace('_', ' ').capitalize()}: {v:.2e} s "
                           f"({v / self.time['iter_mean'] * 100:.2f} %)"
                           for k, v in self.iter_time.items()])
        print(str_a + str_b, file=fileout)

        # Recorto la historia hasta la época final
        self.history["positions"] = self.history["positions"][0:epoch + 2]
        self.history["fitness"] = self.history["fitness"][0:epoch + 2]
        self.history["alpha"] = self.history["alpha"][0:epoch + 2]
        self.history["beta"] = self.history["beta"][0:epoch + 2]
        self.history["global_scale"] = self.history["global_scale"][0:epoch + 2]
        self.history["mean_displacement"] = self.history["mean_displacement"][0:epoch + 2]
        self.history["max_displacement"] = self.history["max_displacement"][0:epoch + 2]
        self.history["success_rate"] = self.history["success_rate"][0:epoch + 2]

        # Guardo la historia
        if save_path is not None:
            self.save_history(save_folder / (save_name + "_results.npz"))

        # Actualizo gráfico y finalizo el trabajo del hilo secundario
        if plot:
            self.plotter.add_to_queue(epoch, self.residue, self.positions, self.history)
            self.plotter.terminate()

        if fileout is not None:
            fileout.close()

    def get_config(self, dictionary: dict = None, indent: int = 0, output: Union[str, Path, object, None] = None) -> \
            str:
        """Returns a string with the current configuration of the algorithm.

        Parameters
        ----------
        dictionary : dict, optional
            Dictionary containing the information to return as a string, with attributes names (as keys) and its
            values. (Default: `None`, which means that it will return all the current attributes of the object.)
        indent : int, optional
            Integer indicating the indentation level of the output string. (Default: `0`.)
        output : Union[str, Path, object, None], optional
            Where to print the output string. It can be a string or `Path` object pointing to a valid file, any object
            with a `write` method or `None` for no printing. (Default: `None`.)

        Returns
        -------
        str
            Formatted string with the configuration of the algorithm.
        """
        if dictionary is None:
            dictionary = self.__dict__
            lines = [f"Configuration of {self.__class__.__name__} object:\n\n"]
        else:
            lines = []

        for k, v in dictionary.items():
            line = " " * indent + f"{k} = "
            if isinstance(v, np.ndarray):
                line += f"NumPy array of shape={v.shape}, dtype={v.dtype})"
            elif isinstance(v, list):
                line += f"list of len={len(v)}"
            elif isinstance(v, dict):
                line += f"dictionary with items\n"
                line += self.get_config(dictionary=v, indent=indent + 4)
            elif isinstance(v, Callable):
                line += f"function with name='{v.__name__}'"
            else:
                line += f"{v}"
            lines.append(line)

        string = "\n".join(lines)

        if output is None:
            pass
        elif isinstance(output, (str, Path)):
            output = Path(output).resolve()
            with open(output, "w+") as file:
                file.write(string)
        else:
            output.write(string)

        return string

    def save_history(self, path: Union[str, Path]):
        """Saves the history of the last run of the algorithm in a NPZ ile.

        Parameters
        ----------
        path: Union[str, Patj]
            String or `Path` objects that points to a valid NPZ file."""
        path = Path(path).resolve()

        output = {"sample": self._sample,
                  "psf": self._psf}
        output.update(self.history)
        np.savez(path, **output)


class ProcessPlotter:

    HISTORY_KEYS_TO_PLOT = ["fitness", "alpha", "beta", "mean_displacement", "max_displacement", "success_rate",
                            "global_scale"]

    PLOT_STYLE = DEFAULT_PLOT_STYLE

    def __init__(self):
        self.fig: plt.Figure = None
        self.axes: Dict[str, plt.Axes] = {}
        self.images: Dict[str, plt.Axes] = {}
        self.lines: Dict[str, plt.Axes] = {}
        self.save: bool = False
        self.save_path: Path = None
        self.save_halfway_plots: bool = False
        self.save_folder: Path = None
        self.save_name: str = None
        self.process = None
        self.queue: JoinableQueue = None
        self.ready = Event()
        self.missed = Event()
        self.stopped = Event()
        self.ended = Event()
        self.interactive_status: bool = None

    def start(self, sample: np.ndarray, residue: np.ndarray, initial_positions: np.ndarray, save_path: Path = None,
              save_halfway_plots: bool = False, save_folder: Path = None, save_name: str = None):
        self.interactive_status = plt.isinteractive()
        plt.ion()
        self.save = bool(save_path)
        self.save_path = Path(save_path).resolve() if self.save else None
        self.save_halfway_plots = save_halfway_plots
        self.save_folder = save_folder
        self.save_name = save_name

        self.queue = JoinableQueue(maxsize=0)
        self.process = Process(target=self.callback, daemon=True,
                               kwargs={"sample": sample, "residue": residue, "initial_positions": initial_positions})
        self.process.start()

    def add_to_queue(self, epoch, residue, positions, history):
        if self.queue.empty():
            self.queue.put((epoch, residue, positions, history), block=False)
            self.missed.clear()
        else:
            # print(f"ADVERTENCIA: el proceso de fondo encargado de actualizar la figura demora más que el "
            #       f"tiempo entre actualizaciones. Se aconseja aumentar el parámetro 'report_every' o reducir "
            #       f"el número de figuras. (Número de actualizaciones en espera: {self.plot_queue.qsize()}.)")
            self.missed.set()

    def callback(self, sample, residue, initial_positions):
        self.make_figure(sample, residue, initial_positions)

        if self.save:
            self.fig.savefig(self.save_folder / (self.save_name + f"_start.png"), dpi=200)

        self.ready.set()

        while not self.stopped.is_set():
            try:
                epoch, residue, positions, history = self.queue.get(block=False)
                self.update_figure(epoch=epoch, residue=residue, positions=positions, history=history)

                if self.save_halfway_plots and self.save:
                    self.fig.savefig(self.save_folder / (self.save_name + f"_{epoch}.png"), dpi=200)

                self.queue.task_done()
            except Empty:
                pass

        if self.save:
            self.fig.savefig(self.save_folder / (self.save_name + f"_end.png"), dpi=200)

        self.ended.set()

    def terminate(self):
        self.queue.join()
        self.stopped.set()
        self.ended.wait()

        self.queue.close()
        self.process.terminate()

        if not self.interactive_status:
            plt.ioff()

    def make_figure(self, sample: np.ndarray, residue: np.ndarray, initial_positions: np.ndarray):
        """Creates the monitor figure.

        Parameters
        ----------
        initial_positions : Union[numpy.ndarray, None]
            Initial solution used to generate the figure. If `None`, it uses the value stored in the attribute
            `initial_positions`. (Default: `None`.)"""

        self.fig: plt.Figure = plt.figure(constrained_layout=False)
        self.canvas = self.fig.canvas
        plt.get_current_fig_manager().window.showMaximized()

        # Initialize empty dicts for elements
        axes = {}
        images = {}
        lines = {}

        # Populate figure
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[0.05, 1, 4], hspace=.2, wspace=.2,
                                   left=.04, right=.96, bottom=.07, top=.98)
        gs: plt.GridSpec
        inner_gs = gs[1, 2].subgridspec(1, 3, hspace=0, wspace=.7)
        axes["sample"] = self.fig.add_subplot(gs[0, 1])
        axes["sample_colorbar"] = self.fig.add_subplot(gs[0, 0])
        axes["residue"] = self.fig.add_subplot(gs[1, 1])
        axes["residue_colorbar"] = self.fig.add_subplot(gs[1, 0])
        axes["fitness"] = self.fig.add_subplot(gs[0, 2:5])
        axes["alpha"] = self.fig.add_subplot(inner_gs[0, 0])
        axes["beta"] = axes["alpha"].twinx()
        axes["mean_displacement"] = self.fig.add_subplot(inner_gs[0, 1])
        axes["max_displacement"] = axes["mean_displacement"].twinx()
        axes["success_rate"] = self.fig.add_subplot(inner_gs[0, 2])
        axes["global_scale"] = axes["success_rate"].twinx()

        # Gráfico de la muestra y residuo
        for key, data in zip(["sample", "residue"], [sample, residue / residue.max()]):
            ax: plt.Axes = axes[key]
            style = self.PLOT_STYLE[key]
            im = ax.imshow(data)
            self.fig.colorbar(im, cax=axes[f"{key}_colorbar"], orientation='vertical')

            this_lines = ax.plot(initial_positions[:, 0], initial_positions[:, 1], **style["line_kwargs"])
            ax.set_title(style["title"], style["title_fontdict"])
            ax.set_xlabel(style["xlabel"], style["xlabel_fontdict"])
            ax.set_ylabel(style["ylabel"], style["ylabel_fontdict"])
            ax.tick_params(axis="x", which="both", labelcolor=style["xlabel_fontdict"]["color"],
                           labelsize=style["xlabel_fontdict"]["size"])
            ax.tick_params(axis="y", which="both", labelcolor=style["ylabel_fontdict"]["color"],
                           labelsize=style["ylabel_fontdict"]["size"])
            ax.set_xscale(style["xscale"])
            ax.set_yscale(style["yscale"])

            images[key] = im
            lines[f"{key}_positions"] = this_lines[0]

        # Gráficos de historia
        for key in self.HISTORY_KEYS_TO_PLOT:
            ax = axes[key]
            style = self.PLOT_STYLE[key]
            this_lines = ax.plot([1], [1], **style["line_kwargs"])
            ax.set_title(style["title"], fontdict=style["title_fontdict"])
            ax.set_xlabel(style["xlabel"], fontdict=style["xlabel_fontdict"])
            ax.set_ylabel(style["ylabel"], fontdict=style["ylabel_fontdict"])
            ax.set_xscale(style["xscale"])
            ax.set_yscale(style["yscale"])
            ax.tick_params(axis="x", labelcolor=style["xlabel_fontdict"]["color"],
                           labelsize=style["xlabel_fontdict"]["size"])
            ax.tick_params(axis="y", labelcolor=style["ylabel_fontdict"]["color"],
                           labelsize=style["ylabel_fontdict"]["size"])

            lines[key] = this_lines[0]

        # Draw figure
        self.canvas.draw()
        self.canvas.flush_events()
        plt.pause(0.001)

        self.axes = axes
        self.lines = lines
        self.images = images

    def update_figure(self, epoch: int, residue: np.array, positions: np.ndarray, history: dict):
        """Updates the monitor figure.

        Parameters
        ----------
        epoch : int
            Current epoch.
        residue : numpy.ndarray
            Current residue image (result of fitness function).
        positions : numpy.ndarray
            Current positions of the virtual sources.
        history : dict
            History dictionary with the values of all monitored variables during the current run.
        """

        # Actualizo residuo
        self.images["residue"].set_data(residue)
        self.images["residue"].set_clim(residue.min(), residue.max())

        # Actualizo posiciones
        self.lines["sample_positions"].set_data(positions[:, 0], positions[:, 1])
        self.lines["residue_positions"].set_data(positions[:, 0], positions[:, 1])

        # Actualizo historia
        for key in self.HISTORY_KEYS_TO_PLOT:
            ax = self.axes[key]
            line = self.lines[key]
            style = self.PLOT_STYLE[key]

            line.set_data(np.arange(-1, epoch), history[key][0:epoch + 1])
            ax.tick_params(axis="x", which="both", labelcolor=style["xlabel_fontdict"]["color"],
                           labelsize=style["xlabel_fontdict"]["size"])
            ax.tick_params(axis="y", which="both", labelcolor=style["ylabel_fontdict"]["color"],
                           labelsize=style["ylabel_fontdict"]["size"])
            ax.relim()
            ax.autoscale_view()

        self.canvas.draw()
        self.canvas.flush_events()



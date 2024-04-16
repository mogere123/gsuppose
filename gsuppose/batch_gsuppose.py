class BatchGSUPPOSe:
    """
    TODO: NOT FUNCTIONAL.
    """
    STOP_METHODS = ["success_rate", "fitness", "mean_displacement", "custom"]

    HISTORY_KEYS_TO_PLOT = ["fitness", "alpha", "beta", "mean_displacement", "max_displacement", "success_rate",
                            "global_scale"]

    def __init__(self,
                 sample: np.ndarray,
                 psf: np.ndarray,
                 initial_positions: np.ndarray = None,
                 sample_pixel_size: Tuple[float, float] = (1, 1),
                 psf_pixel_size: Tuple[float, float] = (1, 1),
                 fitness_function: Callable = mse,
                 dfitness_function: Callable = dmse,
                 normalize_input: str = 'none',
                 allow_beta: bool = False,
                 batch_size: int = 1,
                 global_scale: float = 1E-1,
                 normalize_gradient: bool = False,
                 device_index: Union[int, Sequence[int]] = None):
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
        self.detailed_time = {key: None for key in ["prepare_batches", "convolve_batches", "update_positions",
                                                    "convolve_solution", "fit_parameters", "compute_fitness",
                                                    "update_plot"]}
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
        self.fig: plt.Figure = None
        self.axes: Dict[str, plt.Axes] = {}
        self.images: Dict[str, plt.Axes] = {}
        self.lines: Dict[str, plt.Axes] = {}

        # Configuro con los argumentos
        self.sample = sample
        self.sample_pixel_size = sample_pixel_size
        self.normalize_input = normalize_input
        self.allow_beta = allow_beta
        self.initial_positions = initial_positions
        self.psf = psf
        self.psf_pixel_size = psf_pixel_size
        self.fitness_function = fitness_function
        self.dfitness_function = dfitness_function

        self.batch_size = min(batch_size, self.nsources)
        self.global_scale = global_scale
        self.normalize_gradient = normalize_gradient
        self.device_index = device_index

    @property
    def sample(self):
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
        return self._psf

    @psf.setter
    def psf(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"La propiedad 'psf' debe ser tipo numpy.ndarray. Se recibió {value}.")

        self.raw_psf = np.copy(value)
        self._psf = np.copy(self.raw_psf)

        self._luts[0] = self._psf
        for dim in range(self.ndim):
            self._luts[dim + 1] = np.diff(self._psf, axis=self.ndim - 1 - dim)

    @property
    def nsources(self):
        return self.initial_positions.shape[0]

    @property
    def ndim(self):
        return self.initial_positions.shape[1]

    @property
    def nbatches(self):
        return self.nsources // self.batch_size

    @property
    def subtract_mean(self):
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

    @property
    def stop_method(self):
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
        if self.stop_method is not None:
            return self.history[self.stop_method][current_iter] < self._stop_limit
        else:
            return False

    def batch_gradient(self, source_indices: Sequence[int]):
        dfitness = self.dfitness_function(self.alpha * self.convolution + self.beta, self.sample)
        grads = np.sum(- self.alpha * dfitness * self.batch_convolution[:, source_indices], axis=(2, 3)).T

        if self.normalize_gradient:
            norm = np.linalg.norm(grads, ord=2, axis=grads.ndim - 1, keepdims=True)
            grads = np.divide(grads, norm, where=norm != 0.0)

        return grads

    def batch_update(self, source_indices, grads, epoch) -> np.ndarray:
        pass

    def convolve(self, positions: np.ndarray) -> np.ndarray:
        convolution = self.convmngr[0].sync_convolve(positions[np.newaxis, :, :])[0]

        if self.subtract_mean:
            convolution -= convolution.mean()

        return convolution

    def fit_parameters(self, convolution: np.ndarray) -> Tuple[float, float]:
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
            save_path: Union[str, Path, None] = None, plot: bool = False, save_halfway_plots: bool = False):

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
            self.print_config(output=save_folder / (save_name + "_config.txt"))

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
        self.fitness = np.sum(self.residue)
        self.time = {key: 0.0 for key in ["start", "end", "elapsed", "remaining", "iter_start", "iter_stop",
                                          "iter_total", "iter_mean"]}
        self.detailed_time = {key: 0.0 for key in ["prepare_batches", "convolve_batches", "update_positions",
                                                   "convolve_solution", "fit_parameters", "compute_fitness",
                                                   "update_plot"]}

        source_order = np.arange(self.nsources)
        batch_order = np.zeros((self.nbatches, self.batch_size), dtype=int)

        # Inicializo historia de la corrida
        self.history["epoch"] = np.zeros(self.epochs + 1, dtype=int)
        self.history["positions"] = np.zeros((self.epochs + 1, self.nsources, self.ndim))
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
        self.initialize_optimizer()

        # Inicializa figuras
        if plot:
            # Activa el modo interactivo de Matplotlib
            interactive_status = plt.isinteractive()
            if not interactive_status:
                plt.ion()
            self.make_figure()

            # Guarda figura
            if save_path is not None:
                self.fig.savefig(save_folder / (save_name + "_start.png"), dpi=200)

        self.time["start"] = time.time()

        # Iteración principal (épocas)
        epoch = 0
        for epoch in range(self.epochs):
            self.time["iter_start"] = time.time()

            # Actualizo el orden del lote
            tic = time.time()
            np.random.shuffle(source_order)
            batch_order = np.reshape(source_order[0:self.nbatches * self.batch_size], (self.nbatches, self.batch_size))
            toc = time.time()
            self.detailed_time["prepare_batches"] = (self.detailed_time["prepare_batches"] * epoch + (toc - tic)) / \
                                                    (epoch + 1)

            # Convolución del gradiente
            tic = time.time()
            for dim in range(self.ndim):
                self.convmngr[dim + 1].async_convolve(self.positions[:, np.newaxis, :])
            for dim in range(self.ndim):
                self.batch_convolution[dim] = self.convmngr[dim + 1].sync_get_results()
            toc = time.time()
            self.detailed_time["convolve_batches"] = (self.detailed_time["convolve_batches"] * epoch + (toc - tic)) / \
                                                     (epoch + 1)

            # Actualización por lotes
            time_pos_update = 0.0
            time_convolution = 0.0
            time_parameters = 0.0
            time_fitness = 0.0

            for j in range(self.nbatches):
                # Actualizo las posiciones
                tic = time.time()
                grads = self.batch_gradient(source_indices=batch_order[j])
                dr = self.batch_update(source_indices=batch_order[j], grads=grads, epoch=epoch)
                self.positions[batch_order[j]] += dr
                toc = time.time()
                self.history["mean_displacement"][epoch] += np.sqrt(np.sum(dr ** 2)) / self.nsources
                self.history["max_displacement"][epoch] += np.max(np.abs(dr))
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
                self.fitness = np.sum(self.residue)
                self.history["success_rate"][epoch] += int(self.fitness <= old_fitness) / self.nbatches
                toc = time.time()
                time_fitness += toc - tic

            # Actualizo tiempos detallados
            self.detailed_time["update_positions"] = (self.detailed_time["update_positions"] * epoch + time_pos_update) \
                                                     / (epoch + 1)
            self.detailed_time["convolve_solution"] = (self.detailed_time["convolve_solution"] * epoch +
                                                       time_convolution) / (epoch + 1)
            self.detailed_time["fit_parameters"] = (self.detailed_time["fit_parameters"] * epoch + time_fitness) / \
                                                   (epoch + 1)
            self.detailed_time["compute_fitness"] = (self.detailed_time["compute_fitness"] * epoch + time_fitness) / \
                                                    (epoch + 1)

            # Actualizo historia
            self.history["epoch"][epoch + 1] = epoch
            self.history["positions"][epoch + 1] = self.positions
            self.history["fitness"][epoch + 1] = self.fitness
            self.history["alpha"][epoch + 1] = self.alpha
            self.history["beta"][epoch + 1] = self.beta
            self.history["global_scale"][epoch + 1] = self.global_scale

            # Medidas de tiempos
            self.time["iter_stop"] = time.time()
            self.time["iter_total"] = self.time["iter_stop"] - self.time["iter_start"]
            self.time["iter_mean"] = (self.time["iter_mean"] * epoch + self.time["iter_total"]) / (epoch + 1)
            self.time["elapsed"] = self.time["iter_stop"] - self.time["start"]
            self.time["remaining"] = (self.epochs - (epoch + 1)) * self.time["iter_mean"]

            # Actualizo gráfico
            if plot and epoch % report_every == 0:
                tic = time.time()
                self.update_figure(epoch=epoch)

                if save_path is not None and save_halfway_plots:
                    self.fig.savefig(save_folder / (save_name + f"_{epoch}.png"), dpi=200)
                toc = time.time()
                self.detailed_time["update_plot"] = (self.detailed_time["update_plot"] * epoch + (toc - tic)) / \
                                                    (epoch + 1) / report_every

            # Salida de texto
            if epoch % report_every == 0:
                str_a = f"\nEpoch {epoch}\n" \
                        f"    Fitness: {self.fitness:.2e}\n" \
                        f"    Elapsed time: {self.time['elapsed']:.1f} s\n" \
                        f"    Mean iter time: {self.time['iter_mean']:.2e} s\n"
                str_b = "\n".join([f"      ├ {k.replace('_', ' ').capitalize()}: {v:.2e} s " +
                                   f"({v / self.time['iter_mean'] * 100:.2f} %)" for k, v in
                                   self.detailed_time.items()])
                print(str_a + str_b)

            # Condición de frenado
            if self.stop_condition(epoch):
                print(f"Stop condition '{self._stop_method}' ocurred at epoch {epoch}.")
                break

        self.time["end"] = time.time()
        self.time["elapsed"] = self.time["end"] - self.time["start"]

        # Salida de texto
        str_a = f"\nFinished at epoch {epoch}\n" \
                f"    Fitness: {self.fitness:.2e}\n" \
                f"    Elapsed time: {self.time['elapsed']:.1f} s\n" \
                f"    Mean iter time: {self.time['iter_mean']:.2e} s\n"
        str_b = "\n".join([f"      ├ {k.replace('_', ' ').capitalize()}: {v:.2e} s "
                           f"({v / self.time['iter_mean'] * 100:.2f} %)" for k, v in self.detailed_time.items()])
        print(str_a + str_b)

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

        # Actualizo gráfico
        if plot:
            self.update_figure(epoch=epoch)
            if save_path is not None:
                self.fig.savefig(save_folder / (save_name + "_end.png"), dpi=200)

            # Vuelve el modo interactivo de Matplotlib a su estado original
            if not interactive_status:
                plt.ioff()

    def print_config(self, dictionary=None, indent=0, output=None):
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
                line += self.print_config(dictionary=v, indent=indent + 4)
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
        path = Path(path).resolve()

        output = {"sample": self._sample}
        output.update(self.history)
        np.savez(path, **output)

    def make_figure(self):
        self.fig = fig = plt.figure(constrained_layout=False)
        self.axes = {}
        self.images = {}
        self.lines = {}

        gs = self.fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[0.05, 1, 4], hspace=.2, wspace=.2,
                                   left=.04, right=.96, bottom=.07, top=.98)
        gs: plt.GridSpec
        inner_gs = gs[1, 2].subgridspec(1, 3, hspace=0, wspace=.7)
        self.axes["sample"] = self.fig.add_subplot(gs[0, 1])
        self.axes["sample_colorbar"] = self.fig.add_subplot(gs[0, 0])
        self.axes["residue"] = self.fig.add_subplot(gs[1, 1])
        self.axes["residue_colorbar"] = self.fig.add_subplot(gs[1, 0])
        self.axes["fitness"] = self.fig.add_subplot(gs[0, 2:5])
        self.axes["alpha"] = self.fig.add_subplot(inner_gs[0, 0])
        self.axes["beta"] = self.axes["alpha"].twinx()
        self.axes["mean_displacement"] = self.fig.add_subplot(inner_gs[0, 1])
        self.axes["max_displacement"] = self.axes["mean_displacement"].twinx()
        self.axes["success_rate"] = self.fig.add_subplot(inner_gs[0, 2])
        self.axes["global_scale"] = self.axes["success_rate"].twinx()

        # Gráfico de la muestra y residuo
        for key, data in zip(["sample", "residue"], [self._sample, self.residue / self.residue.max()]):
            ax: plt.Axes = self.axes[key]
            style = self.PLOT_STYLES[key]
            im = ax.imshow(data)
            fig.colorbar(im, cax=self.axes[f"{key}_colorbar"], orientation='vertical')

            lines = ax.plot(self.positions[:, 0], self.positions[:, 1], **style["line_kwargs"])
            ax.set_title(style["title"], style["title_fontdict"])
            ax.set_xlabel(style["xlabel"], style["xlabel_fontdict"])
            ax.set_ylabel(style["ylabel"], style["ylabel_fontdict"])
            ax.tick_params(axis="x", which="both", labelcolor=style["xlabel_fontdict"]["color"],
                           labelsize=style["xlabel_fontdict"]["size"])
            ax.tick_params(axis="y", which="both", labelcolor=style["ylabel_fontdict"]["color"],
                           labelsize=style["ylabel_fontdict"]["size"])
            ax.set_xscale(style["xscale"])
            ax.set_yscale(style["yscale"])

            self.images[key] = im
            self.lines[f"{key}_positions"] = lines[0]

        # Gráficos de historia
        for key in self.HISTORY_KEYS_TO_PLOT:
            ax = self.axes[key]
            style = self.PLOT_STYLES[key]
            lines = ax.plot([], [], **style["line_kwargs"])
            ax.set_title(style["title"], fontdict=style["title_fontdict"])
            ax.set_xlabel(style["xlabel"], fontdict=style["xlabel_fontdict"])
            ax.set_ylabel(style["ylabel"], fontdict=style["ylabel_fontdict"])
            ax.set_xscale(style["xscale"])
            ax.set_yscale(style["yscale"])
            ax.tick_params(axis="x", labelcolor=style["xlabel_fontdict"]["color"],
                           labelsize=style["xlabel_fontdict"]["size"])
            ax.tick_params(axis="y", labelcolor=style["ylabel_fontdict"]["color"],
                           labelsize=style["ylabel_fontdict"]["size"])

            self.lines[key] = lines[0]

        try:
            plt.get_current_fig_manager().window.showMaximized()
        except:
            pass

        plt.ion()
        plt.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def update_figure(self, epoch: int):
        # Actualizo residuo
        self.images["residue"].set_data(self.residue)
        self.images["residue"].set_clim(self.residue.min(), self.residue.max())

        # Actualizo posiciones
        self.lines["sample_positions"].set_data(self.positions[:, 0], self.positions[:, 1])
        self.lines["residue_positions"].set_data(self.positions[:, 0], self.positions[:, 1])

        # Actualizo historia
        for key in self.HISTORY_KEYS_TO_PLOT:
            ax = self.axes[key]
            line = self.lines[key]
            style = self.PLOT_STYLES[key]

            line.set_data(np.arange(-1, epoch), self.history[key][0:epoch + 1])
            ax.tick_params(axis="x", which="both", labelcolor=style["xlabel_fontdict"]["color"],
                           labelsize=style["xlabel_fontdict"]["size"])
            ax.tick_params(axis="y", which="both", labelcolor=style["ylabel_fontdict"]["color"],
                           labelsize=style["ylabel_fontdict"]["size"])
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


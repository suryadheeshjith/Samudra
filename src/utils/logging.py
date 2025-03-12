import datetime
import logging
import resource
import sys
import time
import traceback
import warnings
from collections import defaultdict, deque

import torch


def handle_logging(cfg):
    # Set up logging
    logger = logging.getLogger()  # Use the root logger or specify a name if needed
    logger.setLevel(logging.DEBUG if cfg.debug else logging.INFO)

    # STDOUT handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG if cfg.debug else logging.INFO)
    stdout_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(stdout_handler)

    # Add experiment log file handler
    experiment_log_path = cfg.experiment.output_dir / "experiment.log"
    experiment_handler = logging.FileHandler(experiment_log_path)
    experiment_handler = logging.FileHandler(experiment_log_path)
    experiment_handler.setLevel(logging.INFO)  # Capture info and above
    experiment_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(experiment_handler)

    # Add separate error log file handler
    error_log_path = cfg.experiment.output_dir / "error.log"
    error_handler = logging.FileHandler(error_log_path)
    error_handler.setLevel(logging.WARNING)  # Capture warnings and errors
    error_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(error_handler)


def handle_warnings():
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logging.info("\n=== Warning Details ===")
        logging.info(f"Message: {message}")
        logging.info(f"Category: {category}")
        logging.info(f"File: {filename}")
        logging.info(f"Line: {lineno}")
        logging.info("\nFull stack trace:")
        stack = traceback.extract_stack()[:-1]  # Remove current frame
        for frame in stack:
            logging.info(
                f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}'
            )
            if frame.line:
                logging.info(f"    {frame.line}")
        logging.info("=====================\n")

    warnings.showwarning = warning_handler


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque: deque[float] = deque(maxlen=window_size)
        self.total: float = 0.0
        self.count: int = 0
        self.fmt: str = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n=n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{value:.3f}({avg:.3f})", window_size=print_freq)
        data_time = SmoothedValue(fmt="{value:.3f}({avg:.3f})", window_size=print_freq)
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg_list: list[str] = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        log_msg_list.append("max cpu mem: {cpu_memory:.0f}")
        if torch.cuda.is_available():
            log_msg_list.append("max gpu mem: {gpu_memory:.0f}")
        log_msg = self.delimiter.join(log_msg_list)
        KB = 1024.0
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logging.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            cpu_memory=resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss
                            / KB,
                            gpu_memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logging.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            cpu_memory=resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss
                            / KB,
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )

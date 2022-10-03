import logging 

FILE_LVL    = 'WARNING'
CONSOLE_LVL = 'INFO'

def _logger_add(logger: logging.Logger, handler: logging.StreamHandler, level: int, formatter: logging.Formatter):
    if level:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

class Formatter(logging.Formatter):
    FORMATS = { #'off.off': '\x1b[31;20m',
                'off.windfarm': '\x1b[32;20m',
                'off.states': '\x1b[33;20m',
                'off.wake_model': '\x1b[34;20m',
                'off.wake_solver': '\x1b[35;20m',
                'off.observation_points': '\x1b[37;20m',
                'off.turbine': '\x1b[36;20m' }

    DEFAULT = '\x1b[0;20m'
    RESET = '\x1b[0m'

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt

    def format(self, record):
        log_fmt = self.FORMATS.get(record.name,self.DEFAULT) + self.fmt + self.RESET
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

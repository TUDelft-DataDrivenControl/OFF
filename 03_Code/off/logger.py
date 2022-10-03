import logging 

FILE_LVL    = 'WARNING'
CONSOLE_LVL = 'INFO'

def _logger_add(logger: logging.Logger, handler: logging.StreamHandler, level: int, formatter: logging.Formatter):
    if level:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)




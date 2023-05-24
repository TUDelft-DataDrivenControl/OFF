# Copyright (C) <2023>, M Becker (TUDelft), M Lejeune (UCLouvain)

# List of the contributors to the development of OFF: see LICENSE file.
# Description and complete License: see LICENSE file.
	
# This program (OFF) is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program (see COPYING file).  If not, see <https://www.gnu.org/licenses/>.

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

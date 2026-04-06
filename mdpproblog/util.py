# This file is part of MDP-ProbLog.

# MDP-ProbLog is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# MDP-ProbLog is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with MDP-ProbLog.  If not, see <http://www.gnu.org/licenses/>.


import logging
import sys
import time

# --- Logging ---

TRACE = 5
logging.addLevelName(TRACE, "TRACE")


class MDPProbLogFormatter(logging.Formatter):
    """Custom log formatter: prefixes every line of multi-line messages."""

    def format(self, message):
        msg = str(message.msg) % message.args if message.args else str(message.msg)
        lines = msg.split("\n")
        linestart = "[%s] " % message.levelname
        return linestart + ("\n" + linestart).join(lines)


def init_logger(verbose=None, name="mdpproblog", out=None):
    """Initialize the MDP-ProbLog logger.

    Verbosity levels:

    ========= ========== ================================================
    ``-v``    level      What is shown
    ========= ========== ================================================
    (none)    WARNING    Only errors and warnings
    ``-v``    INFO       Per-phase execution times + FluentSchema summary
    ``-vv``   DEBUG      INFO + pipeline lifecycle details per phase
    ``-vvv``  TRACE (5)  DEBUG + per-iteration residual/elapsed + convergence
    ========= ========== ================================================

    :param verbose: verbosity level (``None``, 1, 2, or 3)
    :type verbose: int or None
    :param name: logger name (default: ``"mdpproblog"``)
    :type name: str
    :param out: output stream (default: ``sys.stdout``)
    :type out: file
    :return: configured logger
    :rtype: logging.Logger
    """
    if out is None:
        out = sys.stdout

    logger = logging.getLogger(name)
    ch = logging.StreamHandler(out)
    ch.setFormatter(MDPProbLogFormatter())
    logger.handlers = []
    logger.addHandler(ch)

    if not verbose:
        logger.setLevel(logging.WARNING)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
        logger.info("Output level: INFO")
    elif verbose == 2:
        logger.setLevel(logging.DEBUG)
        logger.debug("Output level: DEBUG")
    else:
        logger.setLevel(TRACE)
        logger.log(TRACE, "Output level: TRACE")

    return logger


# --- Timing ---

class Timer(object):
    """Context manager that logs elapsed time for a code block at INFO level.

    :param msg: label for the timed block
    :type msg: str
    :param logger: logger name to write to (default: ``"mdpproblog"``)
    :type logger: str
    """

    def __init__(self, msg, logger="mdpproblog"):
        self.message = msg
        self._logger = logger
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        logging.getLogger(self._logger).info("%s: %.4fs", self.message, elapsed)


# --- Output formatting ---

def format_state(state_tuple):
    """Format a state tuple as a human-readable string.

    Filters out inactive fluents (value == 0) and joins active ones with commas.

    :param state_tuple: sequence of ``(fluent, value)`` pairs
    :type state_tuple: tuple
    :return: comma-separated active fluents, or ``"none"`` if all are inactive
    :rtype: str
    """
    if not state_tuple:
        return "none"
    active = [str(fluent) for fluent, val in state_tuple if val == 1]
    return ", ".join(active) if active else "none"

# tramdag/utils/__init__.py

from .logger import Logger

# Shared global logger, configure in your notebook
logger = Logger(verbose=False, debug=False)
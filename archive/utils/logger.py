# utils/logger.py

class Logger:
    def __init__(self, verbose=False, debug=False):
        self.verbose = verbose
        self.debug = debug

    def set_level(self, level):
        if level.lower() == "debug":
            self.verbose = True
            self.debug = True
        elif level.lower() == "info":
            self.verbose = True
            self.debug = False
        elif level.lower() in ["warn", "warning"]:
            self.verbose = False
            self.debug = False
        else:  # silent except errors
            self.verbose = False
            self.debug = False

    def info(self, msg: str):
        if self.verbose:
            print(f"[INFO] {msg}")

    def debug_msg(self, msg: str):
        if self.debug:
            print(f"[DEBUG] {msg}")

    def debug(self, msg: str):
        if self.debug:
            print(f"[DEBUG] {msg}")

    def warning(self, msg: str):
        print(f"[WARNING] {msg}")

    def error(self, msg: str):
        print(f"[ERROR] {msg}")


# -----------------------------------------------------
# Create one global instance with default settings
# -----------------------------------------------------
logger = Logger(verbose=True, debug=False)  # default: INFO level

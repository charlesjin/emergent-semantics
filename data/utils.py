import contextlib
import sys


class _DF:
    def write(*_, **__):
        pass

    def flush(*_, **__):
        pass


_DF_obj = _DF()


@contextlib.contextmanager
def stdout_silencing():
    save_stdout = sys.stdout
    sys.stdout = _DF_obj
    try:
        yield
    finally:
        sys.stdout = save_stdout

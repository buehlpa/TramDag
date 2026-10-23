"""Visual configuration editor (notebook widget and standalone browser app)."""


def serve(config_path=None, data=None, port=8765, open_browser=True):
    from .server import serve as _serve

    return _serve(config_path, data=data, port=port, open_browser=open_browser)

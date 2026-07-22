import logging

log = logging.getLogger("deeprank_gnn")
log.setLevel(logging.DEBUG)

_ch = logging.StreamHandler()
_formatter = logging.Formatter(
    " %(asctime)s %(module)s:%(lineno)d %(levelname)s - %(message)s"
)
_ch.setFormatter(_formatter)
log.addHandler(_ch)

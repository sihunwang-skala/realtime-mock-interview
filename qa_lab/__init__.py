
import os as _os, pathlib as _p
_env = _p.Path(__file__).resolve().parents[1] / ".env"          # git-ignored; KEY=VALUE lines, never printed
if _env.exists():
    for _l in _env.read_text().splitlines():
        if "=" in _l and not _l.lstrip().startswith("#"):
            _k, _v = _l.split("=", 1); _os.environ[_k.strip()] = _v.strip().strip('"')

import json
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Union


def read_json(fname: Union[str, Path]) -> OrderedDict:
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: Any, fname: Union[str, Path]) -> None:
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def write_pickle(content: Any, fname: Union[str, Path]) -> None:
    fname = Path(fname)
    with open(fname, "wb") as f:
        pickle.dump(content, f)


def read_pickle(fname: Union[str, Path]) -> Any:
    fname = Path(fname)
    with open(fname, "rb") as f:
        return pickle.load(f)


def write_txt(content: str, fname: Union[str, Path]) -> None:
    fname = Path(fname)
    with fname.open("w") as handle:
        handle.write(content)


def read_txt(fname: Union[str, Path]) -> str:
    fname = Path(fname)
    with fname.open("r") as handle:
        return handle.read()

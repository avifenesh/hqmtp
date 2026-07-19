#!/usr/bin/env python3
"""Chat-template rendering for corpus generation against the raw /completions-style
endpoint (reconstructed 2026-07-19 after the box loss; original was box-only).

build_template(path) loads a HuggingFace chat_template.jinja into a jinja2
environment carrying the helpers HF templates expect (tojson, raise_exception,
strftime_now), so gen_corpus2.py can render `messages` client-side exactly the way
the serving stack would."""
import datetime
import json

import jinja2


def _raise_exception(msg):
    raise jinja2.exceptions.TemplateError(msg)


def build_template(path):
    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    return env.from_string(open(path).read())

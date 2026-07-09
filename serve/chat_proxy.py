#!/usr/bin/env python3
"""OpenAI chat/completions -> bw24-server raw /v1/completions proxy (hqmtp axis-D).

pi (and any OpenAI chat client) speaks /v1/chat/completions with tools; bw24-server
serves raw-prompt /v1/completions only. This proxy renders the model's OWN
chat_template.jinja (exact HF template, tools included), forwards the rendered prompt
greedy (temperature 0 -> server takes the MTP spec path, [spec-acc] telemetry fires),
parses <think> and <tool_call> blocks out of the generated text, and answers in
OpenAI chat format (SSE single-chunk when stream=true).

Usage:
  python3 chat_proxy.py --template <chat_template.jinja> --upstream http://127.0.0.1:8014 \
      --port 8017 [--log requests.jsonl]
"""
import argparse
import json
import re
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jinja2


def build_template(path):
    env = jinja2.Environment(loader=jinja2.BaseLoader(), trim_blocks=False, lstrip_blocks=False)
    env.filters.setdefault("tojson", lambda v, **kw: json.dumps(v, **kw))
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    return env.from_string(open(path).read())


def normalize_messages(messages):
    out = []
    for m in messages:
        m = dict(m)
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        elif c is None:
            m["content"] = ""
        # HF qwen templates expect tool_calls[].function.arguments as an OBJECT
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except (json.JSONDecodeError, TypeError):
                    pass
        out.append(m)
    return out


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_generation(text):
    reasoning = "\n".join(m.strip() for m in THINK_RE.findall(text)) or None
    text = THINK_RE.sub("", text)
    # unclosed think block (generation cut mid-thought)
    if "<think>" in text:
        head, _, tail = text.partition("<think>")
        reasoning = ((reasoning + "\n") if reasoning else "") + tail.strip()
        text = head
    tool_calls = []
    for i, raw in enumerate(TOOL_CALL_RE.findall(text)):
        try:
            obj = json.loads(raw)
            tool_calls.append({
                "id": f"call_{int(time.time()*1000)}_{i}",
                "type": "function",
                "function": {"name": obj.get("name", ""),
                             "arguments": json.dumps(obj.get("arguments", {}))},
            })
        except json.JSONDecodeError:
            pass
    content = TOOL_CALL_RE.sub("", text).strip() or None
    return content, reasoning, tool_calls


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # undici (node fetch) rejects HTTP/1.0 responses
    template = None
    upstream = None
    log_path = None
    enable_thinking = False

    def log_message(self, *a):  # quiet default access log
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{"id": "hqmtp-9b", "object": "model"}]})
        else:
            self._json(200, {"status": "ok"})

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            self._json(404, {"error": f"unknown path {self.path}"})
            return
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = normalize_messages(req.get("messages", []))
        prompt = self.template.render(
            messages=messages, tools=req.get("tools") or None,
            add_generation_prompt=True, enable_thinking=self.enable_thinking)
        up_body = {
            "model": req.get("model", "hqmtp-9b"),
            "prompt": prompt,
            "max_tokens": min(int(req.get("max_tokens") or 4096), 4096),
            "temperature": 0.0,
            "stop": ["<|im_end|>"],
        }
        t0 = time.time()
        r = urllib.request.urlopen(urllib.request.Request(
            f"{self.upstream}/v1/completions", data=json.dumps(up_body).encode(),
            headers={"Content-Type": "application/json"}), timeout=1200)
        up = json.loads(r.read())
        gen = up.get("text", "").split("<|im_end|>")[0]
        # the qwen generation prompt ends with an OPEN <think> block — the generated text
        # starts inside it; restore the opener so the parser sees a well-formed block.
        if prompt.rstrip().endswith("<think>"):
            gen = "<think>" + gen
        content, reasoning, tool_calls = parse_generation(gen)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({
                    "ts": time.time(), "prompt_chars": len(prompt),
                    "n_tokens": up.get("n_tokens"), "elapsed_s": time.time() - t0,
                    "stop_reason": up.get("stop_reason"), "n_tool_calls": len(tool_calls),
                    "prompt": prompt, "text": up.get("text", "")}) + "\n")
        msg = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["tool_calls"] = tool_calls
        finish = "tool_calls" if tool_calls else (
            "length" if up.get("stop_reason") == "MaxNew" else "stop")
        usage = {"prompt_tokens": len(prompt) // 4,
                 "completion_tokens": up.get("n_tokens", 0),
                 "total_tokens": len(prompt) // 4 + up.get("n_tokens", 0)}
        rid = f"chatcmpl-{int(time.time()*1000)}"
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            delta = {k: v for k, v in msg.items() if k != "role"}
            delta["role"] = "assistant"
            for chunk in (
                {"id": rid, "object": "chat.completion.chunk", "model": up_body["model"],
                 "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {"id": rid, "object": "chat.completion.chunk", "model": up_body["model"],
                 "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                 "usage": usage},
            ):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self._json(200, {
                "id": rid, "object": "chat.completion", "model": up_body["model"],
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": usage})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--upstream", default="http://127.0.0.1:8014")
    ap.add_argument("--port", type=int, default=8017)
    ap.add_argument("--log")
    ap.add_argument("--enable-thinking", action="store_true")
    args = ap.parse_args()
    Handler.template = build_template(args.template)
    Handler.upstream = args.upstream
    Handler.log_path = args.log
    Handler.enable_thinking = args.enable_thinking
    print(f"chat proxy :{args.port} -> {args.upstream} (template {args.template})")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

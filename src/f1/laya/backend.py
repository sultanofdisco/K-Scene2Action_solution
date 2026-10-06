"""Pinned CPU SDK adapter with preflight against the SDK's actual token budgets."""
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

REPO = "convaiinnovations/laya-multilingual"
SDK_VERSION = "0.3.27"
MAX_LEN = 1024
HEAD_MAX_LEN = 256


class InputTooLong(ValueError):
    pass


class InvalidOutput(ValueError):
    pass


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_model(output: Path, revision: str = "main"):
    from huggingface_hub import HfApi, snapshot_download
    from laya.agent import _fix_tokenizer_config

    if output.exists():
        raise FileExistsError("Use a new model directory; existing model is preserved")
    commit = HfApi().model_info(REPO, revision=revision).sha
    output.mkdir(parents=True)
    snapshot_download(REPO, revision=commit, local_dir=output,
                      allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*.json"])
    # The SDK performs this same compatibility conversion on load. Do it before
    # recording local digests so later SDK loads do not invalidate the lock.
    original_tokenizer_config = digest(output / "tokenizer/tokenizer_config.json")
    _fix_tokenizer_config(str(output))
    files = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*"))
             if p.is_file() and ".cache" not in p.relative_to(output).parts}
    lock = {"repo": REPO, "revision": commit, "sdk_version": SDK_VERSION, "files": files,
            "tokenizer_config_upstream_sha256": original_tokenizer_config,
            "tokenizer_compatibility_conversion": "laya.agent._fix_tokenizer_config"}
    (output / "model-lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


def check_budget(tok, state, questions):
    from laya.common import build_sequence, encode_text, render_options, serialize_state

    state_ids = encode_text(tok, serialize_state(state).replace(tok.mask_token, " "),
                            add_special_tokens=False)["input_ids"]
    report = {}
    for name, definition in questions.items():
        q = {"t": "choice", "ins": definition["instructions"], "crit": definition["criteria"]}
        # build_head silently truncates both instructions and option text. Verify
        # their full sizes separately, before trusting its sequence length.
        instruction_size = len(encode_text(tok, "choice question: " + q["ins"], add_special_tokens=False)["input_ids"])
        option_sizes = [len(encode_text(tok, " " + s, add_special_tokens=False)["input_ids"])
                        for s in render_options(q)]
        if max(option_sizes) > 48 or instruction_size + sum(n + 1 for n in option_sizes) > HEAD_MAX_LEN:
            raise InputTooLong("Question or options exceed SDK head budget")
        ids, markers, options, stats = build_sequence(
            tok, state, q, MAX_LEN, HEAD_MAX_LEN, state_ids=state_ids,
            return_stats=True, return_truncation_stats=True)
        report[name] = {**stats, "sequence_tokens": len(ids), "head_tokens": len(ids)-stats["state_tokens_used"]-1}
        if stats["truncated"] or len(markers) != 3 or options["options_distinct"] != 3:
            raise InputTooLong("Full OCR/context does not fit; no truncation permitted")
    return report


class LayaBackend:
    """Load lazily once; completed empty OCR must not load the model."""

    def __init__(self, model_dir="artifacts/laya/model", cpu_threads=4):
        if type(cpu_threads) is not int or cpu_threads <= 0:
            raise ValueError("cpu_threads must be a positive integer")
        self.model_dir = Path(model_dir)
        self.cpu_threads = cpu_threads
        self.agent = None
        self.version = "laya-not-loaded"
        self.metadata = {}

    def _load(self):
        if self.agent is not None:
            return
        import torch
        from laya import Agent

        if version("laya") != SDK_VERSION:
            raise RuntimeError("SDK version differs from the tested adapter")
        lock = json.loads((self.model_dir / "model-lock.json").read_text(encoding="utf-8"))
        if lock["repo"] != REPO or lock["sdk_version"] != SDK_VERSION:
            raise ValueError("Unexpected model lock")
        if not {"model.safetensors", "rl_agent_config.json", "encoder/config.json",
                "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"} <= set(lock["files"]):
            raise ValueError("Incomplete model lock")
        torch.set_num_threads(self.cpu_threads)
        # Local config and tokenizer prevent inference-time network downloads.
        self.agent = Agent(str(self.model_dir.resolve()), device="cpu", fast=False, compile=False,
                           expected_sha256=lock["files"])
        self.version = REPO + "@" + lock["revision"]
        self.metadata = {"model_lock": lock, "packages": {p: version(p) for p in ("laya", "torch", "transformers")},
                         "device": "cpu", "cpu_threads": self.cpu_threads,
                         "max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN}

    def predict(self, state, questions):
        self._load()
        self.metadata.pop("token_budget", None)
        self.metadata["token_budget"] = check_budget(self.agent.tok, state, questions)
        result = self.agent.predict(state, questions, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
        usage = result.get("usage", {})
        if usage.get("truncated") is not False or usage.get("state_tokens_dropped") != 0:
            raise InvalidOutput("SDK did not confirm an untruncated prediction")
        return result

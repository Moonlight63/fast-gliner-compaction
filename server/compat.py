"""Load-time fixes for checkpoints saved by transformers 5 under gliner2's transformers<5 pin.

Several GLiNER2.5 checkpoints (Decide-1B, gliner2.5-*-v1) were saved with
transformers 5. Under 4.x:

- their tokenizer_config lists `extra_special_tokens` as a list (4.x wants a
  dict), and Decide-1B's also names `TokenizersBackend` (a 5.x class), so
  `AutoTokenizer` cannot load them;
- its ModernBERT encoder config stores rope settings as `rope_parameters`,
  which 4.x ignores, silently running the sliding-window layers with
  local_rope_theta=10000 instead of 160000. The model then loads without an
  error and answers ~0.5 for everything.

`apply()` must run before gliner2 loads a model. It is idempotent.
"""

import json
import os

_applied = False


def _patch_tokenizer() -> None:
    import transformers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    original = AutoTokenizer.from_pretrained.__func__

    def from_pretrained(cls, name, *args, **kwargs):
        from huggingface_hub import hf_hub_download

        def fetch(filename: str) -> str:
            local = os.path.join(str(name), filename)
            return local if os.path.isfile(local) else hf_hub_download(str(name), filename)

        def saved_config() -> dict:
            with open(fetch("tokenizer_config.json"), encoding="utf-8") as handle:
                return json.load(handle)

        try:
            return original(cls, name, *args, **kwargs)
        except (ValueError, AttributeError, ImportError) as error:
            config = saved_config()
            extra = config.get("extra_special_tokens")
            if "TokenizersBackend" not in str(error) and not isinstance(extra, list):
                raise
        if config.get("tokenizer_class") != "TokenizersBackend":
            # A real 4.x class whose config lists extra_special_tokens (5.x
            # format; 4.x expects a dict): pass them the 4.x way.
            return original(
                cls,
                name,
                *args,
                **{**kwargs, "extra_special_tokens": {}, "additional_special_tokens": list(extra)},
            )
        special = {
            key: config[key]
            for key in ("cls_token", "sep_token", "pad_token", "mask_token", "unk_token")
            if key in config
        }
        return PreTrainedTokenizerFast(
            tokenizer_file=fetch("tokenizer.json"),
            additional_special_tokens=list(config.get("extra_special_tokens") or []),
            model_max_length=config.get("model_max_length", 8192),
            **special,
        )

    transformers.AutoTokenizer.from_pretrained = classmethod(from_pretrained)


def _patch_modernbert_rope() -> None:
    from transformers import ModernBertConfig

    original_init = ModernBertConfig.__init__

    def init(self, *args, rope_parameters=None, **kwargs):
        if isinstance(rope_parameters, dict):
            full = (rope_parameters.get("full_attention") or {}).get("rope_theta")
            sliding = (rope_parameters.get("sliding_attention") or {}).get("rope_theta")
            if full is not None:
                kwargs.setdefault("global_rope_theta", full)
            if sliding is not None:
                kwargs.setdefault("local_rope_theta", sliding)
        original_init(self, *args, **kwargs)

    ModernBertConfig.__init__ = init


def apply() -> None:
    global _applied
    if _applied:
        return
    import transformers

    if int(transformers.__version__.split(".")[0]) < 5:
        _patch_tokenizer()
        _patch_modernbert_rope()
    _applied = True

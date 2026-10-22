"""Exercise stop/capture/position contracts with CPU arrays, without torch."""
import contextlib
import unittest
from types import SimpleNamespace

import numpy as np

from .runtime import LoopVLProbe


class FakeH:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def __call__(self, hidden):
        for hook in self.hooks:
            hook(self, (), hidden)
        return hidden


class FakeRecurrent:
    def __init__(self):
        self.H_module = FakeH()
        self.calls = []
        self.raise_after_h1 = False

    def __call__(self, **kwargs):
        self.calls = []
        self.kwargs = kwargs
        self.calls.append("H1")
        hidden = self.H_module(kwargs["inputs_embeds"] + 1)
        if self.raise_after_h1:
            raise RuntimeError("simulated forward failure")
        self.calls.append("H2")
        hidden = self.H_module(hidden + 2)
        return SimpleNamespace(last_hidden_state=hidden)


class RuntimeContractTests(unittest.TestCase):
    def make_runtime(self):
        runtime = LoopVLProbe.__new__(LoopVLProbe)
        runtime.torch = SimpleNamespace(
            inference_mode=contextlib.nullcontext, long=np.int64,
            tensor=lambda x, dtype=None, device=None: np.array(x, dtype=dtype),
            zeros=lambda shape, dtype=None, device=None: np.zeros(shape, dtype=dtype),
            ones_like=np.ones_like, cat=lambda xs, dim: np.concatenate(xs, axis=dim),
        )
        runtime.device = "cpu"
        runtime.prefix_length = 3
        runtime._prefix_embeddings = lambda: np.zeros((1, 3, 4))
        runtime._positions = lambda n: (np.arange(3+n)[None, :], None, None, None)
        embed = lambda ids: np.repeat(ids[..., None], 4, axis=-1).astype(float)
        runtime.model = SimpleNamespace(
            recurrent_model=FakeRecurrent(),
            language_model=SimpleNamespace(get_input_embeddings=lambda: embed),
        )
        return runtime

    def test_shallow_stops_before_h2_and_removes_hook(self):
        runtime = self.make_runtime()
        early, final = runtime.forward([7], shallow=True)
        self.assertIsNone(final)
        self.assertEqual(runtime.model.recurrent_model.calls, ["H1"])
        self.assertEqual(runtime.model.recurrent_model.H_module.hooks, [])
        self.assertTrue(np.all(early[0, -1] == 8))

    def test_full_capture_and_prefix_mask(self):
        runtime = self.make_runtime()
        early, final = runtime.forward([7, 9], capture_h1=True)
        self.assertTrue(np.all(final == early + 2))
        self.assertEqual(runtime.model.recurrent_model.kwargs["token_type_ids"].tolist(), [[1,1,1,0,0]])
        self.assertFalse(runtime.model.recurrent_model.kwargs["use_cache"])

    def test_failure_also_removes_hook(self):
        runtime = self.make_runtime()
        runtime.model.recurrent_model.raise_after_h1 = True
        with self.assertRaises(RuntimeError):
            runtime.forward([], capture_h1=True)
        self.assertEqual(runtime.model.recurrent_model.H_module.hooks, [])


if __name__ == "__main__":
    unittest.main()

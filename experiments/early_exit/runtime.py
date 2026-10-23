"""Adapter around the unchanged official runtime; all forwards use no cache."""
from __future__ import annotations

import sys
from pathlib import Path


class _StopAfterH1(Exception):
    def __init__(self, hidden):
        self.hidden = hidden


class LoopVLProbe:
    def __init__(self, model_dir: str, device: str):
        import torch
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "runtime"))
        from modeling_loopvl import LoopVLForConditionalGeneration, load_tokenizer_and_processor
        self.torch = torch
        self.device = torch.device(device)
        self.model = LoopVLForConditionalGeneration.from_pretrained(model_dir, device=self.device)
        self.model.eval()
        self.tokenizer, self.processor = load_tokenizer_and_processor(model_dir, self.model.config)
        cfg = self.model.recurrent_model.config
        if (cfg.H_cycles, cfg.L_cycles) != (2, 3):
            raise ValueError("This experiment requires the unmodified H2L3 checkpoint schedule")

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def prepare(self, image_path: Path, prompt: str):
        from PIL import Image
        from hrm_penguin.inference import build_inference_batch
        with Image.open(image_path) as image:
            self.batch = build_inference_batch(
                self.tokenizer, self.processor, image.convert("RGB"), prompt,
                self.model.config, self.device,
            )
        self.prefix_length = self.batch["input_ids"].shape[1]
        self._base_embeddings = None

    def reference(self, max_new_tokens: int) -> list[int]:
        from hrm_penguin.generation import greedy_generate
        with self.torch.inference_mode():
            result = greedy_generate(self.model, self.batch, max_new_tokens=max_new_tokens,
                                     eos_token_id=self.tokenizer.eos_token_id)
        return result[0].tolist()

    def _positions(self, response_length):
        from hrm_penguin.recurrent.positions import build_compact_positions
        t, b = self.torch, self.batch
        return build_compact_positions(
            t.tensor([b["header_length"]], device=self.device),
            t.tensor([b["instruction_length"]], device=self.device),
            t.tensor([response_length], device=self.device),
            visual_token_counts=b["visual_token_counts"], grid_heights=b["grid_heights"],
            grid_widths=b["grid_widths"], grid_h=self.model.config.vision.grid_height,
            grid_w=self.model.config.vision.grid_width,
            rope_mode=self.model.config.recurrent_visual.rope_mode,
        )

    def _prefix_embeddings(self):
        if self._base_embeddings is None:
            b = self.batch
            _, _, visual_mask, _ = self._positions(0)
            visual = self.model.encode_images(b["pixel_values"], b["grid_sizes"], b["merge_sizes"])
            text = self.model.language_model.get_input_embeddings()(b["input_ids"])
            self._base_embeddings = self.model._replace_visual_tokens(text, visual, visual_mask)
        return self._base_embeddings

    def forward(self, response: list[int], *, shallow=False, capture_h1=False):
        """Capture whole H stack outputs, already normalized by H_module.

        The H1-only callback stops execution after the first complete H call.
        No config/weights are modified, no H2 or future response is used by it.
        This single-request diagnostic adapter is intentionally not thread-safe.
        """
        t = self.torch
        captured = []
        def hook(_module, _args, output):
            captured.append(output)
            if shallow:
                raise _StopAfterH1(output)
        handle = None
        with t.inference_mode():
            base = self._prefix_embeddings()
            ids = t.tensor([response], dtype=t.long, device=self.device)
            embeds = t.cat((base, self.model.language_model.get_input_embeddings()(ids)), dim=1) if response else base
            positions, rope, visual, instruction = self._positions(len(response))
            types = t.zeros((1, embeds.shape[1]), dtype=t.long, device=self.device)
            types[:, :self.prefix_length] = 1
            if shallow or capture_h1:
                handle = self.model.recurrent_model.H_module.register_forward_hook(hook)
            try:
                output = self.model.recurrent_model(
                    inputs_embeds=embeds, attention_mask=t.ones_like(types),
                    token_type_ids=types, position_ids=positions, rope_position_ids=rope,
                    visual_mask=visual, instruction_mask=instruction, use_cache=False,
                )
                if capture_h1 and len(captured) != 2:
                    raise RuntimeError("Expected exactly two H calls")
                return (captured[0] if capture_h1 else None), output.last_hidden_state
            except _StopAfterH1 as stop:
                return stop.hidden, None
            finally:
                if handle is not None:
                    handle.remove()

    def propose(self, response: list[int]) -> int:
        with self.torch.inference_mode():
            early, _ = self.forward(response, shallow=True)
            return int(self.model.language_model.lm_head(early[:, -1]).argmax(-1).item())

    def verify(self, response: list[int], drafts: list[int]) -> list[int]:
        with self.torch.inference_mode():
            _, final = self.forward(response + drafts)
            start = self.prefix_length + len(response) - 1
            hidden = final[:, start:start + len(drafts) + 1]
            return self.model.language_model.lm_head(hidden).argmax(-1)[0].tolist()

    def probe(self, reference: list[int], *, chunk_size: int, threshold: float) -> list[dict]:
        t = self.torch
        if not reference:
            return []
        with t.inference_mode():
            early, final = self.forward(reference[:-1], capture_h1=True)
            start = self.prefix_length - 1
            rows = []
            for offset in range(0, len(reference), chunk_size):
                end = min(offset + chunk_size, len(reference))
                h1 = self.model.language_model.lm_head(early[0, start+offset:start+end]).float()
                h2 = self.model.language_model.lm_head(final[0, start+offset:start+end]).float()
                from .readout_metrics import measure_readouts
                measured = measure_readouts(h1, h2, reference[offset:end])
                for local, row in enumerate(measured):
                    pos = offset + local
                    a, b = early[0, start+pos].float(), final[0, start+pos].float()
                    row.update(
                        position=pos, stage="H1", is_eos=reference[pos] == self.tokenizer.eos_token_id,
                        h1_id=row["early_id"], h2_id=row["final_id"],
                        h1_confidence=row["early_confidence"],
                        h1_probability_margin=row["early_probability_margin"],
                        confidence_threshold=threshold,
                        early_hidden_rms=float(a.square().mean().sqrt().item()),
                        final_hidden_rms=float(b.square().mean().sqrt().item()),
                        hidden_cosine=float(t.nn.functional.cosine_similarity(a, b, dim=0).item()),
                    )
                    for field in ("reference_id", "early_id", "final_id"):
                        row[field.replace("_id", "_text")] = self.tokenizer.decode([row[field]], skip_special_tokens=False)
                    row["early_top10_text"] = [self.tokenizer.decode([i], skip_special_tokens=False) for i in row["early_top10_ids"]]
                    rows.append(row)
        return rows

    def check_h1_prefixes(self, reference, rows, limit):
        """Compare H1-only execution with H1 captured during a full causal replay.

        Prefix j contains reference[:j], never the token being predicted or its suffix.
        Different matrix shapes may produce BF16 top-1 changes; report and stop by default.
        """
        from .readout_metrics import select_positions
        result = []
        for pos in select_positions(len(reference), limit):
            token = self.propose(reference[:pos])
            result.append({"position": pos, "shallow_id": token,
                           "captured_h1_id": rows[pos]["h1_id"],
                           "matches": token == rows[pos]["h1_id"]})
        return result

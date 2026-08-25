"""Inference backend for the trained Whisper-prefix + LoRA policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .native_audio_training import load_audio
from .native_training_pipeline import AudioConditionedPolicy, _saved_adapter_path


class NativeAudioPlanner:
    """Load one final native stage and generate prescriptions from real waveforms."""

    def __init__(
        self,
        stage_dir: str | Path,
        *,
        whisper_model: str = "openai/whisper-small",
        language_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        prefix_tokens: int = 16,
        qlora: bool = True,
        cache_only: bool = False,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            WhisperFeatureExtractor,
            WhisperModel,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("native audio inference requires CUDA")
        self.torch = torch
        self.stage_dir = Path(stage_dir).expanduser().resolve()
        whisper_local = Path(whisper_model).expanduser().is_dir()
        language_local = Path(language_model).expanduser().is_dir()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.extractor = None
        self.encoder = None
        if not cache_only:
            self.extractor = WhisperFeatureExtractor.from_pretrained(
                whisper_model,
                return_attention_mask=True,
                local_files_only=whisper_local,
            )
            self.encoder = WhisperModel.from_pretrained(
                whisper_model,
                torch_dtype=dtype,
                local_files_only=whisper_local,
            ).encoder.cuda()
            self.encoder.requires_grad_(False).eval()
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "local_files_only": language_local,
        }
        if qlora:
            model_kwargs["device_map"] = {"": 0}
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        base = AutoModelForCausalLM.from_pretrained(language_model, **model_kwargs)
        lm = PeftModel.from_pretrained(base, _saved_adapter_path(self.stage_dir)).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            language_model,
            trust_remote_code=True,
            local_files_only=language_local,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only batched generation must left-pad prompts so every row's
        # final non-padding token is the point where generation begins.
        self.tokenizer.padding_side = "left"
        checkpoint = torch.load(
            self.stage_dir / "audio_projector.pt", map_location="cpu", weights_only=True
        )
        encoder_dim = int(checkpoint["projector"]["projector.0.weight"].numel())
        self.policy = AudioConditionedPolicy(
            lm,
            encoder_dim=encoder_dim,
            llm_dim=int(lm.config.hidden_size),
            prefix_tokens=prefix_tokens,
            policy_adapter="default",
        )
        self.policy.projector.load_state_dict(checkpoint["projector"])
        self.policy.missing_audio_prefix.data.copy_(checkpoint["missing_audio_prefix"])
        self.policy.projector.to(device="cuda", dtype=dtype)
        self.policy.missing_audio_prefix.data = self.policy.missing_audio_prefix.data.to(
            device="cuda", dtype=dtype
        )
        self.policy.eval()

    def _encode(self, audio_path: str | Path):
        torch = self.torch
        if self.extractor is None or self.encoder is None:
            raise RuntimeError("waveform encoding is unavailable in cache-only mode")
        waveform = load_audio(Path(audio_path), self.extractor.sampling_rate)
        inputs = self.extractor(
            waveform,
            sampling_rate=self.extractor.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        with torch.inference_mode():
            encoder_dtype = next(self.encoder.parameters()).dtype
            hidden = self.encoder(inputs.input_features.cuda().to(encoder_dtype)).last_hidden_state
        input_mask = inputs.attention_mask[0].numpy().astype(np.float32)
        frames = max(1, round(float(input_mask.sum()) * hidden.shape[1] / input_mask.size))
        mask = torch.zeros((1, hidden.shape[1]), dtype=torch.long, device="cuda")
        mask[:, :frames] = 1
        return hidden, mask

    def generate_cached_batch(
        self,
        embedding_paths: list[str | Path],
        prompt_texts: list[str],
        *,
        max_new_tokens: int = 192,
        temperature: float = 0.0,
    ) -> list[str]:
        """Generate a batch from the immutable Whisper embedding cache."""
        if not embedding_paths or len(embedding_paths) != len(prompt_texts):
            raise ValueError("embedding_paths and prompt_texts must have the same non-zero length")
        hidden_rows: list[np.ndarray] = []
        mask_rows: list[np.ndarray] = []
        for embedding_path in embedding_paths:
            with np.load(Path(embedding_path), allow_pickle=False) as cached:
                hidden_rows.append(np.asarray(cached["hidden"], dtype=np.float16))
                mask_rows.append(np.asarray(cached["frame_mask"], dtype=np.int64))
        hidden_dims = {row.shape[1] for row in hidden_rows if row.ndim == 2}
        if len(hidden_dims) != 1 or any(row.ndim != 2 for row in hidden_rows):
            raise ValueError("cached hidden states must be rank-2 with one shared feature width")
        if any(row.ndim != 1 for row in mask_rows):
            raise ValueError("cached frame masks must be rank-1")
        paired_rows = zip(hidden_rows, mask_rows, strict=True)
        if any(hidden.shape[0] != mask.shape[0] for hidden, mask in paired_rows):
            raise ValueError("each cached hidden state and frame mask must share a frame count")

        max_frames = max(row.shape[0] for row in hidden_rows)
        hidden_width = next(iter(hidden_dims))
        padded_hidden = np.zeros((len(hidden_rows), max_frames, hidden_width), dtype=np.float16)
        padded_masks = np.zeros((len(mask_rows), max_frames), dtype=np.int64)
        for index, (hidden_row, mask_row) in enumerate(zip(hidden_rows, mask_rows, strict=True)):
            frames = hidden_row.shape[0]
            padded_hidden[index, :frames] = hidden_row
            padded_masks[index, :frames] = mask_row

        torch = self.torch
        hidden = torch.from_numpy(padded_hidden).cuda()
        frame_mask = torch.from_numpy(padded_masks).cuda()
        tokens = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        )
        generation_kwargs: dict[str, Any] = {
            "do_sample": temperature > 0,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        with torch.inference_mode():
            generated = self.policy.generate_with_audio(
                encoded_audio=hidden,
                audio_frame_mask=frame_mask,
                audio_present=torch.ones(len(prompt_texts), device="cuda"),
                input_ids=tokens.input_ids.cuda(),
                token_attention_mask=tokens.attention_mask.cuda(),
                **generation_kwargs,
            )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [text.strip() for text in decoded]

    def plan_audio(
        self,
        audio_path: str | Path,
        prompt_text: str,
        *,
        max_new_tokens: int = 192,
        temperature: float = 0.2,
    ) -> str:
        torch = self.torch
        hidden, frame_mask = self._encode(audio_path)
        tokens = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=768)
        with torch.inference_mode():
            generated = self.policy.generate_with_audio(
                encoded_audio=hidden,
                audio_frame_mask=frame_mask,
                audio_present=torch.ones(1, device="cuda"),
                input_ids=tokens.input_ids.cuda(),
                token_attention_mask=tokens.attention_mask.cuda(),
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        # Fail early here; the DSP boundary independently validates every action.
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("native policy did not return a JSON object")
        return json.dumps(parsed, ensure_ascii=False)

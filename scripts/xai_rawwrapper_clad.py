# xai_rawwrapper_clad.py
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


CLAD_NB_SAMP = 64600
CLAD_N_FFT = 1024
CLAD_HOP_LENGTH = 256
CLAD_WIN_LENGTH = 1024
CLAD_CENTER = True
CLAD_IS_LOG_MAG = True
CLAD_INPUT_LAYOUT = "bn"


def _to_complex(mag_or_log: torch.Tensor, phase: torch.Tensor, is_log_mag: bool) -> torch.Tensor:
    """
    Build complex STFT from magnitude and phase.
    mag_or_log: [B, F, T], either linear magnitude or log magnitude.
    phase:      [B, F, T] in radians.
    is_log_mag: True if mag_or_log is log magnitude (natural log).
    Returns:    complex tensor [B, F, T].
    """
    mag = mag_or_log.exp() if is_log_mag else mag_or_log
    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    return torch.complex(real, imag)


def _crop_or_pad_2d(wav_bn: torch.Tensor, length: int) -> torch.Tensor:
    """
    wav_bn: [B, N]
    Return [B, length] by crop/pad with zeros.
    """
    B, N = wav_bn.shape
    if N == length:
        return wav_bn
    if N > length:
        return wav_bn[:, :length]
    return F.pad(wav_bn, (0, length - N))


class ISTFTTorch(nn.Module):
    """
    Torch ISTFT backend. Pass `length=expected_len`
    so ISTFT returns the exact original waveform length.
    """
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        center: bool = True,
        length: Optional[int] = None,
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.center = bool(center)
        self.length = None if length is None else int(length)

        win = torch.hann_window(self.win_length, periodic=True)
        self.register_buffer("window", win)

    def forward(self, stft_complex: torch.Tensor) -> torch.Tensor:
        """
        stft_complex: [B, F, T] complex (torch.stft(..., return_complex=True))
        Returns:      waveform [B, N]
        """
        kwargs = {}
        if self.length is not None:
            kwargs["length"] = self.length

        return torch.istft(
            stft_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=stft_complex.real.dtype),
            center=self.center,
            normalized=False,
            onesided=True,
            return_complex=False,
            **kwargs,
        )


class ISTFTConv1D(nn.Module):
    """
    ISTFT implemented with ConvTranspose1d and fixed sinusoid kernels + dynamic OLA normalization.

    Input:  [B, 2F, T] where channels are [Re bins..., Im bins...]
    Output: [B, N] waveform

    If `center=True`, trims both ends by n_fft//2 to match torch.istft(center=True).
    If `length` is provided, final output is forced to exactly that length.
    """
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        center: bool = True,
        length: Optional[int] = None,
    ):
        super().__init__()
        n_fft = int(n_fft)
        hop_length = int(hop_length)
        win_length = int(win_length)

        assert win_length == n_fft, "ISTFTConv1D expects win_length == n_fft for clean overlap-add."

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = bool(center)
        self.length = None if length is None else int(length)

        self.F = n_fft // 2 + 1

        if self.center and self.length is None:
            raise ValueError("ISTFTConv1D with center=True requires length for exact trimming.")

        t = torch.arange(win_length, dtype=torch.float32)
        win = torch.hann_window(win_length, periodic=True).to(dtype=torch.float32)

        # One-sided inverse scaling:
        #   k=0 and k=F-1 (Nyquist if even n_fft) use 1/n_fft
        #   k=1..F-2 use 2/n_fft
        # Inverse uses: Re*cos - Im*sin
        real_kernels = []
        imag_kernels = []
        for k in range(self.F):
            w = 2.0 * math.pi * k / n_fft
            cosw = torch.cos(w * t)
            sinw = torch.sin(w * t)

            if k == 0 or (k == self.F - 1 and (n_fft % 2 == 0)):
                scale = 1.0 / n_fft
            else:
                scale = 2.0 / n_fft

            real_kernels.append(scale * cosw * win)     # multiplies Re
            imag_kernels.append(-scale * sinw * win)    # multiplies Im (minus sign)

        # Channel order must match: x = [Re bins..., Im bins...]
        weight = torch.stack(real_kernels + imag_kernels, dim=0).unsqueeze(1)  # [2F, 1, W]
        self.register_buffer("weight", weight)

        # Dynamic OLA normalization kernel (window^2)
        win_sq = (win ** 2).view(1, 1, -1)  # [1,1,W]
        self.register_buffer("win_sq", win_sq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 2F, T]
        returns: [B, N] (trimmed if center=True, then forced to `length` if provided)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x [B,2F,T], got {tuple(x.shape)}")
        if x.size(1) != 2 * self.F:
            raise ValueError(f"Expected channel dim 2F={2*self.F}, got {x.size(1)}")

        w = self.weight.to(dtype=x.dtype)
        y = F.conv_transpose1d(x, w, stride=self.hop_length)  # [B, 1, N_full]

        # Dynamic OLA norm
        T = x.size(-1)
        ones = torch.ones((1, 1, T), device=x.device, dtype=x.dtype)
        win_sq = self.win_sq.to(dtype=x.dtype)
        norm = F.conv_transpose1d(ones, win_sq, stride=self.hop_length).clamp_min(1e-8)  # [1,1,N_full]

        y = (y / norm).squeeze(1)  # [B, N_full]

        # Center trimming (remove the n_fft//2 padding region on both ends)
        if self.center:
            pad = self.n_fft // 2
            if y.size(-1) > 2 * pad:
                y = y[:, pad:-pad]
            elif y.size(-1) > pad:
                y = y[:, pad:]
            else:
                y = _crop_or_pad_2d(y, 1)

        # Force exact length if requested
        if self.length is not None:
            y = _crop_or_pad_2d(y, self.length)

        return y


class BaseInputAdapter(nn.Module):
    """
    Adapter to enforce a CLAD-compatible input layout.

    input_layout:
      - "bn":  pass waveform as [B, N] (AASIST/RawNet/CLAD encoders)
      - "bnc": pass waveform as [B, N, 1]
    """
    def __init__(self, base_model: nn.Module, input_layout: str = "bn"):
        super().__init__()
        if input_layout not in {"bn", "bnc"}:
            raise ValueError(f"input_layout must be 'bn' or 'bnc', got {input_layout}")
        self.base = base_model
        self.input_layout = input_layout

    def forward(self, wav_bn: torch.Tensor) -> torch.Tensor:
        if wav_bn.dim() != 2:
            raise ValueError(f"Expected wav_bn [B,N], got {tuple(wav_bn.shape)}")
        if self.input_layout == "bn":
            return self.base(wav_bn)
        return self.base(wav_bn.unsqueeze(-1))


class RawWrapperISTFT(nn.Module):
    """
    RawWrapper with M-only forward:

        RawWrapper_f(M; P_fixed) = f( ISTFT( M, P_fixed ) )

    - Forward accepts ONLY magnitude (typically log-magnitude) M: [B,F,T].
    - Phase is set per-utterance via set_phase(...) and is always treated as FIXED (detached).
    - Gradients / attributions flow through M only.
    """
    def __init__(
        self,
        base_model: nn.Module,
        n_fft: int = CLAD_N_FFT,
        hop_length: int = CLAD_HOP_LENGTH,
        win_length: int = CLAD_WIN_LENGTH,
        center: bool = CLAD_CENTER,
        use_conv_istft: bool = False,
        expected_len: Optional[int] = CLAD_NB_SAMP,
        is_log_mag: bool = CLAD_IS_LOG_MAG,
        input_layout: str = CLAD_INPUT_LAYOUT,
    ):
        super().__init__()
        self.base = BaseInputAdapter(base_model, input_layout=input_layout)
        self.expected_len = None if expected_len is None else int(expected_len)
        self.is_log_mag = bool(is_log_mag)

        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.center = bool(center)

        self.F = self.n_fft // 2 + 1

        if use_conv_istft:
            self.istft = ISTFTConv1D(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                center=self.center,
                length=self.expected_len,
            )
            self.conv_mode = True
        else:
            self.istft = ISTFTTorch(
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                center=self.center,
                length=self.expected_len,
            )
            self.conv_mode = False

        # stored fixed phase: [1,F,T] (detached)
        self._phase: Optional[torch.Tensor] = None

    def set_phase(self, phase: torch.Tensor) -> None:
        """
        Set fixed phase used for subsequent forward passes.

        phase can be:
          - [F,T]
          - [1,F,T]
          - [B,F,T] (we keep only the first item as the fixed reference)
        Stored phase is detached to keep it fixed (no gradients).
        """
        if phase.dim() == 2:
            phase = phase.unsqueeze(0)  # [1,F,T]
        if phase.dim() != 3:
            raise ValueError(f"phase must be [F,T] or [B,F,T], got {tuple(phase.shape)}")
        if phase.size(1) != self.F:
            raise ValueError(f"Phase F must equal n_fft//2+1={self.F}, got {phase.size(1)}")

        # paper intent: fixed phase
        self._phase = phase[:1].detach()

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """
        M: [B, F, T] magnitude in linear or log domain depending on is_log_mag.
        Uses internally stored fixed phase set by set_phase().
        """
        if M.dim() != 3:
            raise ValueError(f"M must be [B, F, T], got {tuple(M.shape)}")
        if M.size(1) != self.F:
            raise ValueError(f"Frequency bins F must equal n_fft//2+1 = {self.F}, got {M.size(1)}")
        if self._phase is None:
            raise RuntimeError("Fixed phase not set. Call raw_wrapper.set_phase(phase) before forward().")

        B, _, T = M.shape

        # use stored fixed phase, expand to batch, match device/dtype
        phase = self._phase.to(device=M.device, dtype=M.dtype)  # [1,F,T]
        if phase.size(-1) != T:
            raise ValueError(f"Phase T must match M T. Got phase T={phase.size(-1)} vs M T={T}.")
        if B > 1:
            phase = phase.expand(B, -1, -1)  # [B,F,T]

        # build complex STFT (phase is already detached)
        C = _to_complex(M, phase, self.is_log_mag)  # [B, F, T] complex

        # ISTFT -> waveform
        if self.conv_mode:
            x_ch = torch.cat([C.real, C.imag], dim=1)  # [B, 2F, T]
            wav_bn = self.istft(x_ch)                  # [B, N]
        else:
            wav_bn = self.istft(C)                     # [B, N]

        # safety: enforce expected_len if provided
        if self.expected_len is not None and wav_bn.size(1) != self.expected_len:
            wav_bn = _crop_or_pad_2d(wav_bn, self.expected_len)

        return self.base(wav_bn)


class CLADRawWrapperISTFT(nn.Module):
    """
    CLAD-specific wrapper with hardwired config matching AASIST defaults.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.wrapper = RawWrapperISTFT(
            base_model,
            n_fft=CLAD_N_FFT,
            hop_length=CLAD_HOP_LENGTH,
            win_length=CLAD_WIN_LENGTH,
            center=CLAD_CENTER,
            use_conv_istft=False,
            expected_len=CLAD_NB_SAMP,
            is_log_mag=CLAD_IS_LOG_MAG,
            input_layout=CLAD_INPUT_LAYOUT,
        )

    def set_phase(self, phase: torch.Tensor) -> None:
        self.wrapper.set_phase(phase)

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        return self.wrapper(M)

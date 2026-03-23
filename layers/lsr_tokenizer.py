import torch
from torch import nn
import torch.nn.functional as F


class LocalSpectralTokenizer(nn.Module):
    """
    Local Spectral Regime-guided Adaptive Patching (LSR-Patch) tokenizer.

    Shape contract:
    - Input: x [B, C, L]
    - Output dict:
      - patches: [B, N_max, C, anchor_len]
      - token_mask: [B, N_max] (True = valid token, False = padded token)
      - start: [B, N_max]
      - end: [B, N_max]
      - center: [B, N_max, 1]
      - span: [B, N_max, 1]
      - regime: [B, N_max, regime_dim]
      - n_tokens: [B]
    """

    def __init__(
        self,
        spec_window: int = 32,
        spec_hop: int = 8,
        pelt_penalty: float = 1.0,
        patch_min: int = 8,
        patch_max: int = 64,
        patch_grid: int = 2,
        anchor_len: int = 16,
        patch_gen_alpha: float = 1.0,
        patch_gen_beta: float = 1.0,
    ):
        super().__init__()
        self.spec_window = max(4, int(spec_window))
        self.spec_hop = max(1, int(spec_hop))
        self.pelt_penalty = float(pelt_penalty)
        self.patch_min = max(1, int(patch_min))
        self.patch_max = max(self.patch_min, int(patch_max))
        self.patch_grid = max(1, int(patch_grid))
        self.anchor_len = max(2, int(anchor_len))
        self.patch_gen_alpha = float(patch_gen_alpha)
        self.patch_gen_beta = float(patch_gen_beta)
        self.regime_dim = 3

    def forward(self, x: torch.Tensor):
        # x: [B, C, L]
        bsz, n_vars, total_len = x.shape
        device, dtype = x.device, x.dtype

        sample_tokens = []
        max_tokens = 0

        for b in range(bsz):
            series = x[b]  # [C, L]
            agg = series.mean(dim=0)  # [L], used only for regime boundary detection
            boundaries = self._detect_boundaries(agg)

            patch_list = []
            start_list = []
            end_list = []
            center_list = []
            span_list = []
            regime_list = []

            for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
                if seg_end <= seg_start:
                    continue

                seg_signal = agg[seg_start:seg_end]
                dom_period, bandwidth = self._segment_stats(seg_signal)
                seg_len = seg_end - seg_start
                patch_len = self._generate_patch_length(dom_period, bandwidth, seg_len)

                seg_ratio = float(seg_len) / float(max(total_len, 1))
                regime = torch.tensor(
                    [
                        dom_period / float(max(total_len, 1)),
                        bandwidth,
                        seg_ratio,
                    ],
                    dtype=dtype,
                    device=device,
                )

                for start in range(seg_start, seg_end, patch_len):
                    end = min(seg_end, start + patch_len)
                    if end <= start:
                        continue
                    raw_patch = series[:, start:end]  # [C, patch_t]
                    resized_patch = self._resize_patch(raw_patch)  # [C, anchor_len]
                    patch_list.append(resized_patch)
                    start_list.append(start)
                    end_list.append(end)

                    if total_len > 1:
                        center_pos = ((start + end - 1) * 0.5) / float(total_len - 1)
                    else:
                        center_pos = 0.0
                    span_ratio = (end - start) / float(max(total_len, 1))
                    center_list.append(center_pos)
                    span_list.append(span_ratio)
                    regime_list.append(regime)

            # Defensive fallback: always emit at least one token.
            if len(patch_list) == 0:
                resized_patch = self._resize_patch(series)
                patch_list.append(resized_patch)
                start_list.append(0)
                end_list.append(total_len)
                center_list.append(0.5 if total_len > 1 else 0.0)
                span_list.append(1.0)
                dom_period, bandwidth = self._segment_stats(agg)
                regime_list.append(
                    torch.tensor(
                        [
                            dom_period / float(max(total_len, 1)),
                            bandwidth,
                            1.0,
                        ],
                        dtype=dtype,
                        device=device,
                    )
                )

            patches = torch.stack(patch_list, dim=0)  # [N_i, C, anchor_len]
            start = torch.tensor(start_list, dtype=torch.long, device=device)  # [N_i]
            end = torch.tensor(end_list, dtype=torch.long, device=device)  # [N_i]
            center = torch.tensor(center_list, dtype=dtype, device=device).unsqueeze(-1)  # [N_i, 1]
            span = torch.tensor(span_list, dtype=dtype, device=device).unsqueeze(-1)  # [N_i, 1]
            regime = torch.stack(regime_list, dim=0)  # [N_i, regime_dim]
            n_tokens = patches.shape[0]

            sample_tokens.append((patches, start, end, center, span, regime, n_tokens))
            max_tokens = max(max_tokens, n_tokens)

        # Pad variable token counts to a single batched representation.
        patches_out = x.new_zeros((bsz, max_tokens, n_vars, self.anchor_len))
        token_mask = torch.zeros((bsz, max_tokens), dtype=torch.bool, device=device)
        start_out = torch.zeros((bsz, max_tokens), dtype=torch.long, device=device)
        end_out = torch.zeros((bsz, max_tokens), dtype=torch.long, device=device)
        center_out = x.new_zeros((bsz, max_tokens, 1))
        span_out = x.new_zeros((bsz, max_tokens, 1))
        regime_out = x.new_zeros((bsz, max_tokens, self.regime_dim))
        n_tokens_out = torch.zeros((bsz,), dtype=torch.long, device=device)

        for b, (patches, start, end, center, span, regime, n_tokens) in enumerate(sample_tokens):
            patches_out[b, :n_tokens] = patches
            token_mask[b, :n_tokens] = True
            start_out[b, :n_tokens] = start
            end_out[b, :n_tokens] = end
            center_out[b, :n_tokens] = center
            span_out[b, :n_tokens] = span
            regime_out[b, :n_tokens] = regime
            n_tokens_out[b] = n_tokens

        return {
            "patches": patches_out,
            "token_mask": token_mask,
            "start": start_out,
            "end": end_out,
            "center": center_out,
            "span": span_out,
            "regime": regime_out,
            "n_tokens": n_tokens_out,
        }

    def _resize_patch(self, patch: torch.Tensor) -> torch.Tensor:
        # patch: [C, patch_t] -> [C, anchor_len]
        if patch.shape[-1] == self.anchor_len:
            return patch
        return F.interpolate(
            patch.unsqueeze(0),
            size=self.anchor_len,
            mode="linear",
            align_corners=False,
        ).squeeze(0)

    def _detect_boundaries(self, signal_1d: torch.Tensor):
        total_len = signal_1d.shape[0]
        if total_len <= max(2, self.patch_min):
            return [0, total_len]

        feat_seq, win_starts = self._local_spectral_features(signal_1d)
        n_windows = feat_seq.shape[0]
        if n_windows <= 1:
            return [0, total_len]

        window_boundaries = self._penalized_dp_changepoints(feat_seq)

        # Convert window-index boundaries into time-index boundaries.
        boundaries = [0]
        for w_idx in window_boundaries[1:-1]:
            if w_idx < 0 or w_idx >= len(win_starts):
                continue
            t_idx = int(win_starts[w_idx])
            if t_idx <= boundaries[-1]:
                continue
            # Keep practical segment lengths for this first baseline implementation.
            if t_idx - boundaries[-1] < self.patch_min:
                continue
            if total_len - t_idx < self.patch_min:
                continue
            boundaries.append(t_idx)
        boundaries.append(total_len)
        return boundaries

    def _local_spectral_features(self, signal_1d: torch.Tensor):
        total_len = signal_1d.shape[0]
        window = min(self.spec_window, total_len)
        hop = self.spec_hop

        if total_len <= window:
            starts = [0]
        else:
            starts = list(range(0, total_len - window + 1, hop))
            if starts[-1] != total_len - window:
                starts.append(total_len - window)

        feats = []
        for start in starts:
            seg = signal_1d[start : start + window]
            feat, _, _ = self._spectral_signature(seg)
            feats.append(feat)
        feat_seq = torch.stack(feats, dim=0)  # [W, D]
        return feat_seq, starts

    def _spectral_signature(self, segment: torch.Tensor):
        length = int(segment.shape[0])
        dtype = segment.dtype
        device = segment.device

        if length < 2:
            zero = torch.zeros(1, dtype=dtype, device=device).squeeze(0)
            feat = torch.stack([zero, zero, zero], dim=0)
            return feat, 1.0, 0.0

        spec = torch.fft.rfft(segment, dim=-1)
        power = spec.abs().pow(2)
        if power.numel() <= 1:
            zero = torch.zeros(1, dtype=dtype, device=device).squeeze(0)
            feat = torch.stack([zero, zero, zero], dim=0)
            return feat, float(length), 0.0

        power = power[1:]  # remove DC component
        total_power = power.sum().clamp_min(1e-8)

        dom_idx = int(torch.argmax(power).item()) + 1
        dom_period = float(length) / float(max(dom_idx, 1))

        freq_idx = torch.arange(1, power.shape[0] + 1, dtype=dtype, device=device)
        centroid = (power * freq_idx).sum() / total_power
        variance = ((freq_idx - centroid) ** 2 * power).sum() / total_power
        bandwidth = torch.sqrt(torch.clamp(variance, min=0.0)) / float(max(length, 1))
        bandwidth_val = float(bandwidth.item())

        # Low-dimensional feature vector for changepoint detection.
        dom_freq = torch.tensor(float(dom_idx) / float(max(length, 1)), dtype=dtype, device=device)
        feat = torch.stack([dom_freq, bandwidth, torch.log1p(power.mean())], dim=0)
        return feat, dom_period, bandwidth_val

    def _penalized_dp_changepoints(self, feat_seq: torch.Tensor):
        """
        Penalty-based dynamic programming segmentation over local spectral features.
        This is a simplified PELT-style offline detector (O(W^2)) for the first baseline.
        """
        n_steps, feat_dim = feat_seq.shape
        if n_steps <= 1:
            return [0, n_steps]

        prefix = torch.cat(
            [torch.zeros((1, feat_dim), dtype=feat_seq.dtype, device=feat_seq.device), feat_seq.cumsum(dim=0)],
            dim=0,
        )
        prefix_sq = torch.cat(
            [torch.zeros((1, feat_dim), dtype=feat_seq.dtype, device=feat_seq.device), (feat_seq ** 2).cumsum(dim=0)],
            dim=0,
        )

        def segment_cost(start: int, end: int):
            seg_len = max(end - start, 1)
            seg_sum = prefix[end] - prefix[start]
            seg_sq = prefix_sq[end] - prefix_sq[start]
            # SSE around segment-wise mean in feature space.
            sse = (seg_sq - (seg_sum ** 2) / float(seg_len)).sum()
            return sse

        dp = feat_seq.new_full((n_steps + 1,), float("inf"))
        prev = torch.full((n_steps + 1,), -1, dtype=torch.long, device=feat_seq.device)

        # Offset so penalty is paid only for actual changepoints.
        dp[0] = -self.pelt_penalty
        for end in range(1, n_steps + 1):
            best_val = None
            best_start = 0
            for start in range(0, end):
                val = dp[start] + segment_cost(start, end) + self.pelt_penalty
                if best_val is None or val < best_val:
                    best_val = val
                    best_start = start
            dp[end] = best_val
            prev[end] = best_start

        boundaries = [n_steps]
        cur = n_steps
        while cur > 0:
            cur = int(prev[cur].item())
            boundaries.append(cur)
            if cur == 0:
                break
        boundaries = sorted(set(boundaries))
        if boundaries[0] != 0:
            boundaries.insert(0, 0)
        if boundaries[-1] != n_steps:
            boundaries.append(n_steps)
        return boundaries

    def _segment_stats(self, segment: torch.Tensor):
        _, dom_period, bandwidth = self._spectral_signature(segment)
        return dom_period, bandwidth

    def _generate_patch_length(self, dom_period: float, bandwidth: float, seg_len: int):
        # p* = alpha * T_r / (1 + beta * B_r)
        p_star = self.patch_gen_alpha * dom_period / (1.0 + self.patch_gen_beta * bandwidth)
        upper = min(self.patch_max, seg_len)
        lower = min(self.patch_min, upper)
        p_star = float(max(lower, min(upper, p_star)))

        # Grid-based discretization.
        if self.patch_grid > 1:
            p_disc = int(self.patch_grid * round(p_star / self.patch_grid))
        else:
            p_disc = int(round(p_star))

        p_disc = max(lower, min(upper, p_disc))
        return max(1, p_disc)

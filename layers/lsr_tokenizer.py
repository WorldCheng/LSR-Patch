import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import hashlib


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
        # Small LRU cache for CPU-side segmentation plans:
        # key -> tuple[(start, end, patch_len, dom_period, bandwidth), ...]
        self._seg_cache = OrderedDict()
        self._seg_cache_size = 4096

    def forward(self, x: torch.Tensor):
        # x: [B, C, L]
        bsz, n_vars, total_len = x.shape
        device, dtype = x.device, x.dtype

        sample_tokens = []
        max_tokens = 0

        for b in range(bsz):
            series = x[b]  # [C, L]
            agg = series.mean(dim=0)  # [L], used only for regime boundary detection
            segment_plan = self._build_segment_plan(agg)

            patch_chunks = []
            start_chunks = []
            end_chunks = []
            center_chunks = []
            span_chunks = []
            regime_chunks = []
            denom_center = float(max(total_len - 1, 1))
            denom_span = float(max(total_len, 1))

            for seg_start, seg_end, patch_len, dom_period, bandwidth in segment_plan:
                if seg_end <= seg_start:
                    continue

                seg_len = seg_end - seg_start

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

                # Fast path: batch process full-length patches for the segment.
                seg_len_full = seg_end - seg_start
                n_full = seg_len_full // patch_len
                full_len = n_full * patch_len
                if n_full > 0:
                    full = series[:, seg_start : seg_start + full_len]  # [C, n_full * patch_len]
                    full = full.reshape(n_vars, n_full, patch_len).permute(1, 0, 2).contiguous()  # [n_full, C, patch_len]
                    if patch_len != self.anchor_len:
                        full = F.interpolate(full, size=self.anchor_len, mode="linear", align_corners=False)
                    patch_chunks.append(full)

                    starts = seg_start + torch.arange(n_full, device=device, dtype=torch.long) * patch_len
                    ends = starts + patch_len
                    start_chunks.append(starts)
                    end_chunks.append(ends)

                    center = ((starts.to(dtype) + ends.to(dtype) - 1.0) * 0.5 / denom_center).unsqueeze(-1)
                    span = ((ends - starts).to(dtype) / denom_span).unsqueeze(-1)
                    center_chunks.append(center)
                    span_chunks.append(span)
                    regime_chunks.append(regime.unsqueeze(0).expand(n_full, -1))

                # Tail patch in the same segment (if the segment is not divisible by patch_len).
                if full_len < seg_len_full:
                    start = seg_start + full_len
                    end = seg_end
                    raw_patch = series[:, start:end]  # [C, patch_t]
                    resized_patch = self._resize_patch(raw_patch).unsqueeze(0)  # [1, C, anchor_len]
                    patch_chunks.append(resized_patch)

                    start_t = torch.tensor([start], dtype=torch.long, device=device)
                    end_t = torch.tensor([end], dtype=torch.long, device=device)
                    start_chunks.append(start_t)
                    end_chunks.append(end_t)

                    center_t = ((start_t.to(dtype) + end_t.to(dtype) - 1.0) * 0.5 / denom_center).unsqueeze(-1)
                    span_t = ((end_t - start_t).to(dtype) / denom_span).unsqueeze(-1)
                    center_chunks.append(center_t)
                    span_chunks.append(span_t)
                    regime_chunks.append(regime.unsqueeze(0))

            # Defensive fallback: always emit at least one token.
            if len(patch_chunks) == 0:
                resized_patch = self._resize_patch(series)
                patch_chunks.append(resized_patch.unsqueeze(0))
                start_chunks.append(torch.tensor([0], dtype=torch.long, device=device))
                end_chunks.append(torch.tensor([total_len], dtype=torch.long, device=device))
                center_chunks.append(torch.tensor([[0.5 if total_len > 1 else 0.0]], dtype=dtype, device=device))
                span_chunks.append(torch.tensor([[1.0]], dtype=dtype, device=device))
                if len(segment_plan) > 0:
                    _, _, _, dom_period, bandwidth = segment_plan[0]
                else:
                    dom_period, bandwidth = float(total_len), 0.0
                regime_chunks.append(
                    torch.tensor(
                        [
                            dom_period / float(max(total_len, 1)),
                            bandwidth,
                            1.0,
                        ],
                        dtype=dtype,
                        device=device,
                    ).unsqueeze(0)
                )

            patches = torch.cat(patch_chunks, dim=0)  # [N_i, C, anchor_len]
            start = torch.cat(start_chunks, dim=0)  # [N_i]
            end = torch.cat(end_chunks, dim=0)  # [N_i]
            center = torch.cat(center_chunks, dim=0)  # [N_i, 1]
            span = torch.cat(span_chunks, dim=0)  # [N_i, 1]
            regime = torch.cat(regime_chunks, dim=0)  # [N_i, regime_dim]
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

    @torch.no_grad()
    def _build_segment_plan(self, signal_1d: torch.Tensor):
        """
        Build or reuse a cached segmentation plan for one aggregated series.

        Input:
        - signal_1d: [L] (device can be CPU/GPU)

        Output:
        - list of tuples (start, end, patch_len, dom_period, bandwidth)
          with strictly increasing [start, end) segments that cover [0, L].
        """
        signal_cpu = signal_1d.detach().to(device=torch.device("cpu"), dtype=torch.float32).contiguous()
        cache_key = self._series_cache_key(signal_cpu)
        cached = self._seg_cache.get(cache_key)
        if cached is not None:
            self._seg_cache.move_to_end(cache_key)
            return list(cached)

        plan = self._build_segment_plan_uncached(signal_cpu)
        if self._seg_cache_size > 0:
            self._seg_cache[cache_key] = tuple(plan)
            self._seg_cache.move_to_end(cache_key)
            if len(self._seg_cache) > self._seg_cache_size:
                self._seg_cache.popitem(last=False)
        return plan

    def _series_cache_key(self, signal_1d_cpu: torch.Tensor) -> str:
        payload = signal_1d_cpu.numpy().tobytes()
        digest = hashlib.blake2b(payload, digest_size=16).hexdigest()
        return f"{signal_1d_cpu.numel()}:{digest}"

    def _build_segment_plan_uncached(self, signal_1d_cpu: torch.Tensor):
        total_len = int(signal_1d_cpu.shape[0])
        if total_len <= 0:
            return [(0, 0, 1, 1.0, 0.0)]

        boundaries = self._detect_boundaries(signal_1d_cpu)
        plan = []
        for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
            if seg_end <= seg_start:
                continue
            seg_signal = signal_1d_cpu[seg_start:seg_end]
            dom_period, bandwidth = self._segment_stats(seg_signal)
            seg_len = int(seg_end - seg_start)
            patch_len = self._generate_patch_length(dom_period, bandwidth, seg_len)
            plan.append((int(seg_start), int(seg_end), int(patch_len), float(dom_period), float(bandwidth)))

        if len(plan) == 0:
            dom_period, bandwidth = self._segment_stats(signal_1d_cpu)
            patch_len = self._generate_patch_length(dom_period, bandwidth, total_len)
            plan = [(0, total_len, int(patch_len), float(dom_period), float(bandwidth))]

        return plan

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
        candidate_boundaries = []
        for w_idx in window_boundaries[1:-1]:
            if w_idx < 0 or w_idx >= len(win_starts):
                continue
            t_idx = int(win_starts[w_idx])
            candidate_boundaries.append(t_idx)

        boundaries = [0]
        for t_idx in sorted(set(candidate_boundaries)):
            if t_idx <= boundaries[-1]:
                continue
            if t_idx - boundaries[-1] < self.patch_min:
                continue
            if total_len - t_idx < self.patch_min:
                continue
            boundaries.append(t_idx)
        if boundaries[-1] != total_len:
            boundaries.append(total_len)
        if len(boundaries) == 1:
            boundaries = [0, total_len]
        return boundaries

    def _local_spectral_features(self, signal_1d: torch.Tensor):
        # signal_1d: [L] on CPU. Returns feat_seq [W, D] and window start indices.
        total_len = signal_1d.shape[0]
        window = min(self.spec_window, total_len)
        hop = self.spec_hop

        if total_len <= window:
            starts = [0]
            windows = signal_1d.unsqueeze(0)
        else:
            starts = list(range(0, total_len - window + 1, hop))
            windows = signal_1d.unfold(0, window, hop)
            if starts[-1] != total_len - window:
                starts.append(total_len - window)
                windows = torch.cat([windows, signal_1d[-window:].unsqueeze(0)], dim=0)

        feat_seq, _, _ = self._spectral_signature_batch(windows)  # [W, D]
        return feat_seq, starts

    def _spectral_signature(self, segment: torch.Tensor):
        feat, dom_period, bandwidth = self._spectral_signature_batch(segment.unsqueeze(0))
        return feat[0], float(dom_period[0].item()), float(bandwidth[0].item())

    def _spectral_signature_batch(self, segments: torch.Tensor):
        # segments: [N, L]
        n_seg, length = segments.shape
        length = int(length)
        dtype = segments.dtype
        device = segments.device

        if length < 2:
            zero = torch.zeros((n_seg,), dtype=dtype, device=device)
            feat = torch.stack([zero, zero, zero], dim=-1)
            dom_period = torch.ones((n_seg,), dtype=dtype, device=device)
            bandwidth = torch.zeros((n_seg,), dtype=dtype, device=device)
            return feat, dom_period, bandwidth

        spec = torch.fft.rfft(segments, dim=-1)
        power = spec.abs().pow(2)
        if power.shape[-1] <= 1:
            zero = torch.zeros((n_seg,), dtype=dtype, device=device)
            feat = torch.stack([zero, zero, zero], dim=-1)
            dom_period = torch.full((n_seg,), float(length), dtype=dtype, device=device)
            bandwidth = torch.zeros((n_seg,), dtype=dtype, device=device)
            return feat, dom_period, bandwidth

        power = power[..., 1:]  # remove DC component
        total_power = power.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        dom_idx = power.argmax(dim=-1) + 1  # [N]
        dom_period = float(length) / dom_idx.to(dtype).clamp_min(1.0)

        freq_idx = torch.arange(1, power.shape[-1] + 1, dtype=dtype, device=device).unsqueeze(0)  # [1, F]
        centroid = (power * freq_idx).sum(dim=-1, keepdim=True) / total_power
        variance = ((freq_idx - centroid) ** 2 * power).sum(dim=-1, keepdim=True) / total_power
        bandwidth = torch.sqrt(torch.clamp(variance, min=0.0)).squeeze(-1) / float(max(length, 1))

        # Low-dimensional feature vector for changepoint detection.
        dom_freq = dom_idx.to(dtype) / float(max(length, 1))
        feat = torch.stack([dom_freq, bandwidth, torch.log1p(power.mean(dim=-1))], dim=-1)
        return feat, dom_period, bandwidth

    def _segment_sse_cost_from_prefix(
        self,
        prefix: torch.Tensor,
        prefix_sq: torch.Tensor,
        starts: torch.Tensor,
        end: int,
    ) -> torch.Tensor:
        """
        O(1) segment SSE query from cumulative sums.

        Inputs:
        - prefix/prefix_sq: [W+1, D]
        - starts: [K] candidate start indices tau in (tau, end]
        - end: scalar segment end index

        Output:
        - cost: [K], C(tau+1:end) with multivariate within-segment SSE in feature space.
        """
        dtype = prefix.dtype
        seg_len = (end - starts).to(dtype=dtype).unsqueeze(-1)  # [K, 1]
        seg_sum = prefix[end].unsqueeze(0) - prefix.index_select(0, starts)  # [K, D]
        seg_sq = prefix_sq[end].unsqueeze(0) - prefix_sq.index_select(0, starts)  # [K, D]
        sse = (seg_sq - (seg_sum ** 2) / seg_len).sum(dim=-1)  # [K]
        return torch.clamp(sse, min=0.0)

    def _penalized_dp_changepoints(self, feat_seq: torch.Tensor):
        """
        True PELT (Pruned Exact Linear Time) over local spectral feature windows.

        Recurrence:
            F(t) = min_{tau < t} [F(tau) + C(tau+1:t) + beta]

        where C is segment SSE around the segment mean in feature space.

        Candidate pruning:
        - Maintain an active candidate set R.
        - After solving F(t), prune tau in R when
              F(tau) + C(tau+1:t) > F(t)
          (SSE cost satisfies PELT's additivity condition with K=0).

        Shapes:
        - feat_seq: [W, D]
        - returns sorted boundary indices in [0, W], including 0 and W.
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

        dp = feat_seq.new_full((n_steps + 1,), float("inf"))
        prev = torch.full((n_steps + 1,), -1, dtype=torch.long, device=feat_seq.device)
        candidates = torch.zeros((1,), dtype=torch.long, device=feat_seq.device)  # R_1 = {0}

        # Offset so penalty is paid only for actual changepoints.
        dp[0] = -self.pelt_penalty
        for end in range(1, n_steps + 1):
            # Evaluate only active candidates in R_t.
            sse = self._segment_sse_cost_from_prefix(prefix, prefix_sq, candidates, end)  # [|R_t|]
            vals = dp[candidates] + sse + self.pelt_penalty  # [|R_t|]

            best_idx = int(torch.argmin(vals).item())
            best_start = candidates[best_idx]
            dp[end] = vals[best_idx]
            prev[end] = best_start

            # PELT pruning: keep tau where F(tau) + C(tau+1:end) <= F(end), then add end.
            keep_mask = (dp[candidates] + sse) <= dp[end]
            kept = candidates[keep_mask]
            end_tensor = torch.tensor([end], dtype=torch.long, device=feat_seq.device)
            candidates = torch.cat([kept, end_tensor], dim=0)

        boundaries = [n_steps]
        cur = n_steps
        while cur > 0:
            next_cur = int(prev[cur].item())
            if next_cur < 0:
                break
            cur = next_cur
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

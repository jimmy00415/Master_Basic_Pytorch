# Convolution basics and multi-filter operations (PyTorch) — Practical guide

This guide explains every code section in the notebook and the core concepts behind it.

## Data and shapes

- Image tensors (PyTorch): NCHW = [batch, channels, height, width].
- RGB images have C=3; grayscale has C=1.
- For visualization with matplotlib, convert to uint8 HWC (height, width, channels).

## Conv2d: parameters and output size

- Signature: `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)`
- Output spatial size (square kernel K):
  - H_out = floor((H + 2P − D·(K−1) − 1)/S + 1)
  - W_out = floor((W + 2P − D·(K−1) − 1)/S + 1)
- "Same" padding for K=3, D=1: use P=1 with S=1 to keep H_out=H and W_out=W.

## Functional conv vs module

- Functional: `F.conv2d(input, weight, bias=None, stride=…, padding=…, groups=…)` (you provide weights).
- Module: `nn.Conv2d(…)` (layer owns learnable weights, initialized randomly).
- Use module for learned filters; functional when applying fixed kernels (edge/blur).

## Depthwise per-channel filtering with groups

- To apply the same 2D kernel independently to each RGB channel, use `groups=C` with weight shape `[C, 1, kH, kW]`.
- In the notebook, we repeat the 2D kernel across channels and call `F.conv2d(..., groups=3)`.

## Demo-1: edge detection and blurring

- Edge kernel (Laplacian-like) emphasizes intensity changes (outlines/edges).
- Box blur averages a local patch, smoothing detail/noise; larger kernels (9×9) blur more.
- Always clamp and convert to uint8 for display to avoid wrap-around artifacts.

## Demo-2: valid convolution vs padding

- Valid (no padding) shrinks output because the kernel must fit fully inside the image.
- Padding adds a border (usually zeros) to control output size (or preserve it with appropriate P).
- Use the output-size formula to reason about spatial changes.

## Demo-3: Conv2d module and feature maps

- A conv layer learns multiple filters (out_channels feature maps).
- Early filters often detect edges/orientations; later filters combine into textures/shapes.
- Use `eval()` and `torch.no_grad()` for inference-only demonstrations.

## Chaining multiple convs

- Stacking convs increases the effective receptive field (larger context seen by deeper layers).
- Keep `stride=1` initially to isolate feature extraction; downsample later with pooling or `stride>1`.
- Ensure the next layer’s `in_channels` matches previous `out_channels`.

## Visualizing weights and features

- Weights: shape `[out_c, in_c, kH, kW]`. Visualize slices to see learned patterns per input channel.
- Feature maps: show a few channels to understand activations.

## Pooling

- `MaxPool2d(2,2)`: picks the maximum in each 2×2 region; adds translation tolerance and halves spatial size.
- `AvgPool2d(2,2)`: averages values; smoother, less selective than max.
- Module vs functional (`F.max_pool2d`) produce identical results with same parameters.

## Good practices and pitfalls

- Don’t `softmax` before `CrossEntropyLoss` — pass logits.
- Track shapes; print input/output sizes when changing stride/padding.
- Use `.eval()` and `with torch.no_grad()` for forward-only demos.
- Use consistent dtype/range when plotting (float32 0..255 → clamp → uint8).
- Avoid brittle data loading (`git clone` in notebooks); prefer local files or safe fallbacks.

## Quick reference

- Same padding (K=3, D=1): P=1, S=1.
- Output size checker (square kernels): `H_out = ((H + 2P − D·(K−1) − 1) // S) + 1`.
- Depthwise per-channel filter: `groups=C`; weight `[C,1,kH,kW]`.

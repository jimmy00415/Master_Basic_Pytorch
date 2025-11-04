# COMP3057 Study Guide — From Tensors to CNNs and LSTMs

Audience: fresher AI engineer using PyTorch and Jupyter

## Roadmap (follow your numbered notebooks)

1. Introduction to Data — tensors, splits, normalization
2. Images as Data — NCHW layout, transforms, dataloaders
3. PyTorch Basics — autograd, modules, training loop
4. Train an AI Model — problem framing, losses, metrics, LR
5. Linear Layer (y=2x+1) — MSE + SGD training loop
6. ReLU — nonlinearity, variants, pitfalls
7. One-Hot Encoding — when to use indices vs one-hot
8. Convolution Basics — kernels, stride, padding, groups
9. CNN Architectures — conv blocks, BN, dropout
10. LeNet — classic small CNN for MNIST
11. VGG/AlexNet/ResNet — deep architectures, residuals
12. Lecture Demo — training dynamics and debugging
13. RNN Training — sequences, shapes, clipping
14–15. LSTM Time Series — windowing, scaling, leakage
16–19. Assignments/Labs — integrate end to end

## Core concepts you must master

- Shapes and semantics
  - Images: [N, C, H, W]; Sequences (batch_first): [N, T, F]; Tabular: [N, D]
  - Always print shapes while debugging; mismatches cause most errors
- PyTorch model structure
  - Subclass `nn.Module`; define layers in `__init__`, compute in `forward`
  - `model.train()` vs `model.eval()`; `with torch.no_grad()` for eval
- Losses and outputs
  - Regression: float outputs; `MSELoss`/`L1Loss`
  - Single-label classification: logits [N, C], Long targets [N]; `CrossEntropyLoss`
  - Multi-label: logits [N, C], float targets [N, C]; `BCEWithLogitsLoss`
- Optimization
  - `zero_grad()` → `backward()` → `step()`
  - Start with Adam(1e-3) or SGD(1e-2, momentum=0.9); tune LR first
- Data loading
  - Implement `Dataset`, use `DataLoader`; use `num_workers`, `pin_memory` with CUDA
- Regularization
  - Weight decay, dropout, data augmentation, early stopping
  - Monitor validation metrics; don’t tune on test
- Initialization and normalization
  - Kaiming for ReLU nets; Xavier for tanh/sigmoid
  - BatchNorm/LayerNorm stabilize and speed training
- Evaluation and reproducibility
  - Separate train/val/test; save best on val
  - Set seeds (`torch`, `numpy`, `random`) and record versions
- Debugging patterns
  - Overfit a tiny subset first (sanity check)
  - Assert shapes; detach and log intermediate tensors

## Minimal training loop contract

Inputs: model, optimizer, loss, dataloader, device, epochs

Steps per epoch:
- model.train(); loop batches
- forward → loss
- optimizer.zero_grad(); loss.backward(); optimizer.step()
- model.eval(); run validation with `torch.no_grad()`

## Convolutions and pooling essentials

- Conv2d parameters: in_channels, out_channels, kernel, stride, padding, dilation, groups
- Output size (square K): `H_out = ((H + 2P − D·(K−1) − 1)//S) + 1`
- Depthwise per-channel: `groups=C`, weight `[C,1,kH,kW]`
- Pooling reduces spatial size and adds invariance: Max vs Avg

## LSTM/time series essentials

- Prepare sliding windows X: [N, T, F]; y: [N, target_dim]
- Batch-first LSTM: `nn.LSTM(input_size=F, hidden_size=H, batch_first=True)`
- Use gradient clipping; evaluate with MAE/MSE/RMSE; avoid leakage (fit scaler on train only)

---

## Practice checklist (do these to cement learning)

1) Shapes and tensors
- [ ] Convert a NumPy array to a tensor and move to GPU if available
- [ ] Verify shapes for: image batch [N,C,H,W], sequence batch [N,T,F]

2) Linear regression (Notebook 05)
- [ ] Train `nn.Linear(1,1)` on y=2x+1 with MSE + SGD; try lr in {1e-2, 1e-3, 1e-4}
- [ ] Plot loss vs epochs; record final (w, b) and compare to ground truth

3) Classification head
- [ ] Build a small MLP for MNIST/FashionMNIST with logits [N,10]
- [ ] Train with `CrossEntropyLoss` and accuracy; confirm targets are Long, not one-hot

4) Convolutions (Notebook 08)
- [ ] Compute output shapes by hand for several (K,S,P) combos; verify with a helper function
- [ ] Visualize 3×3 edge kernel and 9×9 blur; confirm behavior on an RGB image using depthwise groups
- [ ] Chain two convs; verify in/out shapes; visualize a few feature maps

5) Pooling
- [ ] Compare MaxPool(2,2) vs AvgPool(2,2) on the same feature map; explain differences

6) Training hygiene
- [ ] Add early stopping on validation metric and save best checkpoint
- [ ] Log LR and try a scheduler (e.g., StepLR or CosineAnnealingLR)

7) RNN/LSTM (Notebooks 13–15)
- [ ] Implement a windowed dataset for a univariate time series (T=24)
- [ ] Train `nn.LSTM` with hidden_size in {32, 64}; report RMSE on a hold-out set
- [ ] Inverse-transform predictions for metric reporting (if scaled)

8) Debugging
- [ ] Overfit on a tiny subset (e.g., 32 samples) until training loss ≈ 0
- [ ] Print and assert shapes through the model; fix any mismatches

---

## Notes tied to your repo

- One filename has a trailing space: `15_madrid_air_quality_lstm .ipynb`. Consider renaming to `15_madrid_air_quality_lstm.ipynb` for consistency.
- In `05_Exercise_LinearLayer_Learn_.ipynb`, start with lr=1e-3 or 1e-2 for faster convergence on the toy problem.

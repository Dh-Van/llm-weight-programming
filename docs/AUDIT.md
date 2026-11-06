# Skeptic's Audit Report: weightspace-mnist

> **Context.** This is the hostile pre-publication audit of the repo, kept as written. It checked a *draft* set of claims ("zero training data", "~97% / ~70%", "at most 4 calibration steps"). The verdict below is about those draft claims, which were never published. The [README](../README.md) is the corrected write-up. Paths to `runs/<id>/` correspond to `results/<id>/` for the three headline runs. The ablation section has been updated with the final numbers from [`audit_ablation.py`](../audit_ablation.py).

Audited 2026-11-05. Covers every file in `src/weightspace/`, every run in `runs/` (23 logs, 83 logged calculator snippets), `README.md`, and the saved `best_weights.json` / `kept/*.npy` files. Comments and docstrings were not trusted; findings come from execution paths and saved artifacts.

## 1. Verdict: **CRITICAL LEAK FOUND**

The network code, the metric and the test splits are clean. **Two of the headline claims are false, though, and the numbers are overstated:**

1. **"Zero training-data statistics" is false.** The harness runs every neuron on 5,000 labelled *training* images. It returns per-class means, standard deviations, firing rates, top-50 label histograms and the full neuron-by-neuron correlation matrix.
2. **"The LLM programmed the combine weights" is false in the sense HN will read it.** In every top run, the final layer is the output of a regularized **linear discriminant analysis (LDA)**: `np.linalg.solve` / `np.linalg.inv`, run in the calculator on that harness-computed class-mean and covariance table. The model then copied the printed numbers into the JSON.
3. **The numbers are not ~97% and ~70%.** No run anywhere in `runs/` has test accuracy above 95.93% (MNIST) or 67.90% (Imagenette). ~70% is a *validation* number (72.9%), measured on training images.

A cold-room PyTorch re-evaluation reproduces the logged test numbers exactly, so nothing is wrong with the arithmetic. The problem is what the numbers are attributed to.

## 2. Evidence

### Check 1: Training-data leakage (Regime B)

The word "Regime B" doesn't appear anywhere in the repo. The README itself says the LLM "looks at a handful of real digits as raw pixel values" (README.md:7-9).

| What enters the LLM's context | Where | Headline MNIST run `explore-20261105-120713` | Imagenette `explore-imagenette-20261105-231021` |
|---|---|---|---|
| Raw training images, as pixel text | `explore.py:514-516`, `datasets.py:69-76` | 100 digits (`round_1_prompt.md`, 87 KB) | 100 photos as 16×16 RGB hex, **plus** 2 PNG contact sheets (`explore.py:517-523`) |
| **Per-neuron, class-conditional statistics on 5,000 labelled training images** | `neuron_stats` `explore.py:255-266`, sent at `:603-607`; split = training set (`datasets.py:51-54`, `data.py:57-62`) | 200 neurons × 10 class means = 2,000 numbers, plus std, firing rate and top-50 label counts | 475 neurons → 4,750 class means |
| **Full statistics table preloaded into the calculator**: `MEANS` (10×N class means), `MEAN`, `STD`, `CORR` (N×N) | `write_table` `explore.py:365-388`, wired in at `:493` and `:640` | 40,000 correlation entries | 225,625 correlation entries |
| Highest-activating training photos per neuron, as images | `top_sheet` `explore.py:212-234`, sent at `:612-618` | n/a | 14 sheets, 6 photos per neuron |
| Validation (training-set) mistakes with full pixels, scores and per-neuron blame | `combine_mistakes` `explore.py:290-337`, sent at `:672-673` | 2 rounds × 10 images | 1 round × 10 images |

**Answer to your direct question:** yes. During the "understand each neuron" phase the harness fed real dataset images through every neuron and showed the LLM class-conditional firing rates and means against ground-truth labels. It then gave the LLM the class-conditional covariance structure as machine-readable arrays. Data statistics didn't come in through a side door; they came in through the front door, on purpose.

**What was *not* found:** no calculator snippet reads dataset files. All 83 snippets were checked for file, OS, `_np.` and dataset-path access, and the 7 refusals were all JSON/`save`/`eval` formatting attempts. There is also no PCA/SVD of pixels and no pixel-mean templates computed by the harness. The feature extractors themselves (oriented strokes, hole detectors, ink grids, Sobel/Laplacian/colour-opponent edges) are hand-designed by the model.

### Check 2: Closed-form solvers / pretrained weights

**Pretrained weights: clean.** There is no torch, torchvision, timm, HF, CLIP or sklearn anywhere. Dependencies are numpy, pillow, rich and gymnasium (`pyproject.toml`), and there are no `.pth` files.

**Closed-form solver: FOUND, in every top run.** The final layers of all three runs below came from regularized LDA computed in the calculator:

- **MNIST 95.93%** (`runs/explore-20261105-120713/calculator.jsonl`, snippets 4-9):
  `W=T-B; Ws=(1-a)*W+a*np.diag(np.diag(W)); w=np.linalg.solve(Ws,MEANS.T).T; b=-0.5*np.sum(w*MEANS,1)`.
  Snippet 8 prints the weight row `[8.6,6.1,-16.9,-24.6,15.6,-0.5,...]` and bias `[-88.8,-63.8,-81.8,...]`. `round_5_combined.json` has **identical** values. The model even says so (`round_5_reply.md`, line 1): *"scored by a linear discriminant built in one closed-form step."* Rounds 6-7 leave all 2,000 weights unchanged and only nudge biases, based on the validation confusion matrix.
- **Imagenette 67.77%** (`explore-imagenette-20261105-193149`, snippets 5-10): `A=np.linalg.solve(Rw+lam*np.eye(277), m.T).T`, with the shrinkage `lam` swept against a separability proxy computed from `MEANS`. Snippet 8 also re-estimates class priors from a *pasted validation confusion matrix* (`+np.log(p)`).
- **Imagenette 67.90%** (`explore-imagenette-20261105-231021`, snippets 12-18): the same LDA (`np.linalg.solve`), plus `np.linalg.lstsq(MEANS.T, MEAN)` to estimate class priors. This is the run's best round, and the shrinkage `lam` and the feature subset were chosen by trying 4 subsets against the same data-derived Mahalanobis proxy (snippet 15).

The calculator's own rule (`explore.py:147-150`) bans only "training or iterative optimisation". A closed-form least-squares/LDA fit on labelled training statistics is still **fitting**. HN will call it that, and so would a reviewer.

`grad` in the 231021 log is image-gradient (Sobel) feature design, not gradient descent. That part is fine.

### Check 3: Metric and splits

- **Metric: clean.** `evaluate.py:19-29` takes `pred = argmax` over 10 scores and computes `(pred == y).mean()` over the full array. That is strict 10-way top-1. It isn't per-class binary or top-5, and there's no `break`: `Network.predict` (`network.py:159-160`) and `Combined.forward` (`explore.py:409-416`) cover every sample.
- **Splits: test never shown to the model.** Feedback uses a 5,000-image slice of the *training* set (`datasets.py:51-54`). The test sets are MNIST t10k (`data.py:50-54`) and Imagenette's official val, 3,925 images (`datasets.py:231`). The best round is picked by validation (`explore.py:663`).
- **Caveats.**
  - (a) Test accuracy is computed every combine round (`explore.py:653`) and **printed to your console** (`:666-667`). Across 23 runs you added harness features (calculator → overlap → overlap-table → layered) after seeing those printed test scores. That is researcher-level test-set adaptation. Disclose it.
  - (b) For MNIST, the 100 prompt samples are not excluded from the validation slice. For Imagenette they are (`datasets.py:71`). This is minor.
  - (c) Feedback per combine round is far more than a scalar. It includes accuracy, a 10×10 confusion matrix, per-class accuracy, a per-neuron score-gap decomposition for the top 5 confused pairs, and 10 full-pixel mistake dumps.

### Check 4: Genuine neural network?

**Yes, clean.** Inside the network the only operations are conv2d (stride 1), dense, ReLU, abs, tanh, sigmoid, max/avg pool, flatten, scale and shift (`network.py:126-157`). There's no HOG, FFT, cv2 or procedural pixel slicing. `abs` is non-standard but it's a valid activation.

Exact preprocessing before the first weight:
- **MNIST:** `uint8 / 255` (`data.py:52`).
- **Imagenette:** RGB convert → centre square crop of the short side → PIL bicubic resize to 64×64 → `uint8 / 255` (`datasets.py:214-219, 194`).

### Check 5: Evaluation budget and parameters

Your "at most 4 calibration steps" claim is **not accurate**. The real count for each run:

| Run | Rounds that sent data-derived feedback to the LLM | Dataset passes |
|---|---|---|
| MNIST 120713 | 4 explore (1 failed to parse) + 2 combine feedback = **5 data feedback messages**, plus the preloaded stats table | 9 feature files × 5k train; 3 combine × (5k val + 10k test) |
| Imagenette 193149 | 4 explore + 2 combine feedback | 12 files × 5k; 3 × (5k + 3,925) |
| Imagenette 231021 | 5 explore + 1 combine feedback | 14 files × 5k; 2 × (5k + 3,925) |
| Imagenette 195027 | 8 explore + 2 combine feedback | 19 files × 5k; 3 × (5k + 3,925) |

There are also 23 runs in total, and the best one is reported.

| Final model | Architecture | Parameters (torch count) |
|---|---|---|
| MNIST `explore-20261105-120713` | 9 parallel conv/pool branches → 200 features → Linear(200,10) | **6,779** |
| Imagenette `explore-imagenette-20261105-193149` | 12 branches → 277 features → Linear(277,10) | **5,304** |
| Imagenette `explore-imagenette-20261105-231021` | 11 branches (up to 4 conv layers deep) → 375 features → Linear(375,10) | **57,408** |

### Check 6: Cold-room verification

`audit_verify.py` imports nothing from this repo. It rebuilds each model from the JSON tensors with `nn.Conv2d` / `nn.Linear` / `nn.ReLU` / pools in float64. It downloads fresh `torchvision.datasets.MNIST(train=False)` and `torchvision.datasets.Imagenette(split="val", size="160px")` into an empty folder and evaluates 100% of the samples under `torch.no_grad()`:

| Model | Samples | Top-1 (cold-room) | Logged test | Logged val |
|---|---|---|---|---|
| MNIST 120713 | 10,000 | **95.93%** (9593) | 95.93% | 96.28% |
| Imagenette 193149 | 3,925 | **67.77%** (2660) | 67.77% | 72.92% |
| Imagenette 231021 | 3,925 | **67.90%** (2665) | 67.90% | 72.52% |
| Imagenette 195027 | 3,925 | **62.96%** (2471) | 62.96% | 71.76% |

MNIST per-class accuracy ranges from 93.4% (digit 4) to 98.5% (digit 0). Imagenette 231021 ranges from 51.8% (chain saw) to 77.9% (garbage truck).

### Bonus: how much is the LLM vs. the LDA solver?

This is the ablation HN will ask for. [`audit_ablation.py`](../audit_ablation.py) runs **the exact LDA recipe from the calculator logs** on the same 5,000 training images, without any LLM features:

| Features fed to the same LDA | MNIST test | Imagenette test |
|---|---|---|
| raw pixels (MNIST 784 / Imagenette pooled 16×16 RGB) | 86.16% | 34.78% |
| avg-pooled pixels (MNIST 7×7 / Imagenette 4×4) | 84.78% | 40.10% |
| random ReLU projections, same width as the LLM bank (3 seeds) | 86.22–86.92% | 37.73–38.73% |
| 1,000 random ReLU projections | 92.39% | 38.75% |
| **LLM-designed features** (200 / 277) | **95.93%** | **67.77%** |

So the LLM's hand-designed feature bank is worth about **+9 points on MNIST and +29 points on Imagenette over random features at equal width**. That is a real, defensible result. The classifier on top of it, though, is standard LDA fitted to training data.

## 3. Claims

### Will get you roasted (and why)

| Claim | Problem |
|---|---|
| "~97% on MNIST" | Best test result anywhere is **95.93%**. |
| "~70% on Imagenette" | Best test is **67.90%**. 70%+ only shows up on the validation slice of the training set. |
| "ZERO training data pixel statistics" | Class-conditional means and covariance on 5,000 labelled training images were computed and handed to the model, plus 100 raw images and top-activation photos. |
| "LLM synthesized the weights" (of the classifier layer) | The final layer is `np.linalg.solve(within-class cov, class means)`, i.e. LDA, run in a Python tool and copied out. |
| "ZERO gradient descent" with the implication of no fitting | True literally, but a closed-form fit is still fitting. |
| "at most 4 evaluation-loop calibration steps" | 6-10 rounds of data-derived feedback per run, plus a preloaded statistics table, best of 23 runs. |
| "no tools / can't run code" (README.md:57-58) | True only for `ws-generate`. All top results used `--calculator`. |

### Safe wording

> An LLM (Claude Opus), with no gradient descent, hand-designed a bank of 200 convolutional feature detectors for MNIST: oriented strokes, enclosed-hole detectors and ink-density grids. It wrote every conv weight directly as numbers. It could see 100 example digits. It refined the detectors over 3 rounds of feedback, in which the harness reported each neuron's per-class mean activation on 5,000 labelled training images.
>
> The final 10-way layer is a regularized linear discriminant (LDA) that the model derived in a calculator from the harness's per-class mean and covariance tables. That is closed-form, not iterative.
>
> The resulting 6,779-parameter network scores **95.9%** on the untouched MNIST test set (10,000 images). The same LDA on 200 random ReLU features scores ~86-87%, so the LLM's feature design accounts for ~9 points (+29 on Imagenette). On Imagenette (10 classes, 64×64) the same process reaches **67.9%** top-1 on the official 3,925-image validation set, against 10% chance.
>
> The test set was never shown to the model. We ran 23 configurations while developing the harness and report the best.

If you want the stronger "zero data statistics" claim, rerun with the class-conditional statistics removed (`neuron_stats` → unlabelled stats only, no `--overlap-table`), the calculator off for combine rounds, and `--samples 0`. Report whatever that gets. The existing non-explore runs suggest roughly 30-65%.

# Poisson MLE Deconvolution — What Happens

A step-by-step description of the `PoissonMLE` algorithm in DOLPHIN. This file
describes **what** happens at each stage and **why** it makes physical sense. It
contains no derivations; the companion file
[`poisson-mle-deconvolution-derivations.md`](poisson-mle-deconvolution-derivations.md)
gives the full mathematical treatment.

## The imaging problem

Widefield fluorescence microscopy is a *linear*, *shift-invariant* (to good
approximation) physical process. A thin sheet of fluorophores emits photons; the
optical system spreads each point of emission into a blur — the **point spread
function** (PSF), set by diffraction and aberrations. What the camera records is
the sum of all these overlapping blurs, plus a background:

$$
g(x,y,z) \;=\; \bigl(\mathrm{PSF} \ast f\bigr)(x,y,z) + b .
$$

Here

- $g$ is the measured image (the data),
- $f$ is the unknown object we want,
- $\ast$ is convolution,
- $b$ is a nearly uniform background (auto-fluorescence, dark current, etc.).

The blur $\ast$ is easy to do and hard to undo. "Deconvolution" means estimating
$f$ from $g$ and the known PSF. Because the PSF has smeared high-frequency detail
away (it is a low-pass filter), that detail is **not** in $g$ and cannot simply be
recovered — it must be *inferred*. Inferring is ill-posed: many different objects
$ f $ blur to the same $g$, and the noise in the data decides which one the
algorithm settles on. So the algorithm needs a model of the noise and a way to
prefer smooth, plausible objects over noisy ones.

## The data model: photons follow a Poisson law

Fluorescence images are dominated by **shot noise**: photons arrive as discrete,
random events. The count in each pixel is a random integer whose uncertainty
scales with the square root of the count. This is exactly a **Poisson process**.

So the model says: given the true (blurred, background-included) intensity
$m = Hf + b$, the number of photons we would observe in a pixel $i$ is a Poisson
random variable with mean $m_i$:

$$
g_i \sim \mathrm{Poisson}(m_i) .
$$

The "best" estimate of $f$ under this model is the one that makes the observed
counts $g$ **most probable** — the **maximum likelihood estimate** (MLE). The
quantity the algorithm drives down is the negative log-likelihood of the data,
up to terms that do not depend on $f$. For a Poisson model this works out to the
**I-divergence** (a form of relative entropy), which penalizes, pixel by pixel,
the mismatch between the measured counts and the model's predicted counts:

$$
J(f) \;=\; \sum_i \Bigl[ \; g_i \ln\!\tfrac{g_i}{m_i} \;-\; g_i \;+\; m_i \;\Bigr],
\qquad m_i = (Hf)_i + b .
$$

This is the objective the algorithm minimizes. Unlike a Gaussian (least-squares)
model, the Poisson/I-divergence model is *count-based*: a given absolute error is
more surprising at low counts than at high counts. That is precisely the right
behavior for photon-limited microscopy, so the algorithm weights low-intensity
regions appropriately instead of being swamped by bright ones.

Two things to keep in mind about $J$:

- It is **convex** in $f$ but **not differentiable at the constraint boundary**
  (the non-negativity we impose below), so the minimization is done by an
  iterative scheme with a non-negativity constraint, not a single linear solve.
- It is **not scale-invariant**. The PSF must carry the correct total light (a
  probability distribution), so the algorithm first normalizes it.

## What the algorithm does, end to end

The inputs are the data $g$, the PSF (as its Fourier transform / optical
transfer function $H$), and a few control parameters. The output is the estimate
$f$ of the object, on a zero background. The run is a fixed number of
iterations; each iteration is one **data / prior / sieve** cycle described below.

### Step 0 — Prepare the forward model

Before iterating, the algorithm sets up the ingredients that make the forward
model $m = Hf + b$ physically correct:

1. **Normalize the PSF to unit total light.** The OTF is rescaled so that its DC
   term — the sum of the PSF — equals 1. Then convolution conserves total flux:
   blurring an object does not create or destroy light. This is essential for the
   count-based model (and is why, for instance, a PSF loaded with an arbitrary
   brightness scale is fine: the algorithm fixes the scale itself).
2. **Initialize the object to the data.** $f$ starts as a copy of $g$, with any
   negative pixels set to zero. (Negative counts are unphysical.)
3. **Estimate the background level $b$.** If the user did not supply it, it is
   taken as the mean of the **dimmest decile** of pixels — the bottom 10 % of the
   intensity histogram, which are overwhelmingly background.
4. **Estimate the PSF size.** The PSF is inverse-Fourier-transformed to recover
   the spatial point-spread, and its standard deviations are computed along each
   axis (the width and depth of the blur). These widths later set how much
   smoothing the regularizers use, so the smoothing matches the physics of the
   optical blur rather than an arbitrary scale.
5. **Set the regularization strength from the SNR.** A single dial, the effective
   signal-to-noise ratio, sets how strongly the algorithm trusts the data versus
   preferring a smooth object. **High SNR → trust the data, stay close to the
   pure MLE.** Low SNR → regularize harder. The shrinkage strength $\lambda$
   defaults to $1/\mathrm{SNR}$.

### Step 1 — The data step (fit the image, respecting the noise model)

This is the workhorse: move $f$ so that the predicted image better explains the
measured image, using the Poisson likelihood.

- **Form the predicted image.** Compute $m = Hf + b$: blur the current object and
  add the background.
- **Compare predicted to measured, and push back.** Compute the ratio $g/m$
  (measured ÷ predicted) and back-project it through the *transposed* blur
  $H^{\mathsf T}$, giving $R = H^{\mathsf T}(g/m)$. Wherever the model
  under-predicted the data ($g>m$, so $g/m>1$), $R$ is large and the object will
  be pushed *up*; wherever it over-predicted, it is pushed *down*. Intuitively the
  algorithm redistributes light toward regions that demand it.
- **Choose a direction to move.** From this ratio the algorithm forms a search
  direction $d = f(R-1)$. (This is the additive analogue of the classic
  Richardson–Lucy update.)
- **Pick the step size optimally.** Rather than taking a fixed-size step, the
  algorithm searches *along the line* $f + \alpha d$ for the step size $\alpha$
  that makes the likelihood $J$ as small as possible. This one-dimensional
  function is strictly convex, so it has a single minimum, which the algorithm
  finds with a Newton method (using the function's value and its first two
  derivatives). If that ever fails, it falls back to the safe unit step. The net
  effect is: each iteration takes the largest well-justified step the data allow,
  which makes convergence fast and robust.
- **Take the step and keep the object physical.** Update $f \leftarrow f +
  \alpha d$ (with a small safety damping if the objective unexpectedly rose) and
  clamp any negative pixels back to zero. The object stays non-negative, as a
  real image must.

### Step 2 — The prior step (prefer smooth, plausible objects)

A pure likelihood fit will happily chase the noise, especially at low SNR, and
can develop spiky or ringing artifacts. The prior tames this by gently shrinking
the object, with the strength controlled by $\lambda = 1/\mathrm{SNR}$:

- **Shrink each pixel concavely.** The map $f \mapsto (\sqrt{2\lambda f + 1}-1)/\lambda$
  pulls values toward zero, but *less* for large values and *more* for small
  ones (a concave, sub-linear shrinkage). Small, noise-level fluctuations are
  suppressed; strong, real signals are preserved. This is a soft version of the
  "keep only what is confidently above the noise" idea.
- **Conserve the total light.** Shrinking would otherwise remove photons. The
  object is then rescaled so that its **total integrated intensity is exactly the
  same as before the shrinkage**. The prior can reshape the object (smooth it)
  but cannot change how much light it contains — matching the physical fact that
  the camera already recorded a fixed total photon budget.

At high SNR, $\lambda$ is small, so this step barely touches the image. At low
SNR it is the main force keeping the reconstruction from going wild.

### Step 3 — The sieve (spatial smoothing that fades with progress)

The prior shrinks point by point (independent of neighbors). The **sieve**
regularizes *spatially*: it applies a gentle, low-pass smoothing so that the
object does not oscillate from voxel to voxel.

- **A Gaussian blur matched to the optics.** In the frequency domain this is
  multiplication by a Gaussian low-pass filter whose width is set by the PSF
  widths estimated in Step 0. Smoothing at the scale of the actual optical blur —
  not finer, not coarser — is a natural choice: it removes detail the optics
  could not have produced in the first place.
- **Annealing: smooth early, refine later.** The strength of the smoothing
  "cools" over the iterations (it is proportional to $1 - k/k_{\text{last}}$).
  In the first iterations the object is rough and far from the solution, so
  stronger smoothing stabilizes it. As it approaches the final estimate the
  smoothing fades to zero, so the last iterations are a near-pure likelihood fit
  that recovers fine detail. This is the classic *simulated-annealing / sieve*
  strategy for deconvolution.
- **Repeat the pass a few times.** By default the smoothing is applied a few
  times per iteration (`poissonSievePasses`), a cheap way to deepen the
  low-pass without changing its frequency content.

### Step 4 — After all iterations: remove the residual floor

The forward model $m = Hf + b$ cannot fully separate a **flat** offset added to
the object from the background $b$ (blurring a constant leaves the same
constant, so the two are degenerate). As a result, the MLE settles the object's
own background at a small constant offset above zero rather than exactly at zero.
Because this leftover is uniform, the final step simply measures it (again, the
mean of the dimmest decile of the result) and **subtracts it**, then clamps at
zero. The output object is thus left sitting on a clean zero background.

## The loop at a glance

Each of the `iterations` runs:

1. **Data step** — blur $f$ to $m=Hf+b$, compare with $g$, back-project, and take
   an optimally-sized likelihood step; keep $f\ge 0$.
2. **Prior step** — concave shrinkage scaled by $1/\mathrm{SNR}$, then
   rescale to conserve total light.
3. **Sieve** — anisotropic Gaussian low-pass, annealed down to zero across the
   run.

After the loop, a uniform residual floor is subtracted so the object rests on a
zero background.

## Parameter → physics

| Parameter | Default | What it controls, physically |
|-----------|---------|------------------------------|
| `poissonSnr` | 10 | The effective signal-to-noise ratio / sharpness dial. High → trust the data, little regularization, closer to the pure MLE. Low → stronger shrinkage + smoothing. |
| `poissonBackground` | −1 (auto) | The flat background level $b$ in the forward model. If unset (−1), it is estimated from the dimmest decile of the data. |
| `poissonLambda` | 0 (auto) | The shrinkage strength $\lambda$ of the prior. If unset (0), it is derived as $1/\mathrm{SNR}$. |
| `poissonSieve` | true | Whether the annealed anisotropic Gaussian smoothing (roughness regularizer) is applied. |
| `poissonSievePasses` | 2 | How many smoothing passes per iteration (deeper low-pass, same bandwidth). |
| `iterations` | 10 | Number of data/prior/sieve cycles. Stopping is by count, not by a quality test. |

## Summary

The algorithm is a **regularized maximum-likelihood fit of a Poisson (count)
imaging model**. It minimizes the I-divergence between the measured image and
the predicted image $Hf+b$ under a non-negativity constraint, using an exact
convex line search for fast, stable progress. Two regularizers — a flux-conserving
shrinkage prior (controls small-scale noise) and an annealed anisotropic Gaussian
sieve (controls spatial roughness) — keep the solution physical, with their
strengths tuned by a single SNR dial. The result is an estimate of the object on
a zero background, robust across a wide range of signal levels.

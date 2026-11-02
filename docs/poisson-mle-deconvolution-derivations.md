# Poisson MLE Deconvolution — Derivations & Explanations

A mathematical companion to [`poisson-mle-deconvolution.md`](poisson-mle-deconvolution.md).
It derives the objective, the gradient, the convex line search, the shrinkage
prior, and the sieve OTF, and explains the non-obvious choices (unit-sum PSF,
floor clamping, flux conservation, final floor subtraction).

Notation.

- $x,y,z$ label the $N= W H D$ voxels; $i$ runs over all of them.
- $g$ measured image; $f$ unknown object (above background); $b$ background.
- $H$ the linear forward operator (circular convolution by the PSF); $\mathrm{PSF}\ast$ its spatial form.
- $m = Hf + b$ the predicted (model) image.
- All sums are over the full volume.
- $\ln$ is the natural log.

---

## 1. From physics to the forward model

A widefield fluorophore sheet of brightness $f$ is imaged through a linear,
shift-invariant optical system with impulse response $\mathrm{PSF}$. Ignoring
detector effects, the ideal (noiseless) intensity at each pixel is the
convolution

$$
(Hf)_i \;=\; \sum_{j} \mathrm{PSF}_{i-j}\, f_j .
$$

The PSF is a **probability distribution over arrival positions**: a single
point-source emits a fixed number of photons, and $\mathrm{PSF}$ gives where they
land. Hence it is non-negative and has **total mass (sum) 1**. Convolution by a
unit-mass kernel then **conserves total flux**:

$$
\sum_i (Hf)_i \;=\; \Bigl(\sum_i \mathrm{PSF}_i\Bigr)\Bigl(\sum_j f_j\Bigr)
\;=\; \sum_j f_j ,
$$

and, crucially, **preserves constants**: $H(\mathbf c) = c\,\mathbf c$ (a uniform
object blurs to the same uniform object). Both facts are used later.

A near-uniform background $b$ (auto-fluorescence, stray light, dark current) is
additive and essentially flat across the field, so the forward model is

$$
\boxed{\; m_i \;=\; (Hf)_i + b \;}
$$

with the object constrained to be **non-negative**, $f_i \ge 0$ (it is the object
*above* the background, so its natural floor is zero).

### 1.1 Why the PSF must be normalized to unit sum

The data model in §2 assumes $m$ is a **mean photon count**, so $Hf$ must be a
flux-conserving blur. If the supplied PSF has sum $S \neq 1$, the predicted image
$Hf$ is $S$ times too large for any given $f$, and the fit would compensate by
shrinking the recovered object by a factor $1/S$ — a systematic brightness error.
The algorithm therefore rescales the OTF so its **DC term** (the value at zero
frequency, equal to the spatial sum of the PSF) is exactly 1:

$$
H \;\leftarrow\; \frac{H}{\mathrm{DC}(H)} .
$$

This is why the *Richardson–Lucy* algorithm (which uses a ratio $f\,H^{\mathsf
T}(g/Hf)$) is **scale-invariant** and needs no such step, while the
I-divergence/Poisson fit is **not**: the ratio cancels the common scale, the
likelihood does not. A PSF loaded with an arbitrary brightness is thus harmless.

---

## 2. The Poisson likelihood and the I-divergence objective

Shot noise makes each pixel count a Poisson random variable with mean $m_i$:

$$
P(g_i\mid m_i) \;=\; \frac{m_i^{\,g_i}\, e^{-m_i}}{g_i!} .
$$

The counts are independent across pixels, so the joint likelihood is the product.
The negative log-likelihood is

$$
-\ln P(g\mid m)
\;=\; \sum_i \Bigl[ -\,g_i \ln m_i \;+\; m_i \;+\; \ln(g_i!) \Bigr].
$$

The term $\ln(g_i!)$ depends only on the data, not on $f$, so it is dropped. The
quantity to minimize over $f$ (equivalently over $m = Hf+b$) is

$$
\sum_i \bigl[ -\,g_i \ln m_i + m_i \bigr] .
$$

Re-arranging into the canonical **I-divergence** (Poisson NLL up to a constant):

$$
\boxed{\;
J(f) \;=\; \sum_i \Bigl[ g_i \ln\!\tfrac{g_i}{m_i} - g_i + m_i \Bigr]
\;}
\qquad\text{with } m_i=(Hf)_i+b .
$$

Indeed $J = \sum_i[g_i\ln g_i - g_i] + \sum_i[-\,g_i\ln m_i + m_i]$: the second
sum is exactly the NLL above, and the first is independent of $f$. So minimizing
$J$ is identical to maximizing the likelihood. $J$ is the sum of a
pointwise-convex function of $m_i$, and $m_i$ is affine in $f$,
so **$J$ is convex in $f$**. (It is not strictly convex in general because $H$
has a null space — uniform offsets in the direction $H$ kills — which is exactly
the degeneracy removed in §8.)

**Why I-divergence and not least squares.** A Gaussian (shot-noise-negligible)
model would minimize $\sum_i (g_i-m_i)^2$, penalizing an absolute error equally
everywhere. The Poisson model penalizes via $I(g_i,m_i)\propto
m_i-g_i- g_i\ln(m_i/g_i)$, whose curvature $\partial^2 I/\partial m^2 = g_i/m_i^2$
is **large at low counts and small at high counts**. So a discrepancy in a dark
region is weighted heavily (it is surprising), while the same absolute discrepancy
in a bright region is barely penalized (it is unremarkable). This is the correct
photon-statistics weighting and is the reason to prefer the Poisson objective for
fluorescence images.

---

## 3. Gradient and the data step

Write $m=Hf+b$. The gradient of $J$ with respect to $f$ comes through $m$:

$$
\frac{\partial J}{\partial f}
\;=\; H^{\mathsf T}\!\Bigl(\tfrac{\partial J}{\partial m}\Bigr),
\qquad
\frac{\partial J}{\partial m_i}
\;=\; -\,g_i\,\frac{1}{m_i} + 1
\;=\; 1 - \tfrac{g_i}{m_i}.
$$

Hence

$$
\nabla_f J \;=\; H^{\mathsf T}\!\Bigl(1 - \tfrac{g}{m}\Bigr).
$$

**The back-projected ratio.** Define the "ratio of model to data" pushed through
the transposed blur

$$
R \;\triangleq\; H^{\mathsf T}\!\Bigl(\tfrac{g}{m}\Bigr).
$$

Then $\nabla_f J = H^{\mathsf T}(\mathbf 1) - R$. Because $H$ is unit-mass
convolution, $H^{\mathsf T}(\mathbf 1)=\mathbf 1$ (each row of the adjoint sums
to the PSF mass $=1$), so the gradient is simply

$$
\nabla_f J \;=\; \mathbf 1 - R
\qquad\text{(elementwise).}
$$

**The descent direction.** The algorithm does not use the raw gradient; it uses
the **multiplicative (Richardson–Lucy) descent direction**

$$
d \;=\; f\,(R - 1) \;=\; -\,f\odot\bigl(\mathbf 1 - R\bigr)
\;=\; -\,\operatorname{diag}(f)\,\nabla_f J ,
$$

i.e. the negative gradient preconditioned elementwise by $\operatorname{diag}(f)$.
This is a valid descent direction since

$$
\nabla_f J\cdot d \;=\; -\sum_i f_i\,(1-R_i)^2 \;\le\; 0
$$

with strict inequality wherever $f_i(1-R_i)\ne0$; it also encodes the
non-negativity *structure* of RL (a small $f_i$ moves little, so zero voxels stay
zero to first order). The algorithm then optimizes the step size *along* $d$
(§4) rather than fixing it to $1$.

**Connection to Richardson–Lucy.** The classic RL multiplicative update is
$f \leftarrow f\,R = f\,H^{\mathsf T}(g/Hf)$. A unit step along this direction is
exactly $f + f(R-1) = fR$: so RL is the special case $\alpha=1$ of the DOLPHIN
update. The algorithm keeps the same direction $d=f(R-1)$ but treats it as a
**line of descent** and picks $\alpha$ optimally (§4) instead of fixing it to $1$.

**The physical floor clamp.** With $f\ge0$ and a non-negative, unit-mass PSF, the
model satisfies $m_i\ge b_i = b$ exactly. In floating point, however, the sieve's
FFT ringing and small negative lobes of a *measured* PSF can push $m$ slightly
below $b$ at isolated voxels. Left unchecked, the ratio $g/m$ would be divided by
a value floored at a tiny $\varepsilon$, amplifying $g$ by $\sim 1/\varepsilon$
and driving the iteration to `inf`/`NaN`. The algorithm therefore clamps

$$
m_i \;\leftarrow\; \max(m_i,\, b)
$$

so that $g_i/m_i \le g_{\max}/b$ and the step stays bounded. This is a numerical
safety measure, not a modeling assumption.

---

## 4. The exact, safeguarded convex line search

We move along the ray

$$
f(\alpha) \;=\; f + \alpha\, d,
\qquad
m(\alpha) \;=\; m + \alpha\,(Hd) \;\triangleq\; m + \alpha\, h,
$$

where $h \triangleq H d$ is the forward model of the direction (computed once,
like $m$). Substituting $m(\alpha)$ into $J$ gives a **one-variable function**

$$
J(\alpha) \;=\; \sum_i \Bigl[ g_i \ln\tfrac{g_i}{m_i + \alpha h_i} - g_i + m_i + \alpha h_i \Bigr].
$$

Its first two derivatives with respect to $\alpha$ are

$$
\boxed{\;
\frac{dJ}{d\alpha}
\;=\; \sum_i h_i\Bigl(1 - \tfrac{g_i}{m_i + \alpha h_i}\Bigr)
}
\qquad
\boxed{\;
\frac{d^2J}{d\alpha^2}
\;=\; \sum_i \tfrac{g_i\, h_i^2}{(m_i + \alpha h_i)^2}
\;}.
$$

Note $d^2J/d\alpha^2 > 0$ whenever any $g_i h_i \ne 0$ and the denominator is
positive (enforced by the floor), so **$J(\alpha)$ is strictly convex** in
$\alpha$ and has a **unique** minimizer $\alpha^\star$ satisfying
$dJ/d\alpha=0$.

**Solving for $\alpha^\star$: safeguarded Newton.** Newton's method
$\alpha \leftarrow \alpha - \frac{dJ/d\alpha}{d^2J/d\alpha^2}$ converges
quadratically near the root but can overshoot a 1-D convex function; the
algorithm clamps each iterate into a bracket $[\alpha_{\min},\alpha_{\max}]$
(here $[0.005, 25]$) and **falls back to bisection** whenever the Newton step
leaves the bracket or the curvature is numerically zero. This is the standard
robust combination: Newton's speed where it is safe, bisection's guarantee
elsewhere. It is capped at a small number of iterations (30) and a relative
tolerance.

**Guard against overshoot in practice.** Even with a robust 1-D minimizer, the
direction $d$ is only a *descent* direction for the true multi-dimensional $J$,
not a guarantee that the full step decreases $J$. The algorithm compares the
objective after the step against the current one: if $J$ **rose**, it halves a
damping factor $\text{pwr}$ and proceeds with a smaller update
$f\leftarrow f + \alpha\,\text{pwr}\,d$. Separately, it always compares the
Newton-optimum step against the **RL unit step $\alpha=1$** and keeps whichever
gives the smaller $J$ — the unit step is the classically *provably* safe RL
descent, so the algorithm can never do worse than classical RL.

The net step is the **additive**

$$
f \;\leftarrow\; f + \alpha\,\text{pwr}\,d,
\qquad d = f(R-1),
$$

followed by the **non-negativity projection** $f\leftarrow \max(f,0)$. The
projection is the mechanism that enforces the constraint $f\ge 0$ (a
projected/alternating step). It is the difference from RL's multiplicative form,
which enforces non-negativity automatically because $f\,R$ keeps the sign of $f$.

---

## 5. The prior: concave shrinkage with flux conservation

A pure likelihood fit (especially at low SNR) overfits shot noise. The prior
regularizes by shrinking $f$ toward zero with a strength $\lambda$ that is set
from the SNR:

$$
\lambda \;=\; \frac{1}{\mathrm{SNR}}
\qquad\bigl(\text{or a user override; } \lambda=0 \text{ disables the prior}\bigr).
$$

The shrinkage map is

$$
\boxed{\;
\phi_\lambda(u) \;=\; \frac{\sqrt{2\,\lambda\,u + 1}\; - 1}{\lambda}
\;}
\qquad\text{applied as } f \leftarrow \phi_\lambda(f),
$$

with the square-root argument floored at zero so it is well-defined for $u\ge0$.
(If $\lambda$ is numerically zero it is replaced by a small floor to avoid
division by zero, which in the $\lambda\to0$ limit gives $\phi_\lambda(u)\to u$,
i.e. no shrinkage — consistent with "high SNR = pure MLE".)

**What kind of regularizer is $\phi$?** Expand for small $u$ ($u \ll 1/\lambda$):
$\sqrt{1+2\lambda u} \approx 1 + \lambda u - \tfrac12 \lambda^2 u^2 + \dots$, so
$\phi_\lambda(u)\approx u - \tfrac12\lambda u^2 + \dots$ — the identity to
first order, with a **quadratic pull toward zero** that is *weak* near $u=0$
relative to large $u$. For large $u$ ($u\gg 1/\lambda$), $\phi_\lambda(u)\approx
\sqrt{2u/\lambda} \ll u$, a strong sub-linear damping. So $\phi$ **compresses
large values sub-linearly and leaves small values nearly intact in relative
terms**, but because it is *concave* ($\phi''<0$) it reduces the **range** of
$[0,f_{\max}]$: a concave map that is near-linear at the top and flattens the
slope at the bottom **attenuates high-frequency detail / spikes** (which require
both very large and very small adjacent values) more than low-frequency structure.
It is a soft, nonlinear "roughness" penalty — a smooth surrogate for
$\ell_1$-type sparsifying shrinkage, but one that never hard-zeroes values and
preserves positivity.

**Flux conservation.** $\phi_\lambda$ strictly reduces the total sum
$\sum_i f_i$ (for $f\ne 0$). But the physical object already contains a **fixed
total photon budget** — the camera recorded a definite integrated intensity, and
there is no physical reason the reconstruction should contain *less* light than
the data. The prior is therefore meant to *reshape* the object (smooth it), not
*drain* it. So after shrinking, the object is **rescaled back to its original
total flux**:

$$
f \;\leftarrow\; \frac{\sum_{\text{old}} f_i}{\sum_{\text{new}} f_i}\; f ,
\qquad
\text{so that } \sum f_{\text{after}} = \sum f_{\text{before}} .
$$

The guard $\sum_{\text{old}},\sum_{\text{new}} > 0$ avoids division by zero on an
all-dark object. The upshot: the prior is a **zero-sum, total-flux-preserving
smoother** — it can move light around and damp spikes, but it cannot create or
destroy the total light, matching the unit-mass, flux-conserving forward model of
§1.

---

## 6. The sieve: annealed anisotropic Gaussian low-pass

The prior acts *pointwise* (independent of neighbors). The **sieve** adds an
explicit *spatial* roughness penalty by low-pass filtering, in the frequency
domain.

**The OTF.** A separable anisotropic 3-D Gaussian with spatial standard
deviations $(\sigma_x,\sigma_y,\sigma_z)$ in voxels has Fourier transform

$$
\mathrm{OTF}(\mathbf k)
\;=\; \exp\!\Bigl[-2\pi^2\bigl(\sigma_x^2 f_x^2 + \sigma_y^2 f_y^2 + \sigma_z^2 f_z^2\bigr)\Bigr],
$$

where the frequencies $f_x,f_y,f_z$ are in **cycles per sample**, obtained by
dividing the wrapped bin index by the size along that axis:
$f_x = k_x/W$ with $k_x$ wrapped to $[-W/2, W/2]$. (The factor is $2\pi^2\sigma^2
f^2$ because $f$ is in cycles/sample; using the angular-frequency form
$2(2\pi)^2\sigma^2 f^2$ would be 4× too large.) Applying the sieve to $f$ means
$\widehat f \leftarrow \mathrm{OTF}\cdot \widehat f$ followed by an inverse FFT —
a blur in the spatial domain.

**Anisotropy from the optics.** The $\sigma$'s are *not* free; they are set from
the PSF widths estimated in Step 0 (§0/prepare). Specifically the lateral
standard deviations $\sigma_x,\sigma_y$ and axial $\sigma_z$ are scaled by the
PSF's lateral half-width $\tfrac12(\mathrm{sig}_x+\mathrm{sig}_y)$ and axial
width $\mathrm{sig}_z$ respectively (each multiplied by $\tfrac12$ as a base
fraction). Low-passing at the *optical* scale removes detail the PSF could not
have resolved in the first place — a physically-motivated band-limit, not an
arbitrary one. The lateral/axial split captures the usual
axially-elongated (spheroidal) optical PSF.

**Annealing.** The smoothing strength is multiplied by a cooling schedule

$$
t(k) \;=\; \max\bigl(0,\; 1 - k/k_{\text{last}}\bigr),
\qquad k_{\text{last}} = \max\bigl(1,\;0.8\,K\bigr),
$$

with $K$ the total iteration count and $k$ the current (0-based) iteration. So
the effective sigmas are $\sigma(k)=\tfrac12\,\mathrm{psf}\,\cdot t(k)$: they are
largest (up to $\tfrac12$ the PSF width) in the early iterations and **decay
linearly to zero** by $k\approx 0.8K$. This is the *sieve* (a.k.a. annealed)
strategy: heavy smoothing while the estimate is rough and far from the solution
(stabilizes, prevents noise chasing), tapering to the **identity** in the final
$\sim 20\%$ of iterations so the last steps are a near-pure likelihood fit that
recovers fine detail. Because a near-zero sigma makes $\mathrm{OTF}\approx 1$,
the "sieve off" end of the schedule is exactly the identity — no special case.

**Multiple passes.** Applying the same low-pass $p$ times (`poissonSievePasses`)
is equivalent to one low-pass with sigma scaled by $\sqrt{p}$; it is used as a
cheap way to deepen the attenuation within the low band without broadening the
passband, giving a steeper roll-off per iteration.

The sieve and the prior are complementary: the prior is a **nonlinear,
flux-conserving, pointwise** regularizer; the sieve is a **linear, band-limited,
neighbor-dependent** regularizer. Together they suppress both isolated spikes
(prior) and spatial oscillation (sieve).

---

## 7. Background estimation

If $b$ is not supplied, it is estimated as the **mean of the dimmest decile**:
sort all pixel values and average the lowest 10 % (clamped to at least 1 pixel).
The reasoning: in a sparse-fluorescence image, the bottom of the histogram is
overwhelmingly background (no real signal sits there), so its mean is a robust
estimate of $b$ that is insensitive to the bright tail (fluorophores) — unlike a
global mean, which is dragged up by the signal. The same dimmest-decile statistic
is reused in §8 to measure the residual floor.

---

## 8. Final floor subtraction (the constant-offset degeneracy)

Recall §1.1: $H$ **preserves constants**, $H(\mathbf c)=c\,\mathbf c$. Therefore
the forward model is blind to a *uniform offset* in $f$: adding a constant $c$ to
$f$ simply shifts every $(Hf)_i$ by $c$, which is indistinguishable from adding
$c$ to the background. The MLE splits this ambiguity between the object and the
background rather than resolving it, so the converged object's background settles
at a small **flat residual offset** (the dimmest-decile estimate of $b$ is itself
slightly below the true background mean, so a mismatch leaks into $f$ as a
constant floor).

Because this leftover is **uniform**, it can be removed exactly: measure the
result's background again as the mean of its dimmest decile, and subtract it
globally,

$$
f \;\leftarrow\; \max\bigl(0,\; f - \text{floor}\bigr),
$$

leaving the object on a **clean zero background** without touching its shape
(a uniform subtraction commutes with the object's structure). This is the
final output.

---

## 9. Stopping and the SNR dial

The run is a **fixed number of iterations** $K$ (`iterations`); there is no
quality-factor / convergence test. The behavior across $K$ is shaped by the
sieve annealing schedule (§6) rather than by a stopping criterion.

**`poissonSnr` is the master dial.** It sets $\lambda = 1/\mathrm{SNR}$ and hence
the strength of *both* regularizers (the prior directly; the sieve implicitly,
since the PSF widths it uses are data-derived). The limits:

- **SNR $\to\infty$**: $\lambda\to0$ (prior off), and the scheme reduces toward
  the **pure maximum-likelihood fit** of the Poisson model (a guarded, exact-line-
  search, non-negatively-constrained I-divergence minimizer).
- **SNR small**: $\lambda$ large → strong shrinkage + strong early smoothing,
  biasing the estimate toward smoothness in exchange for stability — the usual
  bias/variance tradeoff of a regularized inverse problem.

The other knobs are local overrides: `poissonBackground` fixes $b$ instead of the
dimmest-decile estimate; `poissonLambda` fixes $\lambda$ instead of $1/\mathrm{SNR}$;
`poissonSieve`/`poissonSievePasses` toggle/deepen the spatial smoother.

---

## 10. Symbol summary

| Symbol | Meaning |
|--------|---------|
| $g$ | measured image (data) |
| $f$ | unknown object, above background, constrained $f\ge0$ |
| $b$ | flat background level |
| $H$ | forward operator (circular convolution by unit-mass PSF); preserves constants and total flux |
| $m=Hf+b$ | predicted (model) image |
| $J(f)=\sum_i[g_i\ln(g_i/m_i)-g_i+m_i]$ | Poisson negative log-likelihood / I-divergence (convex in $f$) |
| $R=H^{\mathsf T}(g/m)$ | back-projected model-to-data ratio |
| $d=f(R-1)$ | additive search direction (RL form) |
| $h=Hd$ | forward model of the direction |
| $J(\alpha)=\sum_i[g_i\ln(g_i/(m_i+\alpha h_i))-g_i+m_i+\alpha h_i]$ | objective along the ray $f+\alpha d$ (strictly convex) |
| $\alpha^\star$ | optimal step, found by safeguarded Newton (bisection fallback), guarded against RL unit step |
| $\text{pwr}$ | damping factor, halved if a step raises $J$ |
| $\lambda=1/\mathrm{SNR}$ | prior shrinkage strength |
| $\phi_\lambda(u)=(\sqrt{2\lambda u+1}-1)/\lambda$ | concave, sub-linear shrinkage map (flux-preserving after rescale) |
| $\mathrm{OTF}(\mathbf k)=\exp[-2\pi^2(\sigma_x^2 f_x^2+\sigma_y^2 f_y^2+\sigma_z^2 f_z^2)]$ | sieve Gaussian low-pass |
| $\sigma(k)=\tfrac12\,\mathrm{psf}\,(1-k/k_{\text{last}})$ | annealed sieve width (→ 0 at the end) |
| $\text{floor}$ | dimmest-decile mean of the result; subtracted at the end |

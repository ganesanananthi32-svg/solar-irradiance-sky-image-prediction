# Short-Term Solar Irradiance Nowcasting From All-Sky Imagery With a Differentiable Clear-Sky Model and Sun-Centric Polar Attention

**Abstract**—High-penetration integration of solar photovoltaic (PV) power into modern electricity grids requires ultra-short-term (0–30 minute) solar irradiance nowcasting to mitigate sudden frequency deviations and costly reserve spinning adjustments triggered by moving cloud fields. While deep learning methods have increasingly integrated physical Clear-Sky Models (CSM) via residual learning ($\Delta GHI = GHI - GHI_{cs}$), existing approaches treat the CSM as an offline, non-differentiable preprocessor and apply uniform 2D Cartesian spatial pooling over $180^\circ$ hemispherical sky images. This formulation dilutes the direct solar beam (which occupies $<5\%$ of camera pixels yet accounts for over $80\%$ of clear Global Horizontal Irradiance) and causes significant ramp-smoothing under standard Mean Squared Error (MSE) objectives. 

In this work, we propose **Diff-CSM Polar-Net**, an end-to-end physics-informed nowcasting architecture featuring three key contributions: (1) a **Differentiable Clear-Sky Model (Diff-CSM)** where atmospheric Linke Turbidity and optical transmission coefficients receive end-to-end gradient updates directly from irradiance losses; (2) an astronomical **Sun-Centric Log-Polar Coordinate Transformation** that dynamically maps fisheye camera pixels into a sun-centered $(r, \theta)$ domain, simplifying cloud advection towards the direct beam into a 1D radial translation along $r \to 0$; and (3) an **Attenuation-Lensing Disentanglement Head** coupled with **Inductive Conformal Prediction (ICP)** to provide distribution-free $90\%$ guaranteed prediction intervals. 

Evaluated across 3,079 real daytime observations on multi-year ground radiometric and all-sky camera datasets during highly intermittent cloud dynamics, our framework achieves a Root Mean Squared Error (RMSE) of **$150.98\text{ W/m}^2$** and Mean Absolute Error (MAE) of **$104.79\text{ W/m}^2$**, representing a **$+19.96\%$ Forecast Skill Score ($SS_{RMSE}$)** over the mandatory Smart Persistence baseline and outperforming the benchmark Gao & Liu (WACV 2022) model by more than threefold in predictive skill gain.

**Index Terms**—Solar Irradiance Forecasting, All-Sky Imagers, Physics-Informed Neural Networks, Differentiable Radiative Transfer, Conformal Prediction, Smart Grid Integration.

---

## I. INTRODUCTION

THE escalating decarbonization of global energy systems has driven unprecedented growth in distributed and utility-scale photovoltaic (PV) generation. However, solar power is governed by the volatile interaction between top-of-the-atmosphere extraterrestrial radiation, earth-sun celestial mechanics, and stochastic atmospheric cloud dynamics [1], [2]. Rapid drops in Global Horizontal Irradiance (GHI)—frequently exceeding ramp rates of $100\text{ W/m}^2/\text{minute}$—destabilize distribution feeders, trigger voltage flickers, and force utility operators to dispatch expensive, carbon-intensive fast-ramping spinning reserves [3].

Accurate intra-hour solar nowcasting (forecast horizons $\Delta t \in [1, 30]\text{ minutes}$) provides grid operators with look-ahead visibility to schedule battery energy storage systems (BESS) and maintain frequency regulation [4]. Ground-based All-Sky Imagers (ASI) equipped with fish-eye lenses capture hemispherical cloud spatial distributions at high temporal (sub-minute) and spatial resolution, capturing approaching cloud boundaries long before they are registered by geostationary meteorological satellites [5].

### A. Limitations of Existing Work
Despite progress in deep computer vision architectures (CNNs, LSTMs, and Vision Transformers), the prevailing state-of-the-art exhibits three fundamental bottlenecks:
1. **Offline, Non-Differentiable Clear-Sky Preprocessing:** Landmark works—most notably Gao & Liu (*IEEE/CVF WACV 2022*) [6]—demonstrated the power of decomposing irradiance into a deterministic astronomical baseline and a residual cloud perturbation ($\Delta GHI = GHI - GHI_{cs}$). However, all prior formulations compute $GHI_{cs}$ strictly as a static, external preprocessing step. Atmospheric parameters, such as Linke Turbidity ($T_L$) and Aerosol Optical Depth (AOD), are held rigid. On humid, smoky, or hazy days, uncalibrated clear-sky baselines drift by $50\text{--}120\text{ W/m}^2$, compelling the downstream neural network to compensate for systematic optical biases rather than focusing on cloud occlusion [7].
2. **Cartesian Spatial Dilution and Direct Beam Neglect:** In an all-sky imager, the direct solar beam (governed by Direct Normal Irradiance, DNI) accounts for $75\text{--}85\%$ of total clear-sky radiation, yet occupies a circumsolar cone ($\pm 15^\circ$) representing less than $5\%$ of total pixels [8]. Standard 2D Cartesian convolutional filters and self-attention tokens pool uniformly across the entire sky dome. Consequently, localized cloud boundaries approaching the sun disk are diluted by expansive regions of irrelevant clear blue sky, delaying ramp detection until after the sun is already occluded.
3. **Severe Ramp Smoothing and Lack of Rigorous Uncertainty:** Conventional point regression optimizing Mean Squared Error (MSE) suffers from the "regression-to-the-mean" phenomenon. During broken-cloud conditions, MSE-trained models predict smoothed average trajectories, failing to alert grid operators to sharp ramp events. Furthermore, point predictions provide no formal confidence bounds for spinning reserve sizing [9].

### B. Summary of Contributions
To address these limitations, we introduce **Diff-CSM Polar-Net**, an advanced physics-guided solar nowcasting framework. Our core contributions are:
1. **Differentiable Clear-Sky Layer (`Diff-CSM`):** We re-formulate the Ineichen-Perez atmospheric radiative transfer model as a fully differentiable PyTorch/Autograd module. Optical transmission parameters ($T_L$, AOD, and water vapor absorption) receive analytical gradient updates directly from ground radiometer error signals.
2. **Sun-Centric Log-Polar Coordinate Transformation:** We introduce an astronomical unwarping transformation that re-projects the fish-eye camera image into polar coordinates $(r, \theta)$ centered dynamically at the celestial sun coordinates $(x_{sun}, y_{sun})$. This reduces complex 2D curvilinear cloud advection into a 1D linear translation along the radial axis ($r \to 0$).
3. **Physical Disentanglement & Inductive Conformal Prediction:** We decouple irradiance predictions into direct beam transmission, diffuse sky radiance, and circumsolar forward-scattering enhancement ("cloud-edge lensing"). We equip the model with Inductive Conformal Prediction (ICP), yielding distribution-free $90\%$ guaranteed coverage intervals.
4. **Empirical Validation on Multi-Year Real Datasets:** Benchmarking across 3,079 real daytime observations from the Folsom and Stanford SKIPP'D stations demonstrates an RMSE of **$150.98\text{ W/m}^2$** and a Forecast Skill Score ($SS_{RMSE}$) of **$+19.96\%$** over Smart Persistence, substantially outperforming prior residual baselines.

---

## II. THEORETICAL FORMULATION & METHODOLOGY

### A. Astronomical Solar Geometry and Fish-Eye Coordinate Mapping
Given a geographic station with latitude $\phi$, longitude $\lambda$, elevation $h$, and UTC timestamp $t$, the solar position is determined by celestial ephemeris. The solar declination $\delta$ and the equation of time ($EoT$) are derived using the Spencer formulations:
$$\Gamma = \frac{2\pi}{365} (d_n - 1 + \frac{t_{hour} - 12}{24})$$
where $d_n \in [1, 366]$ is the day of the year. The local hour angle $\omega$ is calculated from True Solar Time ($TST$):
$$TST = t_{hour} \cdot 60 + EoT + 4\lambda$$
$$\omega = \frac{TST}{4} - 180^\circ$$
The solar zenith angle $\theta_z$ and solar azimuth $\gamma_s$ satisfy:
$$\cos \theta_z = \sin \phi \sin \delta + \cos \phi \cos \delta \cos \omega$$
$$\cos \gamma_s = \frac{\sin \delta \cos \phi - \cos \delta \sin \phi \cos \omega}{\sin \theta_z}$$

For an equidistant all-sky camera lens with optical center $(c_x, c_y)$ and calibration radius $R_{max}$, celestial coordinates map to image plane sensor coordinates $(x_{sun}, y_{sun})$ via:
$$r_{sun} = R_{max} \cdot \left(\frac{\theta_z}{90^\circ}\right)$$
$$x_{sun} = c_x + r_{sun} \sin(\gamma_s - \gamma_{north})$$
$$y_{sun} = c_y - r_{sun} \cos(\gamma_s - \gamma_{north})$$
where $\gamma_{north}$ is the azimuthal compass offset of the camera mount.

### B. Differentiable Clear-Sky Modeling (Diff-CSM)
Rather than computing clear-sky radiation statically, we formulate the Ineichen-Perez atmospheric radiative transfer formulation [10] as a differentiable computational graph. Theoretical clear-sky GHI is expressed as:
$$GHI_{cs}(\theta_z, T_L) = c_{g1} I_0 \cos \theta_z \exp\left( -c_{g2} AM \cdot [f_{h1} + f_{h2}(T_L - 1)] \right) \cdot \exp\left( 0.01 AM^{1.8} \right)$$
where $I_0 = 1367 \cdot (1 + 0.034221 \cos \Gamma)\text{ W/m}^2$ is extraterrestrial normal irradiance, $AM$ is the optical air mass corrected for site atmospheric pressure ratio $p/p_0 = \exp(-h / 8434.5\text{ m})$ via Kasten-Young formulation [11], and coefficients $c_{g1}, c_{g2}, f_{h1}, f_{h2}$ model site altitude corrections.

Crucially, we treat the Linke Turbidity $T_L$ as an adaptive parameter:
$$T_L = T_{L, base} + \Delta T_L(X_{visual})$$
where $\Delta T_L \in [-1.0, +2.5]$ is predicted by the network from background Rayleigh-Mie sky color gradients. The analytical gradient with respect to atmospheric turbidity:
$$\frac{\partial GHI_{cs}}{\partial T_L} = -c_{g2} AM f_{h2} \cdot GHI_{cs}$$
flows backward seamlessly during backpropagation, enforcing strict physical atmospheric calibration.

### C. Sun-Centric Log-Polar Coordinate Transformation
Standard Cartesian grid convolutions $K * I(x, y)$ are shift-invariant in Euclidean space, but fisheye sky camera cloud dynamics are rotational around the zenith and directed toward the sun. We implement a dynamic bilinear interpolation operator $\mathcal{W}_{polar}: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{N_r \times N_\theta \times 3}$ defined by:
$$x(\rho, \theta) = x_{sun} + \rho \cos \theta, \quad y(\rho, \theta) = y_{sun} + \rho \sin \theta$$
where radial coordinate $\rho \in [1, R_{cutoff}]$ discretizes the distance from the sun disk and $\theta \in [0, 2\pi)$ discretizes the surrounding halo angles.

Under this transformation:
- $\rho = 0$ corresponds to the solar disk.
- Approaching cloud vectors $\vec{v}_{advect}$ oriented toward the direct beam collapse into a **strictly 1D radial translation along the $\rho$-axis ($\frac{d\rho}{dt} < 0$)**.
- The time-to-occlusion is directly governed by $\tau_{arrival} = \frac{\rho_{cloud}}{|\vec{v}_\rho|}$.

### D. Physical Attenuation-Lensing Disentanglement
When clouds intersect the circumsolar cone, irradiance undergoes three distinct physical regimes:
1. **Direct Beam Attenuation ($\mathcal{A}_{direct} \in [0, 1]$):** Optical depth absorption and backward scattering blocking the solar disk.
2. **Diffuse Sky Radiance ($\mathcal{S}_{diffuse} \approx 0.18\text{--}0.25$):** Multi-scattered background radiation reaching the horizontal surface even when the sun is fully occluded.
3. **Circumsolar Lensing Enhancement ($\mathcal{E}_{lens} \ge 0$):** Forward Mie scattering along thin cumulus cloud edges which, when adjacent to the sun, concentrates light, causing measured GHI to exceed theoretical clear-sky irradiance by $10\text{--}25\%$ [12].

Our network predicts these components explicitly:
$$\widehat{GHI}(t + \Delta t) = GHI_{cs}(t + \Delta t) \cdot \left[ (1 - \mathcal{A}_{direct}) + \mathcal{S}_{diffuse} + \mathcal{E}_{lens} \right]$$
where $\mathcal{A}_{direct} = \sigma(W_a h_{fused} + b_a)$ and $\mathcal{E}_{lens} = \text{ReLU}(W_e h_{fused} + b_e) \cdot 0.15$.

### E. Inductive Conformal Prediction (ICP)
To equip forecasts with formal statistical validity, we implement Inductive Conformal Prediction [13]. On a proper validation calibration set $\mathcal{D}_{val} = \{(X_i, y_i)\}_{i=1}^M$, we compute normalized non-conformity scores:
$$\alpha_i = |y_i - \hat{y}_i|$$
Given a desired user-specified coverage level $1 - \epsilon = 0.90$, we compute the empirical $(1 - \epsilon)(1 + 1/M)$-th quantile $\hat{q}$:
$$\hat{q} = \text{Quantile}\left(\{\alpha_i\}_{i=1}^M, \lceil (1 - \epsilon)(M + 1) \rceil / M\right)$$
For any new test prediction $\hat{y}_{test}$, the prediction interval is given by:
$$\mathcal{C}(X_{test}) = \left[ \max(0, \hat{y}_{test} - \hat{q}), \hat{y}_{test} + \hat{q} \right]$$
By exchangeability of daytime meteorological sequences within seasonal regimes, ICP provides distribution-free, finite-sample coverage guarantees:
$$\mathbb{P}\left( y_{test} \in \mathcal{C}(X_{test}) \right) \ge 1 - \epsilon$$

---

## III. EXPERIMENTAL CONFIGURATION

### A. Benchmark Datasets
Experiments were conducted using multi-year secondary standard pyranometer GHI, Direct Normal Irradiance (DNI), and all-sky camera datasets from **Folsom ($38.642^\circ\text{ N}, -121.148^\circ\text{ W}$)** [14] and **Stanford SKIPP'D ($37.4275^\circ\text{ N}, -122.1697^\circ\text{ W}$)** [15].

### B. Evaluation Protocol on Intermittent Cloud Regimes
In solar forecasting literature, evaluating models over uninterrupted clear-sky periods yields artificially deflated RMSE metrics that mask true operational skill. To ensure rigorous validation, our test protocol explicitly isolates **intermittent cloudy and broken-sky days** ($\sigma(k_t) > 0.20$ and ramp rates $|\Delta GHI / \Delta t| > 80\text{ W/m}^2/\text{min}$). The forecast horizon was configured for **10 minutes ahead ($\Delta t = 10\text{ min}$)**.

---

## IV. EMPIRICAL RESULTS

### A. Quantitative Forecasting Performance
Table I reports the quantitative performance across 3,079 real daytime observations during dynamic cloud transitions.

**TABLE I: Multi-Model Benchmark Comparison on Real Sky Camera Data ($\Delta t = 10\text{ min}$ Horizon)**
| Model Architecture | Test RMSE ($\text{W/m}^2$) | Test MAE ($\text{W/m}^2$) | rRMSE (%) | Forecast Skill Score ($SS_{RMSE}$) |
|---|---|---|---|---|
| Standard Persistence | 189.46 | 117.63 | 66.60% | -0.44% |
| Clear-Sky Only (Ineichen CSM) | 299.67 | 250.95 | 105.34% | -58.87% |
| **Smart Persistence ($k_t$ Reference)** | **188.62** | **116.56** | **66.31%** | **+0.00% (Baseline)** |
| Direct Vision Regression (Without CSM) | 173.70 | 131.39 | 61.06% | +7.91% |
| Gao & Liu (*WACV 2022*) Residual Model | 176.33 | 147.65 | 61.99% | +6.52% |
| **2026+ Diff-CSM Polar-Net (Proposed)** | **150.98** | **104.79** | **53.08%** | **+19.96%** |

### B. Key Findings
1. **Significant Predictive Skill (+19.96%):** Proposed Diff-CSM Polar-Net reduces RMSE from $188.62\text{ W/m}^2$ (Smart Persistence) down to **$150.98\text{ W/m}^2$**, establishing a strong positive Skill Score of **$+19.96\%$**.
2. **Three-Fold Improvement Over Gao & Liu (2022):** While the Gao & Liu (2022) clear-sky residual formulation achieved $+6.52\%$ skill ($RMSE = 176.33\text{ W/m}^2$), our proposed architecture improves skill to $+19.96\%$, reducing RMSE by an additional $25.35\text{ W/m}^2$ and reducing MAE from $147.65\text{ W/m}^2$ to **$104.79\text{ W/m}^2$** (a **$29.0\%$ MAE reduction**).
3. **Conformal Uncertainty Calibration:** Calibrated using $N_{val} = 460$ validation samples, the $90\%$ margin achieved $\hat{q} = \pm 286.8\text{ W/m}^2$, with an empirical test coverage of **$91.4\%$**, meeting grid safety standards.

---

## V. REFERENCES

- [1] R. Perez et al., "Spatial and temporal variability of solar energy," *Found. Trends Renew. Energy*, vol. 1, no. 1, pp. 1–44, 2016.
- [2] J. Kleissl, *Solar Energy Forecasting and Resource Assessment*, Academic Press, 2013.
- [3] H. T. Pedro and C. F. Coimbra, "Assessment of forecasting techniques for solar power production with no exogenous inputs," *Solar Energy*, vol. 86, no. 7, pp. 2017–2028, 2012.
- [4] D. Yang et al., "History and trends in solar irradiance and PV power forecasting using sky images," *Solar Energy*, vol. 165, pp. 120–138, 2018.
- [5] F. M. West et al., "A review of machine learning applications for solar nowcasting using ground-based sky images," *Renew. Sustain. Energy Rev.*, vol. 145, p. 111079, 2021.
- [6] H. Gao and M. Liu, "Short-term solar irradiance prediction from sky images with a clear sky model," in *Proc. IEEE/CVF WACV*, 2022, pp. 3074–3082.
- [7] C. A. Gueymard, "Clear-sky irradiance predictions for solar resource mapping and project development: A review," *Solar Energy*, vol. 86, no. 8, pp. 2148–2169, 2012.
- [8] P. Blanc et al., "Direct normal irradiance related definitions and applications: The circumsolar issue," *Solar Energy*, vol. 110, pp. 561–577, 2014.
- [9] P. Lauret et al., "Probabilistic solar forecasting using quantile regression methods," *Energies*, vol. 10, no. 10, p. 1591, 2017.
- [10] P. Ineichen and R. Perez, "A new airmass independent formulation for the Linke turbidity coefficient," *Solar Energy*, vol. 73, no. 3, pp. 151–157, 2002.
- [11] F. Kasten and A. T. Young, "Revised optical air mass tables and approximation formula," *Appl. Opt.*, vol. 28, no. 22, pp. 4735–4738, 1989.
- [12] A. Yordanov et al., "Cloud enhancement of global horizontal irradiance: Physical mechanisms and statistical modeling," *IEEE J. Photovolt.*, vol. 11, no. 4, pp. 1024–1031, 2021.
- [13] V. Vovk et al., *Algorithmic Learning in a Random World*, Springer, 2005.
- [14] H. T. Pedro et al., "A comprehensive dataset for the accelerated development and benchmarking of solar forecasting methods," *Renew. Energy*, vol. 142, pp. 1122–1132, 2019.
- [15] Y. Nie et al., "SKIPP'D: A SKy Images and Photovoltaic Power Generation Dataset for short-term solar forecasting," *Solar Energy*, vol. 255, pp. 185–194, 2023.

# fema-p695-ssf

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)

FEMA P-695 Spectral Shape Factor (SSF) & seismic fragility analysis toolkit.

This package provides an end-to-end workflow to compute the target spectrum and Conditional Mean Spectrum (CMS), 
record-wise indices (ε(T), SaRatio), regressions, SSFs, and maximum-likelihood fragility parameters. 
The public API intentionally exposes a **single stable class** for users:

```python
from SSF_calculation import FEMAP695SSF
```

(Top-level API export defined in `src/fema_p695_ssf/__init__.py`.)

---

## Features

- **Target Spectrum & CMS**
  - USGS deaggregation (2008 / 2023) to build UHS and extract mean M, R.
  - BSSA2014 GMM median and ln-std across periods.
  - Baker & Jayaram (2008) ρ(T, T*) correlation.
  - Conditional Mean Spectrum (CMS), target epsilon(T), and target SaRatio.
- **Record-wise Indices & Regressions**
  - Per-record epsilon(T) and SaRatio from response spectra and earthquake metadata.
  - Two regressions with plots: `ln(Sac) ~ epsilon(T)` and `Sac ~ SaRatio`.
- **SSF & Fragility**
  - SSF from regression slopes evaluated at target indices.
  - GLM (Binomial + Probit) MLE for fragility parameters (θ, β).
  - Plot original vs shifted fragility curves.

---



## Installation

> Requires Python ≥3.9

Please download the code from the git and build the environment using requirements.txt
## Quick Start

```python
from SSF_calculation import FEMAP695SSF
import pandas as pd

# A two-column table [EQ_ID, Sac]
Sac_Form = pd.read_excel("Sac_Form.xlsx")

model = FEMAP695SSF(
    T=1.12,
    longitude=-118.25, latitude=34.05,
    Vs30=259,
    return_period=9950,
    a=0.2, b=3,
    version="2008",  # "2008" or "2023"
    Sac_Form=Sac_Form,
    region="california",
    mechanism="SS",
    Z10=-1
)

model.run_SSF()

print("SSF (epsilon):", model.SSF_epsilon)
print("SSF (SaRatio):", model.SSF_SaRatio)
```

---

## Dependencies

- numpy
- scipy
- pandas
- matplotlib
- requests
- statsmodels
- openpyxl
- pygmm

> Internet required for USGS APIs. Ensure longitude/latitude are valid for 2008 WUS or 2023 CONUS endpoints.

---

## References

- Baker, J. W., & Jayaram, N. (2008). *Earthquake Spectra* — Correlation of spectral acceleration values from NGA models.
- Boore, D. M., Stewart, J. P., Seyhan, E., & Atkinson, G. M. (2014). *Earthquake Spectra* — NGA-West2 spectra equations.
- FEMA P-695 (2009/2013). *Quantification of Building Seismic Performance Factors (P-695)*.

---


## License

MIT License © 2025 **Zhiyang Mao & Christianos Burlotos**  
See [LICENSE](./LICENSE)

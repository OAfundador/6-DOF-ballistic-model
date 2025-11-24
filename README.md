# 6-DOF Ballistic Model

This repository contains an implementation of the 6-DOF ballistic equations for a spinning projectile, based on the book *Modern Exterior Ballistics: The Launch and Flight Dynamics of Symmetric Projectiles (2nd ed.)* by R. McCoy.  

The implementation was developed with the help of artificial intelligences such as Claude and ChatGPT, reviewed by me, and used in my undergraduate thesis in Applied and Computational Mathematics at IME-USP.

---

## Features

- Full 6-DOF rigid-body model for a symmetric projectile  
- Numerical integration with SciPy (`solve_ivp`)
- Aerodynamic coefficients read from a CSV table and interpolated
- Simple Monte Carlo perturbations on initial firing conditions (elevation / azimuth errors)
- Basic trajectory plots (3D path, ground impact, etc.)
- Code used in a real undergraduate thesis, with parameters tuned for a naval artillery use-case

---

## Repository structure

- `Motor.py`  
  Main script containing the 6-DOF model, configuration of the projectile and environment, reading of the aerodynamic coefficients and plotting of the results.

- `Exemplo.py`  
  Contains an example function/code for a single shot configuration that can be copied into `Motor.py` to run a demonstration.

- `Coeficientes que vi 2 casas.csv`  
  Aerodynamic coefficient table used by the model (drag, lift, moments, etc.) as a function of Mach and total angle of attack.

- `TCC/`  
  (Optional) Material related to my undergraduate thesis where this code was used.

---

## Requirements

- Python 3.9
- The following Python packages:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `pandas`


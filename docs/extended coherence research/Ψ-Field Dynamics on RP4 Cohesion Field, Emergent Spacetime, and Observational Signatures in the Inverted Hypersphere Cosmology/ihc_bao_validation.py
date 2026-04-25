"""
IHC BAO Validation — Full Covariance + k=1 Shell Correction
=============================================================
Complete implementation matching the published IJT paper exactly.

Includes:
  - Full 6×6 BOSS DR12 covariance (Alam+2017)
  - Full 3×3 WiggleZ covariance (Kazin+2014)
  - k=1 co-rotating shell correction to H(z)
  - Z3 standing-wave modulation to D_M and D_H
  - All 33 measurements across 7 surveys

Published results (Peacock & Hall 2026, IJT):
  IHC  χ²/n = 0.916   ΛCDM χ²/n = 1.196   Δχ² = +9.22

Samuel Peacock / Lauren Hall  March 2026
"""

import sys
import io

# Standardize console encoding for math symbols
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import quad
from scipy.optimize import brentq
import shutil

# IHC constants 
φ        = (1 + np.sqrt(5)) / 2
H0        = 67.4              # km/s/Mpc
c_kms     = 2.998e5           # km/s
Ω_m       = 0.3111
Ω_L       = 0.6889
R_H       = c_kms / H0        # 4448.0 Mpc
r_s_camb  = 147.78            # Mpc  CAMB acoustic integral
r_s_IHC   = R_H * φ**(-7)      # 153.20 Mpc  k=7 shell
ξ         = r_s_IHC / r_s_camb  # 1.0367
A_Z3      = 5.9441 / 1344.5   # 0.00442 = β_coh / β
λ_Z3      = R_H / 11           # 404.4 Mpc

# k=1 shell correction parameters (all zero-parameter predictions)
R_1  = R_H * φ**(-1)           # 2749 Mpc  k=1 co-rotating shell
Δz = (R_H*(φ**(-1) - φ**(-2)) * H0 *
           np.sqrt(Ω_m*(1+0.754)**3 + Ω_L) / c_kms)  # = 0.363

CIHC  = '#c0392b'; CGOLD = '#b8860b'; CFIT  = '#2980b9'
CGN   = '#27ae60'; CPUR  = '#7d3c98'; CGRY  = '#7f8c8d'
plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})

# Background functions 
def E_base(z):
    return np.sqrt(Ω_m*(1+z)**3 + Ω_L)

# k=1 shell correction: f(z) = 1 + (ξ-1)/2 * [1 + tanh((z1-z)/Δz)]
def chi_fn(z):
    return quad(lambda zp: c_kms/(H0*E_base(zp)), 0, z, limit=300)[0]

# Find z_1 (redshift where χ = R_1)
z1 = brentq(lambda z: chi_fn(z) - R_1, 0.5, 1.5)

def f_k1(z):
    """k=1 shell expansion correction factor."""
    return 1 + (ξ-1)/2 * (1 + np.tanh((z1 - z) / Δz))

def DH_IHC(z):
    """D_H/r_s with k=1 shell correction and Z3 modulation."""
    ch   = chi_fn(z)
    E_ξ = E_base(z) * f_k1(z)
    z3   = 1 + A_Z3 * np.sin(2*np.pi*ch/λ_Z3)
    return (c_kms / (H0 * E_ξ)) * z3 / r_s_camb

def DM_IHC(z):
    """D_M/r_s with Z3 modulation (no k=1 correction – cancels over LOS)."""
    ch = chi_fn(z)
    z3 = 1 - A_Z3 * np.cos(2*np.pi*ch/λ_Z3)
    return ch * z3 / r_s_camb

def DV_IHC(z):
    """D_V/r_s = [z D_M^2 D_H]^(1/3) / r_s."""
    ch   = chi_fn(z)
    DM_m = ch * (1 - A_Z3*np.cos(2*np.pi*ch/λ_Z3))
    E_ξ = E_base(z) * f_k1(z)
    DH_m = c_kms/(H0*E_ξ) * (1 + A_Z3*np.sin(2*np.pi*ch/λ_Z3))
    return (z * DM_m**2 * DH_m)**(1/3) / r_s_camb

def DH_LCDM(z):  return (c_kms/(H0*E_base(z))) / r_s_camb
def DM_LCDM(z):  return chi_fn(z) / r_s_camb
def DV_LCDM(z):
    ch=chi_fn(z); dh=c_kms/(H0*E_base(z))
    return (z*ch**2*dh)**(1/3) / r_s_camb

# Covariance matrices 
# BOSS DR12 6×6 (Alam+2017): order = DM(0.38), DH(0.38), DM(0.51), DH(0.51), DM(0.61), DH(0.61)
sigma_boss = np.array([0.152, 0.580, 0.183, 0.420, 0.251, 0.390])
rho_boss   = np.array([
    [ 1.000, -0.524,  0.396, -0.171,  0.168, -0.074],
    [-0.524,  1.000, -0.185,  0.319, -0.083,  0.128],
    [ 0.396, -0.185,  1.000, -0.524,  0.391, -0.176],
    [-0.171,  0.319, -0.524,  1.000, -0.193,  0.333],
    [ 0.168, -0.083,  0.391, -0.193,  1.000, -0.524],
    [-0.074,  1.000, -0.176,  0.333, -0.524,  1.000],
])
C_boss     = np.outer(sigma_boss, sigma_boss) * rho_boss
C_inv_boss = np.linalg.inv(C_boss)

# WiggleZ 3×3 (Kazin+2014): order = DV(0.44)/r_s, DV(0.60)/r_s, DV(0.73)/r_s
sigma_wz = np.array([83.0, 101.0, 86.0]) / r_s_camb
rho_wz   = np.array([
    [1.000, 0.390, 0.332],
    [0.390, 1.000, 0.520],
    [0.332, 0.520, 1.000],
])
C_wz     = np.outer(sigma_wz, sigma_wz) * rho_wz
C_inv_wz = np.linalg.inv(C_wz)

print('=' * 65)
print('IHC BAO VALIDATION — FULL COVARIANCE + k=1 CORRECTION')
print('=' * 65)
print(f'  r_s^IHC = {r_s_IHC:.3f} Mpc    ξ = {ξ:.5f}')
print(f'  A_Z3 = {A_Z3*100:.3f}%   λ_Z3 = {λ_Z3:.1f} Mpc')
print(f'  k=1 shell: R_1 = {R_1:.1f} Mpc  z_1 = {z1:.3f}   Δz = {Δz:.3f}')
print(f'  BOSS 6x6 covariance: positive definite = {np.all(np.linalg.eigvalsh(C_boss)>0)}')
print(f'  WiggleZ 3x3 covariance: positive definite = {np.all(np.linalg.eigvalsh(C_wz)>0)}')

# Survey data 
surveys_diag = [
    # name,         z,     type, obs,                      err,          use_cov
    ('6dFGS',       0.106, 'DV', 456.0/r_s_camb,           27.0/r_s_camb, False),
    ('MGS',         0.150, 'DV', 4.47,                      0.17,          False),
    ('SDSS_LRG',    0.350, 'DV', 8.88*153.19/r_s_camb,     0.175,         False),
    ('eBOSS_DM1',   0.700, 'DM', 17.65,                    0.300,         False),
    ('eBOSS_DH1',   0.700, 'DH', 19.77,                    0.470,         False),
    ('eBOSS_DM2',   1.480, 'DM', 30.69,                    0.800,         False),
    ('eBOSS_DH2',   1.480, 'DH', 13.26,                    0.550,         False),
    ('eBOSS_DM3',   2.330, 'DM', 37.55,                    1.140,         False),
    ('eBOSS_DH3',   2.330, 'DH',  8.93,                    0.280,         False),
    ('eBOSS_DM4',   2.340, 'DM', 37.34,                    1.250,         False),
    ('eBOSS_DH4',   2.340, 'DH',  9.08,                    0.290,         False),
    ('DESI_DV1',    0.295, 'DV',  7.942,                   0.076,         False),
    ('DESI_DM2',    0.510, 'DM', 13.588,                   0.168,         False),
    ('DESI_DH2',    0.510, 'DH', 21.863,                   0.429,         False),
    ('DESI_DM3',    0.706, 'DM', 17.351,                   0.180,         False),
    ('DESI_DH3',    0.706, 'DH', 19.455,                   0.334,         False),
    ('DESI_DM4',    0.934, 'DM', 21.576,                   0.162,         False),
    ('DESI_DH4',    0.934, 'DH', 17.642,                   0.201,         False),
    ('DESI_DM5',    1.321, 'DM', 27.601,                   0.325,         False),
    ('DESI_DH5',    1.321, 'DH', 14.176,                   0.225,         False),
    ('DESI_DM6',    1.484, 'DM', 30.512,                   0.764,         False),
    ('DESI_DH6',    1.484, 'DH', 12.817,                   0.518,         False),
    ('DESI_DM7',    2.330, 'DM', 38.989,                   0.532,         False),
    ('DESI_DH7',    2.330, 'DH',  8.632,                   0.101,         False),
]

# Correlated blocks
boss_obs  = np.array([10.238, 24.930, 13.361, 22.380, 15.612, 20.720])
boss_z    = [0.38, 0.38, 0.51, 0.51, 0.61, 0.61]
boss_types= ['DM','DH','DM','DH','DM','DH']

wz_obs  = np.array([1716.0, 2221.0, 2516.0]) / r_s_camb
wz_z    = [0.44, 0.60, 0.73]
wz_errs = np.array([83.0, 101.0, 86.0]) / r_s_camb

# Predictions 
print('\nComputing predictions...')

def pred_IHC(z, tp):
    if tp=='DM': return DM_IHC(z)
    if tp=='DH': return DH_IHC(z)
    return DV_IHC(z)

def pred_LCDM(z, tp):
    if tp=='DM': return DM_LCDM(z)
    if tp=='DH': return DH_LCDM(z)
    return DV_LCDM(z)

# BOSS predictions
boss_pred_ihc  = np.array([pred_IHC(z, tp)  for z,tp in zip(boss_z, boss_types)])
boss_pred_lcdm = np.array([pred_LCDM(z, tp) for z,tp in zip(boss_z, boss_types)])

# WiggleZ predictions
wz_pred_ihc  = np.array([DV_IHC(z)  for z in wz_z])
wz_pred_lcdm = np.array([DV_LCDM(z) for z in wz_z])

# χ² calculations 
# BOSS: full 6x6
r_boss_ihc  = boss_obs - boss_pred_ihc
r_boss_lcdm = boss_obs - boss_pred_lcdm
chi2_boss_ihc  = float(r_boss_ihc  @ C_inv_boss @ r_boss_ihc)
chi2_boss_lcdm = float(r_boss_lcdm @ C_inv_boss @ r_boss_lcdm)

# WiggleZ: full 3x3
r_wz_ihc  = wz_obs - wz_pred_ihc
r_wz_lcdm = wz_obs - wz_pred_lcdm
chi2_wz_ihc  = float(r_wz_ihc  @ C_inv_wz @ r_wz_ihc)
chi2_wz_lcdm = float(r_wz_lcdm @ C_inv_wz @ r_wz_lcdm)

# Diagonal surveys
chi2_diag_ihc = chi2_diag_lcdm = 0.0
diag_results = []
for name,z,tp,obs,err,_ in surveys_diag:
    pi = pred_IHC(z,tp);  pl = pred_LCDM(z,tp)
    ri = (obs-pi)/err;    rl = (obs-pl)/err
    chi2_diag_ihc  += ri**2
    chi2_diag_lcdm += rl**2
    diag_results.append(dict(name=name,z=z,tp=tp,obs=obs,err=err,
                             ihc=pi,lcdm=pl,ri=ri,rl=rl))

# Total
n_total = 6 + 3 + len(surveys_diag)   # = 33
chi2_total_ihc  = chi2_boss_ihc  + chi2_wz_ihc  + chi2_diag_ihc
chi2_total_lcdm = chi2_boss_lcdm + chi2_wz_lcdm + chi2_diag_lcdm

# Tests 
print('\n Tests ')
passed = 0; failed = 0

def check(label, val, expected, tol=0.10):
    global passed, failed
    err = abs(val-expected)/abs(expected) if abs(expected)>1e-9 else abs(val)
    ok  = err < tol
    if ok: passed += 1
    else:  failed += 1
    print(f'  {"PASS" if ok else "FAIL"}  {label}')
    print(f'         got {val:.5g}  expected {expected:.5g}  err {err*100:.1f}%')

check('r_s^IHC = R_H φ⁻⁷',     r_s_IHC,    153.2,   tol=0.001)
check('ξ = 1.0367',              ξ,         1.0367,  tol=0.001)
check('z_1 = 0.754 (k=1 shell)', z1,         0.754,   tol=0.005)
check('Δz = 0.363',              Δz,    0.363,   tol=0.01)
check('n_total = 33',            float(n_total), 33., tol=0.001)
check('BOSS χ²/n (IHC)',   chi2_boss_ihc/6,  0.97, tol=0.15)
check('BOSS χ²/n (ΛCDM)',  chi2_boss_lcdm/6, 1.11, tol=0.15)
check('WiggleZ χ²/n (IHC)',chi2_wz_ihc/3,    0.42, tol=0.30)
check('Total χ²/n (IHC)',       chi2_total_ihc/n_total,  0.916, tol=0.10)
check('Total χ²/n (ΛCDM)',      chi2_total_lcdm/n_total, 1.196, tol=0.10)
check('Δχ² > 0 (IHC better)',   float(chi2_total_lcdm > chi2_total_ihc), 1.0, tol=0.001)

# Results table 
print(f'\n{"":=<65}')
print(f'{"Survey":<14} {"z":>5} {"T":>2}  {"Obs":>7} {"IHC":>7} {"ΛCDM":>7}  '
      f'{"resIHC":>7} {"resΛCDM":>8}')
print('-'*72)

# Print BOSS block
sigma_eff_boss = np.sqrt(np.diag(C_boss))
for i,(z,tp) in enumerate(zip(boss_z,boss_types)):
    obs=boss_obs[i]; se=sigma_eff_boss[i]
    pi=boss_pred_ihc[i]; pl=boss_pred_lcdm[i]
    print(f'  BOSS {tp}{["38","38","51","51","61","61"][i]}    '
          f'{z:>5.3f} {tp:>2}  {obs:>7.4f} {pi:>7.4f} {pl:>7.4f}  '
          f'{(obs-pi)/se:>+6.2f} {(obs-pl)/se:>+7.2f}')
print(f'   BOSS χ²/n:  IHC={chi2_boss_ihc/6:.4f}  ΛCDM={chi2_boss_lcdm/6:.4f}  '
      f'(6x6 cov, pub: 0.97/1.11)')
print()

# WiggleZ block
for i,z in enumerate(wz_z):
    obs=wz_obs[i]; se=wz_errs[i]
    pi=wz_pred_ihc[i]; pl=wz_pred_lcdm[i]
    print(f'  WigZ_{z}   {z:>5.2f} DV  {obs:>7.4f} {pi:>7.4f} {pl:>7.4f}  '
          f'{(obs-pi)/se:>+6.2f} {(obs-pl)/se:>+7.2f}')
print(f'   WiggleZ χ²/n: IHC={chi2_wz_ihc/3:.4f}  ΛCDM={chi2_wz_lcdm/3:.4f}  '
      f'(3x3 cov, pub: 0.42/0.52)')
print()

# Diagonal surveys
prev_grp = ''
for r in diag_results:
    grp = r['name'].split('_')[0]
    if grp != prev_grp and prev_grp: print()
    prev_grp = grp
    print(f'  {r["name"]:<13} {r["z"]:>5.3f} {r["tp"]:>2}  '
          f'{r["obs"]:>7.4f} {r["ihc"]:>7.4f} {r["lcdm"]:>7.4f}  '
          f'{r["ri"]:>+6.2f} {r["rl"]:>+7.2f}')

print(f'\n{"-"*65}')
print(f'  TOTAL ({n_total} meas):  IHC χ²/n = {chi2_total_ihc/n_total:.4f}   '
      f'ΛCDM χ²/n = {chi2_total_lcdm/n_total:.4f}')
print(f'  Δχ² = +{chi2_total_lcdm-chi2_total_ihc:.3f} in favour of IHC')
print(f'  Published: IHC=0.916  ΛCDM=1.196  Δχ²=+9.22')

# Figure 
print('\nGenerating figure...')

fig = plt.figure(figsize=(15, 11))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.34)

z_c  = np.linspace(0.05, 2.5, 200)
print('  Computing prediction curves...')
DM_c = np.array([DM_IHC(z)  for z in z_c])
DH_c = np.array([DH_IHC(z)  for z in z_c])
DV_c = np.array([DV_IHC(z)  for z in z_c])
DM_l = np.array([DM_LCDM(z) for z in z_c])
DH_l = np.array([DH_LCDM(z) for z in z_c])
DV_l = np.array([DV_LCDM(z) for z in z_c])

# Survey colours
s_col = {'6dFGS':CGRY,'MGS':CPUR,'SDSS':CPUR,'WigZ':('#e67e22'),
         'BOSS':CGOLD,'eBOSS':('#16a085'),'DESI':CGN}
def get_col(name):
    return next((v for k,v in s_col.items() if k in name), CGRY)

for ax_i, (ylabel, Yc, Yl, tp_key) in enumerate([
        ('$D_M(z)/r_s$', DM_c, DM_l, 'DM'),
        ('$D_H(z)/r_s$', DH_c, DH_l, 'DH'),
        ('$D_V(z)/r_s$', DV_c, DV_l, 'DV')]):
    ax = fig.add_subplot(gs[0, ax_i])
    ax.plot(z_c, Yc, '-',  color=CIHC, lw=2.2, label='IHC')
    ax.plot(z_c, Yl, '--', color=CFIT, lw=1.8, label=r'$\Lambda$CDM')
    # BOSS points
    if tp_key in ('DM','DH'):
        for i,(z,tp) in enumerate(zip(boss_z,boss_types)):
            if tp==tp_key:
                se=sigma_eff_boss[i]
                ax.errorbar(z, boss_obs[i], yerr=se, fmt='o',
                            color=CGOLD, ms=6, capsize=3, lw=1.5, zorder=6)
    # WiggleZ DV
    if tp_key=='DV':
        ax.errorbar(wz_z, wz_obs, yerr=wz_errs, fmt='^',
                    color='#e67e22', ms=6, capsize=3, lw=1.5, zorder=6, label='WiggleZ')
    # Diagonal
    for r in diag_results:
        if r['tp']==tp_key:
            ax.errorbar(r['z'], r['obs'], yerr=r['err'], fmt='s',
                        color=get_col(r['name']), ms=5, capsize=2, lw=1, zorder=5, alpha=0.8)
    if tp_key=='DH':
        ax.axvline(z1, color=CPUR, lw=1.2, ls=':', alpha=0.6,
                   label=f'$z_1$={z1:.3f}')
    ax.set_xlabel('Redshift $z$', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

# Residuals panel — all 33 measurements
ax4 = fig.add_subplot(gs[1, :2])

# BOSS (diagonal approx for residual plot)
for i,(z,tp) in enumerate(zip(boss_z,boss_types)):
    se = sigma_eff_boss[i]
    mk = 'o' if tp=='DM' else 's'
    ax4.scatter(z-0.01, (boss_obs[i]-boss_pred_ihc[i])/se,
                marker=mk, color=CIHC, s=60, zorder=6)
    ax4.scatter(z+0.01, (boss_obs[i]-boss_pred_lcdm[i])/se,
                marker=mk, color=CGRY, s=35, zorder=5, alpha=0.6)

# WiggleZ
for i,z in enumerate(wz_z):
    se = wz_errs[i]
    ax4.scatter(z-0.01, (wz_obs[i]-wz_pred_ihc[i])/se,
                marker='^', color=CIHC, s=60, zorder=6)
    ax4.scatter(z+0.01, (wz_obs[i]-wz_pred_lcdm[i])/se,
                marker='^', color=CGRY, s=35, zorder=5, alpha=0.6)

# Diagonal
for r in diag_results:
    mk = {'DM':'o','DH':'s','DV':'^'}[r['tp']]
    ax4.scatter(r['z']-0.01, r['ri'], marker=mk,
                color=CIHC, s=50, zorder=6, alpha=0.9)
    ax4.scatter(r['z']+0.01, r['rl'], marker=mk,
                color=CGRY, s=30, zorder=5, alpha=0.5)

for y,c,ls,lw in [(0,'k','-',0.8),(1,CGOLD,'--',1.2),(-1,CGOLD,'--',1.2),
                   (2,'salmon',':',1.0),(-2,'salmon',':',1.0)]:
    ax4.axhline(y, color=c, lw=lw, ls=ls, alpha=(1 if y==0 else 0.6))

ax4.axvline(z1, color=CPUR, lw=1.2, ls=':', alpha=0.5,
            label=f'k=1 shell z={z1:.3f}')

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax4.legend([Patch(color=CIHC), Patch(color=CGRY),
            Line2D([0],[0],color=CGOLD,ls='--'),
            Line2D([0],[0],color='salmon',ls=':'),
            Line2D([0],[0],color=CPUR,ls=':')],
           [f'IHC   (χ²/n={chi2_total_ihc/n_total:.3f})',
            f'ΛCDM (χ²/n={chi2_total_lcdm/n_total:.3f})',
            '1σ', '2σ', f'k=1 shell z={z1:.3f}'],
           fontsize=9, loc='upper right')
ax4.set_xlabel('Redshift $z$', fontsize=11)
ax4.set_ylabel('$(O-P)/\\sigma$', fontsize=11)
ax4.set_title(
    f'Residuals — All {n_total} Measurements (7 Surveys)\n'
    f'IHC χ²/n={chi2_total_ihc/n_total:.3f}  ΛCDM χ²/n={chi2_total_lcdm/n_total:.3f}  '
    f'Δχ²=+{chi2_total_lcdm-chi2_total_ihc:.2f}  '
    f'(published: 0.916 vs 1.196, Δχ²=+9.22)', fontsize=10)
ax4.set_xlim(0, 2.5); ax4.set_ylim(-3.2, 3.2)
ax4.grid(True, alpha=0.15)

# Per-survey chi/n bar chart
ax5 = fig.add_subplot(gs[1, 2])
bar_data = [
    ('6dFGS\n(n=1)',    sum(r['ri']**2 for r in diag_results if r['name']=='6dFGS'),
                        sum(r['rl']**2 for r in diag_results if r['name']=='6dFGS'), 1),
    ('MGS+SDSS\n(n=2)', sum(r['ri']**2 for r in diag_results if r['name'] in ['MGS','SDSS_LRG']),
                        sum(r['rl']**2 for r in diag_results if r['name'] in ['MGS','SDSS_LRG']), 2),
    ('WiggleZ\n(n=3)',  chi2_wz_ihc, chi2_wz_lcdm, 3),
    ('BOSS\nDR12 (n=6)',chi2_boss_ihc, chi2_boss_lcdm, 6),
    ('eBOSS\nDR16 (n=8)',sum(r['ri']**2 for r in diag_results if 'eBOSS' in r['name']),
                         sum(r['rl']**2 for r in diag_results if 'eBOSS' in r['name']), 8),
    ('DESI\nDR2 (n=13)',sum(r['ri']**2 for r in diag_results if 'DESI' in r['name']),
                        sum(r['rl']**2 for r in diag_results if 'DESI' in r['name']), 13),
]

x = np.arange(len(bar_data)); w = 0.35
g_ihc  = [d[1]/d[3] for d in bar_data]
g_lcdm = [d[2]/d[3] for d in bar_data]
g_lbl  = [d[0] for d in bar_data]

ax5.bar(x-w/2, g_ihc,  w, color=CIHC, alpha=0.85, label='IHC',  edgecolor='white')
ax5.bar(x+w/2, g_lcdm, w, color=CGRY, alpha=0.85, label='ΛCDM', edgecolor='white')
ax5.axhline(1.0, color='k', lw=1.2, ls='--', alpha=0.6, label='χ²/n=1')
ax5.set_xticks(x)
ax5.set_xticklabels(g_lbl, fontsize=8)
ax5.set_ylabel('χ²/n', fontsize=11)
ax5.set_title('Per-Survey χ²/n\n(BOSS+WiggleZ: full covariance)', fontsize=10)
ax5.legend(fontsize=9); ax5.grid(True, axis='y', alpha=0.15)

plt.savefig('fig_ch3_bao_validation.pdf')
print('  Saved.')

print()
print('=' * 65)
print(f'TESTS: {passed} passed | {failed} failed')
print('=' * 65)
print(f"""
FINAL RESULT ({n_total} measurements, 7 surveys):
  Covariance: BOSS 6x6 + WiggleZ 3x3 + diagonal elsewhere

  IHC  χ²/n = {chi2_total_ihc/n_total:.4f}   ΛCDM χ²/n = {chi2_total_lcdm/n_total:.4f}
    Δχ² = +{chi2_total_lcdm-chi2_total_ihc:.3f}  (IHC better)
  Published: IHC=0.916  ΛCDM=1.196  Δχ²=+9.22

  BOSS DR12:   IHC={chi2_boss_ihc/6:.4f}  ΛCDM={chi2_boss_lcdm/6:.4f}  (pub: 0.97/1.11)
  WiggleZ:     IHC={chi2_wz_ihc/3:.4f}  ΛCDM={chi2_wz_lcdm/3:.4f}  (pub: 0.42/0.52)

  Max IHC residual: {max(abs(r['ri']) for r in diag_results):.2f} (diagonal surveys)
""")

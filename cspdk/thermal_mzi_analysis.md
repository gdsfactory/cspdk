# Thermal MZI Circulax Simulation

## Circuit Topology

```
                      ┌──────────────┐
           W1  o1─────┤  X1 (heater) ├─────o2  W3
          ┌───────────┘  I1 drives   └───────────┐
          │              current                  │
LS1 ──W7──┤ X2 (mmi1x2)                    X4 (mmi2x2) ├──W5── PD1
          │                                      │
          └───────────┐              ┌───────────┘├──W6── PD2
           W2  o1─────┤  X3 (heater) ├─────o2  W4
                      │  both heater │
                      │  pins GND    │
                      └──────────────┘
```

### Instances

| Instance | Model                          | Function            | Notes                         |
|----------|--------------------------------|---------------------|-------------------------------|
| LS1      | OpticalSource                  | CW laser, 1 mW     | wl=1.55 µm, phase=0          |
| X2       | mmi1x2                         | 50/50 splitter      | Real-valued equal split       |
| X1       | straight_heater_meander        | Tunable arm         | I1 sweeps current (0–100 mA)  |
| X3       | straight_heater_meander        | Reference arm       | Both heater pins grounded     |
| X4       | mmi2x2                         | Combiner            | 90 deg phase between bar/cross |
| PD1      | OpticalDetector                | Photodetector       | On W5 (bar output of X4)      |
| PD2      | OpticalDetector                | Photodetector       | On W6 (cross output of X4)    |
| I1       | isource                        | DC current source   | Sweep: 0 to 100 mA, 1001 pts |

### Net Connectivity

| Net   | From          | To            |
|-------|---------------|---------------|
| W7    | LS1.p1        | X2.o1         |
| W1    | X2.o2         | X1.o1         |
| W2    | X2.o3         | X3.o1         |
| W3    | X1.o2         | X4.o1         |
| W4    | X3.o2         | X4.o2         |
| W5    | X4.o3         | PD1.p1        |
| W6    | X4.o4         | PD2.p1        |
| net5  | I1.N          | X1.r_e2       |
| GND   | I1.P, X1.l_e2, X3.l_e2, X3.r_e2 | Ground |

X3 has `transform = [1,0,0,-1]` (Y-flip), but the net assignments already
account for this — o1 connects to the splitter side (W2) and o2 to the
combiner side (W4), matching the VCVS propagation direction.

## Component Models

### ThermalPhaseShifter (X1, X3)

Computes a complex transmission coefficient:

```
T = T_mag * exp(-j * phi)
```

Where:
- `T_mag = 10^(-loss/20)` — amplitude from propagation loss
- `phi = 2*pi * neff * L/lambda + dphi` — total accumulated phase
- `dphi = pi * eta_pi_W * P_diss` — thermal phase shift from heater
- `P_diss = V_bias^2 / R` — dissipated electrical power

Default parameters: R=120 Ohm, eta_pi=12.5 rad/W, length=100 um,
loss=1 dB/cm, neff=2.4, wl=1.55 um.

The model uses a **VCVS constraint** with a **matched input admittance**.
MNA adds a state variable `i_fwd` (the output branch current) and the
residual equation `o2 - T*o1 = 0`. Light enters o1 and exits o2.

The input port stamps `Y_in = 1/z0 = 1` (matched admittance to ground),
giving S11 = 0 regardless of heater phase. The output port stamps
`-i_fwd`, where `i_fwd` is determined by KCL at the output node.

### Why Matched Input (Not Shared Branch Current)

A naive VCVS stamps the same branch current `i_fwd` at both ports
(+i_fwd at o1, -i_fwd at o2). This enforces current conservation but
gives the device an equivalent input reflection coefficient:

```
S11 = (1 - T) / (1 + T)       |S11| = |tan(phi/2)|
```

As the heater phase rotates, |S11| sweeps from 0 to infinity. The
reflected wave propagates backward through the mmi1x2, and because
the mmi1x2's Y-matrix is ill-conditioned (det(S+I) = 1 - 2t² ≈ 0.067
for 0.3 dB loss), the reflected current gets amplified ~15×. This
makes V(W1) depend on the heater phase:

```
V(W1) ∝ 1 / (T(I) + T_fixed + 2)
```

producing non-physical power oscillations at intermediate nodes.

Stamping `Y_in = 1` instead decouples the input from the output loading.
The KCL at W1 becomes `(Y_22 + 1)·V(W1) + Y_21·V(W7) + Y_23·V(W2) = 0`
— no T(I) term — so V(W1) is constant across the sweep.

### Why Not S-to-Y Conversion

The alternative formulation converts the 2-port S-matrix to an admittance
Y-matrix via Kurokawa power-wave conversion:

```
S = [[0, T], [T, 0]]   =>   Y = (1/(1-T^2)) * [[1+T^2, -2T], [-2T, 1+T^2]]
```

This works for SAX models (mmi1x2, mmi2x2, waveguides) because their S
parameters are fixed during a DC current sweep — the denominator `1 - T^2`
is a constant.

For the ThermalPhaseShifter, `T` varies with heater current. Since
`|T| ~ 1` (low loss), `T^2` traces a circle near the unit circle in the
complex plane. The denominator `1 - T^2` passes through near-zero values
periodically, causing:

- Y-matrix entries to oscillate between ~0.5 and ~500
- Non-physical amplitude oscillations at internal nodes
- PD outputs that are tiny and non-complementary

The VCVS constraint with matched input avoids both failure modes: no
denominator singularity and no phase-dependent back-reflection.

### mmi1x2 (X2)

SAX model. Real-valued equal split: `S(o1,o2) = S(o1,o3) = amp/sqrt(2)`.
No relative phase between the two output arms.

### mmi2x2 (X4)

SAX model. The bar and cross paths have a 90 deg relative phase:

- Bar: `S(o1,o3) = S(o2,o4) = thru` (real)
- Cross: `S(o1,o4) = S(o2,o3) = j * cross` (imaginary)

This makes the MZI outputs complementary. For equal-amplitude inputs
with phase difference dphi:

```
PD1 ~ |thru * E1 + j*cross * E2|^2  ~  1 + sin(dphi)
PD2 ~ |j*cross * E1 + thru * E2|^2  ~  1 - sin(dphi)
```

PD1 + PD2 = constant (power conservation).

### OpticalSource (LS1)

Single-port voltage constraint: `p1 = sqrt(power) * exp(j*phase)`.
Fixes the field amplitude and phase at the injection node using an MNA
auxiliary variable `i_src` for the source current. Zero output impedance
(voltage source), but the mmi1x2 has S11 = 0 so source mismatch is
irrelevant — no wave reflects back.

### OpticalDetector (PD1, PD2)

Single-port matched load: stamps `Y = 1/z0 = 1` at the detector node
(KCL contribution: `i = V · 1.0`). This provides proper termination for
the mmi2x2 output ports in the Y-matrix formulation. The simulation
reports `|field|^2` as detected power in mW.

## Expected Behavior

Sweeping I1 from 0 to 100 mA:
- X1 accumulates thermal phase shift `dphi = pi * 12.5 * I^2 * 120`
- X3 has zero phase shift (both heater pins grounded)
- PD1 and PD2 show complementary sinusoidal fringes
- At I=0: some initial phase offset from the baseline `2*pi*neff*L/lambda`
- Number of fringes depends on max current: at 60 mA, `dphi_max ~ 1.7*pi`

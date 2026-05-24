# Electric Circuits And Electrostatics Knowledge Base

This document is a code-owned physics knowledge base for deterministic routing,
planning validation, and executor design. It is intentionally scoped to electric
circuits and electrostatics:

- resistance, voltage, current, and power
- capacitance and capacitor energy
- electric fields, electric potential, Coulomb force, and electrostatic energy
- LC/RLC circuits, resonance, reactance, impedance, inductance
- magnetic field and magnetic flux

Core rule:

```text
Qwen plans. Deterministic code solves. Verifier decides confidence.
```

Qwen may select whitelisted IDs, but all formulas below must be represented by
code-owned registries/executors before they can produce final numeric answers.

## 1. Units And Canonical Dimensions

Use SI internally.

- Charge: `C`; common prefixes `mC`, `μC`, `nC`, `pC`.
- Voltage / potential difference: `V`.
- Current: `A`.
- Resistance / reactance / impedance: `Ω`.
- Capacitance: `F`; common prefixes `μF`, `nF`, `pF`.
- Inductance: `H`; common prefixes `mH`, `μH`.
- Energy: `J`; common prefixes `mJ`, `μJ`, `nJ`.
- Power: `W`.
- Force: `N`.
- Length: `m`; common prefixes `cm`, `mm`.
- Area: `m^2`; common prefixes `cm^2`.
- Frequency: `Hz`.
- Angular frequency: `rad/s`.
- Electric field: `V/m` or `N/C`.
- Magnetic field: `T`.
- Magnetic flux: `Wb`.
- Turn density: `turns/m`.

## 2. DC Circuit Core

Ohm law:

```text
U = I R
I = U / R
R = U / I
```

Power:

```text
P = U I
P = I^2 R
P = U^2 / R
```

Series resistors:

```text
R_eq = R1 + R2 + ...
I is the same through all series elements.
U_i = I R_i.
U_total = sum(U_i).
```

Parallel resistors:

```text
1/R_eq = 1/R1 + 1/R2 + ...
U is the same across all parallel branches.
I_i = U / R_i.
I_total = sum(I_i).
```

Safe executor patterns:

- If only `U` and one `R` are given and target is current, use `I=U/R`.
- If multiple parallel resistors are explicit and target is total current,
  compute each branch current and sum.
- Do not use DC formulas for RLC/AC text containing `impedance`, `reactance`,
  `resonance`, `quadrature`, or frequency-change cues unless an AC/RLC executor
  explicitly handles the case.

Kirchhoff laws:

```text
KCL: sum(I_in) = sum(I_out)
KVL: sum(voltage rises) = sum(voltage drops)
```

Use KCL/KVL only through deterministic circuit-topology templates. Qwen may not
invent arbitrary node labels or loop directions. Unsupported topologies must
return `Uncertain`.

Internal resistance of a source:

```text
U_terminal = E - I r
I = E / (R + r)
P_source = E I
P_load = I^2 R
P_loss_internal = I^2 r
efficiency = R/(R+r)
```

Safe triggers include `emf`, `internal resistance`, `terminal voltage`,
`battery`, and `source resistance`. Do not apply these formulas to a normal
ideal voltage source.

Joule heating and electric work:

```text
Q_heat = I^2 R t = U I t = U^2 t/R
A_work = q U
A_work = q E d
```

Executor must disambiguate electric charge `Q` from heat `Q_heat`.

## 3. Capacitor Core

Basic relations:

```text
Q = C U
C = Q / U
U = Q / C
W = 1/2 C U^2
W = Q^2 / (2C)
W = 1/2 Q U
```

Parallel-plate capacitor:

```text
C = epsilon_r epsilon_0 A / d
epsilon_r = C d / (epsilon_0 A)
```

Series capacitors:

```text
1/C_eq = 1/C1 + 1/C2 + ...
Q is the same on each capacitor.
U_i = Q / C_i.
For two series capacitors: U1 = U_total C2/(C1+C2), U2 = U_total C1/(C1+C2).
```

Parallel capacitors:

```text
C_eq = C1 + C2 + ...
U is the same across each capacitor.
Q_total = C_eq U.
```

Charge sharing with like terminals connected:

```text
U_final = (C1 U1 + C2 U2 + ...)/(C1 + C2 + ...)
```

State-change assumptions:

- Disconnected/isolated capacitor: charge `Q` stays constant.
- Still connected to ideal voltage source: voltage `U` stays constant.
- Insert dielectric while isolated: `C' = epsilon_r C`, `U' = U/epsilon_r`,
  `W' = W/epsilon_r`.
- Insert dielectric while connected: `C' = epsilon_r C`, `U' = U`,
  `W' = epsilon_r W`.
- Increase plate distance while isolated: `C` decreases as `1/d`; if distance
  doubles then `C' = C/2`, `U' = 2U`, `W' = 2W`.
- Increase plate distance while connected: `U` stays fixed; if distance doubles
  then `C' = C/2`, `W' = W/2`.
- Short-circuited ideal capacitor: final `Q=0`, final `W=0`.

## 4. Coulomb Force And Electric Field

Coulomb force magnitude:

```text
F = k |q1 q2| / r^2
k = 9e9 N*m^2/C^2
```

Equal charges:

```text
q = sqrt(F r^2 / k)
```

Electric field:

```text
E = F / |q|
F = |q| E
E_point = k |q| / r^2
E_uniform = U/d
```

Electric potential and potential energy:

```text
V_point = k q / r
U_pair = k q1 q2 / r
A_field = q E d
F = q E
```

Charged-particle motion in a uniform electric field:

```text
F = |q|E
a = |q|E/m
qU = 1/2 m v^2
v = sqrt(2|q|U/m)
stopping distance from v0 to 0 under constant electric deceleration:
s = m v0^2 / (2 |q| E)
```

Use only when particle identity or mass/charge is explicit or supplied by the
implicit KB. Signed direction must be handled by geometry/vector templates; if
direction is ambiguous, return `Uncertain`.

Microscopic current:

```text
I = n q A v_d
```

Vector superposition:

```text
F_net = vector sum of pairwise forces.
E_net = vector sum of point-charge electric fields.
```

Safe geometry templates:

- two collinear vectors, same direction
- two collinear vectors, opposite direction
- two perpendicular vectors
- two vectors with known included angle
- midpoint between equal opposite charges
- midpoint between equal same-sign charges
- equilateral triangle with explicit symmetric source charges

Reject if the diagram cannot be deterministically reconstructed from text.

## 5. Force Vector Resultants

Two-force resultant:

```text
R = F1 + F2                         same direction
R = |F1 - F2|                       opposite directions
R = sqrt(F1^2 + F2^2)               perpendicular
R = sqrt(F1^2 + F2^2 + 2F1F2cosθ)   known included angle θ
```

Inverse angle for equal forces:

```text
R^2 = F^2 + F^2 + 2F^2 cosθ
cosθ = (R^2 - 2F^2)/(2F^2)
```

## 6. LC And RLC Circuits

LC natural oscillation:

```text
omega = 1/sqrt(LC)
f = 1/(2πsqrt(LC))
T = 2πsqrt(LC)
```

Ideal LC energy exchange:

```text
E_total = 1/2 C U_max^2 = 1/2 L I_max^2
W_C = 1/2 C U^2
W_L = 1/2 L I^2
```

Reactance:

```text
X_L = omega L = 2πfL
X_C = 1/(omega C) = 1/(2πfC)
```

Series RLC impedance:

```text
Z = sqrt(R^2 + (X_L - X_C)^2)
I = U / Z
cos(phi) = R / Z
P = U I cos(phi) = I^2 R = U^2 R / Z^2
```

At series resonance:

```text
X_L = X_C
Z = R
I_max = U/R
P = U^2/R
phi = 0
cos(phi) = 1
```

Frequency changes:

```text
X_L scales linearly with frequency.
X_C scales inversely with frequency.
```

Only execute frequency-change problems when the original/final reactance
relations are explicitly recoverable from the question.

RC circuits:

```text
tau = R C
charging: U_C = U(1 - exp(-t/RC))
charging: Q = C U(1 - exp(-t/RC))
charging: I = (U/R) exp(-t/RC)
discharging: U_C = U0 exp(-t/RC)
discharging: Q = Q0 exp(-t/RC)
discharging: I = I0 exp(-t/RC)
```

Safe shortcuts:

- after a long time in DC steady state, capacitor is open circuit
- at the initial moment for an uncharged capacitor, capacitor behaves like a
  short circuit
- after one time constant, use `e^-1` only if the problem explicitly asks for
  one time constant or gives `t = RC`

Transformer relations:

```text
U1/U2 = N1/N2
I1/I2 = N2/N1
P1 approximately equals P2 for an ideal transformer
```

## 7. Magnetic Field, Inductance, And Flux

Long solenoid magnetic field:

```text
B = μ0 n I
μ0 = 4π×10^-7 N/A^2
```

Magnetic flux:

```text
Phi = B A cos(theta)
Phi = B A when field is perpendicular to the loop area.
```

Inductor energy:

```text
W_L = 1/2 L I^2
```

Induced emf:

```text
|emf| = N |dPhi/dt|
|emf| = L |dI/dt|
|emf| = B l v
```

Lorentz and wire magnetic force:

```text
F = |q| v B sin(theta)
F = B I l sin(theta)
r = m v/(|q|B) for circular motion perpendicular to B
```

Capacitor field energy density and plate force:

```text
u = 1/2 epsilon E^2
F_plate = 1/2 epsilon A (U/d)^2
```

## 8. Circuit Topology And Constraints

Topology patterns to support deterministically:

- `pure_series`
- `pure_parallel`
- `series_parallel_nested`
- `balanced_wheatstone_bridge`
- `symmetric_split`

Constraint registry:

- `series_current_equal`
- `parallel_voltage_equal`
- `isolated_capacitor_charge_conserved`
- `connected_capacitor_voltage_fixed`
- `ideal_ammeter_zero_resistance`
- `ideal_voltmeter_infinite_resistance`
- `steady_state_capacitor_open`
- `steady_state_inductor_short`

Validation rules:

- reject negative resistance, capacitance, distance, frequency, area, or mass
- reject zero Coulomb distance
- reject division by zero
- reject DC executor when AC cues appear: `rms`, `phase`, `sinusoidal`,
  `omega`, `reactance`, `impedance`, `cos(phi)`, `resonance`
- require `geometry_recoverable = true` for vector geometry execution
  Missing angle, ambiguous orientation, or unspecified signs must return
  `Uncertain`.

The current deterministic executor only uses the perpendicular flux and
long-solenoid direct patterns; angle/time-varying induction requires explicit
template support before it can return a verified answer.

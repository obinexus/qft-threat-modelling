# Quantum Dimensional Game Theory for QFT Threat Modeling

[![OBINexus](https://img.shields.io/badge/OBINexus-Quantum%20Security-purple)](https://obinexus.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)
[![Toolchain](https://img.shields.io/badge/Toolchain-riftlang→gosilang-orange)](docs/toolchain.md)

## Overview

This repository implements a revolutionary approach to quantum threat modeling by integrating **Dimensional Game Theory (DGT)** with **Quantum Field Theory (QFT)** principles. Using Feynman diagram formalism, we model cyber threats as quantum particles, vulnerabilities as interaction vertices, and system states as propagators in a 3D lattice security framework.

**Key Innovation**: Threats are treated as coherent mathematical objects in variadic dimensional space, enabling provable safety guarantees through the multiplicative safety function `S = Gx · Gy · Gz`.

## Purpose

Traditional threat modeling fails to capture the quantum nature of modern attacks, especially side-channel vulnerabilities in quantum systems. This framework addresses:

1. **Quantum Side-Channel Attacks**: Timing, power, and EM emanation analysis
2. **Multi-Stage Attack Chains**: Supply chain and cascading vulnerabilities
3. **Post-Quantum Cryptography**: Lattice-based threat assessment
4. **Real-Time Mitigation**: Node Zero ZKP integration for dynamic gate control

## Theoretical Foundation

### Quantum Field Theory Mapping

| QFT Concept | Cybersecurity Analog |
|-------------|---------------------|
| Particle | Threat agent or vulnerability |
| Propagator | Information/exploit transmission |
| Vertex | Interaction point (exploit attempt) |
| Coupling constant | System susceptibility (g_sys) |
| Amplitude | Incident probability |

### 3D Lattice Threat Model

```
T(x,y,z) = αX_qa + βY_quantum + γZ_blockchain
```

Where:
- **X_qa** ∈ [-12, 12]: Software QA verification axis
- **Y_quantum** ∈ [-12, 12]: Quantum integration axis  
- **Z_blockchain** ∈ [-12, 12]: Blockchain verification axis
- **α + β + γ = 1**: Normalized threat weights

### Safety Function

```
S = Gx · Gy · Gz
```

Where each gate G_i ∈ {0, 1} represents verification status. **Any gate failure forces S = 0**, triggering fail-safe mode.

## Repository Contents

### Core Implementation
- **`quantum-threat-implementation.py`**: Complete Python implementation
  - `QuantumThreatCalculator`: Amplitude calculations using Feynman rules
  - `SideChannelAnalyzer`: Timing and power analysis
  - `FeynmanDiagramVisualizer`: Threat diagram generation
  - `NodeZeroIntegration`: Real-time ZKP verification

### Visualizations
- **`side_channel_attack.png`**: Quantum side-channel attack Feynman diagram
- **`multi_stage_attack.png`**: Supply chain attack visualization
- **`threat_evolution.gif`**: Animated threat landscape evolution

### Documentation
- **`Vision Document_*.pdf`**: Comprehensive theoretical framework and integration guide

## Installation

```bash
# Clone repository
git clone https://github.com/obinexus/qft-threat-modelling.git
cd qft-threat-modelling

# Install dependencies
pip install -r requirements.txt

# Verify OBINexus toolchain
riftlang.exe --version  # Should show 1.0+
gosilang --check        # Should show polyglot support
```

## Usage

### Basic Threat Analysis

```python
from quantum_threat_implementation import QuantumThreatCalculator, ThreatState

# Initialize calculator
calc = QuantumThreatCalculator({
    'system': 0.1,
    'leakage': 0.05,
    'quantum': 0.3
})

# Define threat state
state = ThreatState(
    x_qa=-3,        # Unverified hotfix
    y_quantum=5,    # Degraded quantum source
    z_blockchain=-2, # Pending verification
    gates=(0, 1, 1), # Gx=0 (unsafe)
    timestamp=datetime.now()
)

# Calculate threat score
threat_score = calc.calculate_threat_function(state)
print(f"Threat Score: {threat_score}")
```

### Side-Channel Attack Detection

```python
from quantum_threat_implementation import SideChannelAnalyzer

analyzer = SideChannelAnalyzer()

# Analyze timing trace
timing_trace = [0, 1e-9, 4e-9, 6e-9, 11e-9]  # Quantum gate timings
results = analyzer.analyze_timing_attack(timing_trace, gates=(1, 1, 1))

print(f"Identified gates: {results['identified_gates']}")
print(f"Threat level: {results['threat_level']}")
```

### Generate Feynman Diagrams

```python
from quantum_threat_implementation import FeynmanDiagramVisualizer

visualizer = FeynmanDiagramVisualizer()
visualizer.draw_side_channel_attack()
plt.savefig('my_threat_diagram.png')
```

### Node Zero Integration

```python
import asyncio
from quantum_threat_implementation import NodeZeroIntegration

async def verify_system():
    node_zero = NodeZeroIntegration()
    
    # Create identities
    await node_zero.create_identity("system")
    await node_zero.create_identity("verifier")
    
    # Challenge-response protocol
    challenge = await node_zero.challenge("verifier", "system", {"axis": "x"})
    proof = await node_zero.generate_proof(challenge['id'], "system")
    result = await node_zero.verify_proof(proof['id'])
    
    if result['valid']:
        await node_zero.update_gate('x', 1.0)
        print("System verified - Gate X activated")

asyncio.run(verify_system())
```

## Dimensional Game Theory Integration

The framework implements variadic dimensional reduction:

```
Active Dimensions ≤ Θ (computational threshold)
```

When threats activate too many dimensions, automatic reduction ensures tractability while maintaining security guarantees.

### Example: Multi-Dimensional Attack

```python
# Attack vector spans multiple dimensions
dimensions = ['timing', 'power', 'electromagnetic', 'acoustic']

# DGT automatically reduces to top 3 most relevant
active_dims = dgt_reduce(dimensions, threshold=3)
# Result: ['timing', 'power', 'electromagnetic']
```

## OBINexus Toolchain Integration

This project integrates with the full OBINexus polyglot toolchain:

```
riftlang.exe → .so.a → rift.exe → gosilang
     ↓           ↓         ↓          ↓
[Propagators][Lattice][Policies][Coordinator]
```

### Build Process

```bash
# Complete build pipeline
./build.sh

# Individual components
riftlang.exe --input threats.rift --output propagators.so.a
rift.exe --gates "Gx,Gy,Gz" --safety-threshold 3.0
gosilang build --target threat-analyzer --polyglot
nlink --polybuild-config polybuild.toml
```

## Mathematical Formalism

### Feynman Rules

1. **Threat Propagator**: `i/(p² - m_T² + iε)`
2. **Vulnerability Propagator**: `iΔ_V(p)/(p² - m_V² + iε)`
3. **Quantum Propagator**: `iη_μν/p² · exp(-λt)`
4. **Vertex Factor**: `ig_sys · S`

### Amplitude Calculation

```
M = Π(propagators) × Π(vertices) × S
```

### Incident Rate (Fermi's Golden Rule)

```
Γ_fi = 2π|M|²ρ(E_f)
```

## Security Guarantees

1. **Fail-Safe Design**: S = 0 when any verification fails
2. **Post-Quantum**: Lattice-based cryptography throughout
3. **Real-Time Mitigation**: Sub-second Node Zero verification
4. **#NoGhosting Compliance**: All states traceable and verifiable

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Key principles:
- Maintain mathematical rigor
- Preserve fail-safe guarantees
- Document threat models clearly
- Include unit tests for amplitudes

## Patents & Publications

This work is based on:
- "Dimensional Game Theory for Fault-Tolerant Systems" (OBINexus, 2025)
- "Quantum Field Theory Decision Framework" (Patent pending)
- "The RIFT Architecture: Quantum Determinism Through Governed Computation"

## License

MIT License - See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{obinexus2025qft,
  title={Quantum Dimensional Game Theory for QFT Threat Modeling},
  author={Okpala, Nnamdi},
  year={2025},
  organization={OBINexus Computing},
  url={https://github.com/obinexus/qft-threat-modelling}
}
```

---

**Remember**: When S = 0, the system is safe. When S = 1, verify all gates!

*"The best defense is a quantum offense - but only with proper verification."* - OBINexus Philosophy
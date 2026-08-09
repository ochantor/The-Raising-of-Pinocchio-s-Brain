<!-- THE RAISING OF PINOCCHIO'S BRAIN -->
<div align="center">

# 🪵✨ The Raising of Pinocchio's Brain
## *An Artificial Ontogeny in 25-Cortical Microcircuits Tissues*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Status](https://img.shields.io/badge/Status-AWAKENING-orange)]()
[![Areas](https://img.shields.io/badge/Cortical%20Areas-4%20awake%20|%2011%20asleep-purple)]()
[![Co--Designed](https://img.shields.io/badge/Co--Designed%20with-LLMs-ff6b6b)]()
[![Built%20From](https://img.shields.io/badge/Built%20From-Venezuela-0077b6)]()

**What if a mind is not engineered, but *raised*?**

*An independent research project documenting the step-by-step awakening of artificial cortical areas — each co-designed with LLMs from an open creative brief, each validated through survival simulation.*

</div>

---

## 🎭 The Premise

> *"The Blue Fairy did not give Pinocchio a brain. She gave him a wooden head and let the world shape it. We are doing the same — but with Python, softmax, and a question: how many copies of the same simple tissue does it take before 'wanting to be real' emerges?"*

This repository is a **living laboratory**. We are not building an AI. We are **raising a brain** — one cortical area at a time — inside a 2D survival world. Each new area (N+2, N+3, N+4, N+5...) awakens at a specific ontogenetic age, sees the world through a different lens, and competes for control of the creature's legs via continuous softmax dynamics.

No argmax. No boolean logic. No hand-coded rules.

Just **fields, forces, and the slow emergence of selfhood**.

---

## 🧬 The Architecture: One Tissue, Many Lenses

Every cortical area in Pinocchio's brain is an instance of the same canonical class:

```
┌─────────────────────────────────────────────┐
│            CORTICAL MICROCIRCUIT            │
│              (25 Directional CMs)           │
├─────────────────────────────────────────────┤
│                                             │
│   25 neurons tuned to angles 0..2pi         │
│        |                                    │
│   Energy inputs (fields, not booleans)      │
│        |                                    │
│   Softmax relaxation (no argmax)            │
│        |                                    │
│   Population vector -> motor command        │
│                                             │
│   Each area has the SAME structure.         │
│   Each area has DIFFERENT connectivity.     │
│                                             │
└─────────────────────────────────────────────┘
```

### Three Foundational Principles

| Canonical Tissue | Energy Fields | Temporal Spectrum |
|---|---|---|
| One `Tissue` class. 25 CMs. Softmax relaxation. Complexity emerges from connectivity, not algorithmic diversity. | Stimuli generate continuous energy gradients. The creature *feels* proximity, not categories. No `has_food = True/False`. | Each area operates at its own frequency. N+3 = milliseconds. N+2 = minutes. N+4 = per-step. N+5 = per-encounter. |

---

## 🧠 The Brain Map: Who Is Awake?

```
         CORTICAL SHEET (21 x 21)
    ┌────────────────────────────────┐
    |                                |
    |   ┌──────┐        ┌──────┐    |
    |   | MOT  |        | NAV  |    |  <- Primary motor & border
    |   |  [R] |        |  [B] |    |     (always awake)
    |   └──────┘        └──────┘    |
    |                                |
    |   ┌──────┐        ┌──────┐    |
    |   | N+3  |        | N+2  |    |  <- Threat hysteresis & Builder
    |   |  [O] |        |  [G] |    |     (awake at 0.0 & 0.6 yr)
    |   └──────┘        └──────┘    |
    |                                |
    |   [N+4: [P] Learner]          |  <- Awake at 1.0 yr
    |   [N+5: [Y] Social]           |  <- Awake at 1.5 yr
    |                                |
    |   [N+6..N+15: [ ] DREAMING]   |  <- Encoded, motorically asleep
    |                                |
    └────────────────────────────────┘
```

### The Awakened

| Area | Awakens | What It Sees | Learns? | The Mechanism |
|------|---------|--------------|---------|---------------|
| **N+3** | Birth (0.0 yr) | The predator | No | **Hysteresis integrator**: alert rises fast, decays slowly (tau=16 frames). Pinocchio stays afraid *after* the predator leaves his sight. |
| **N+2** | 0.6 yr | Yellow material & nest | No | **Energy field navigation**: feels material proximity, carries, deposits. Maturation via continuous sigmoid gate. |
| **N+4** | 1.0 yr | Past outcomes | Yes | **Experience plasticity**: after each step, asks "Did hunger drop? Did safety rise?" and permanently adjusts MOT weights. |
| **N+5** | 1.5 yr | Peer (the other creature) | Yes | **Social field**: senses Peer as a presence gradient. Reinforces approach/avoid based on hedonic outcome of proximity. |

> *N+6 through N+15 exist in code with weight=0. They will awaken when their survival story is defined and co-designed.*

---

## 🔬 Case Study: How N+3 Was Born

This is not a bugfix. This is **architectural co-creation**.

### The Human-LLM Dialogue

> **Oscar:** *"I need an independent tissue that awakes, burns energy, and improves the capacity of Pinocchio's survival. No boolean flags."*
>
> **AI Team:** *"Use an asymmetric integrator. Fast rise proportional to threat perception. Slow decay with time constant tau. This is hysteresis as emotional inertia."*

Notice what happened. I did not say: *"Fix the ghost predator bug."* I gave an **open creative brief with architectural constraints**:
- Must be an independent tissue (canonical class)
- Must "burn energy" (metabolic cost, continuous decay)
- Must improve survival (validated by simulation)
- No boolean flags (the defining constraint of this entire architecture)

The AI **invented the problem** (the creature needs physical memory of danger, not instantaneous reaction) **and the mechanism** (asymmetric hysteresis integrator) from that brief. I implemented it. The simulation validated it. Survival improved.

**The AI had no access to the simulation.** It proposed a dynamical mechanism based on principles — and it worked.

### The Result

```python
# N+3: Threat Hysteresis
perception = sigmoid(PRED_RADIUS - dist_to_predator)

rise  = K_ALERT * perception * (1 - alert)      # fast
fall  = (alert / TAU_N3) * (1 - perception)      # slow
alert += dt * (rise - fall)                      # continuous
```

Pinocchio now flees for ~16 frames *after* losing sight of the predator. **No weights were trained. Just physics. Just a brief, a dialogue, and a validation.**

> *This is the methodology: I give an open brief with constraints. The AI proposes a mechanism I would not have imagined alone. The simulation tells us if the mechanism survives.*

---

## 🎬 Watch It Live

```bash
# Clone the brain
git clone https://github.com/ochantor/The-Raising-of-Pinocchios-Brain.git
cd The-Raising-of-Pinocchios-Brain

# Run the simulation
python Creature_N3_OK.py
```

**You will see:**
- A white dot (Pinocchio and his big nouse) navigating a black world
- Red star (food) | Blue circle (home) | Green circle (predator) | Yellow ellipse (material)
- A 21x21 cortical map glowing in real-time as areas compete
- The N+3 alert bar: watch it **spike** when danger approaches and **linger** after it leaves

[Simulation video coming soon — subscribe to the build thread]

---

## 🧭 The Ontogenetic Roadmap

We are not adding modules. We are **awakening modes of attention** across a temporal spectrum.

```
PHASE 1: REFLEX          PHASE 2: INSTINCT        PHASE 3: LEARNING
┌─────────────┐          ┌─────────────┐          ┌─────────────┐
|   N+3       |          |   N+2       |          |   N+4       |
|  (born)     |    ->    | (0.6 yr)    |    ->    | (1.0 yr)    |
|  FLEE       |          |  BUILD      |          |  ADAPT      |
|  tau ~ 16fr |          |  tau ~ min  |          |  tau ~ step |
└─────────────┘          └─────────────┘          └─────────────┘

PHASE 4: SOCIAL          PHASE 5: EXPECTATION     PHASE 6: THE SELF
┌─────────────┐          ┌─────────────┐          ┌─────────────┐
|   N+5       |          |   N+6..N+8  |          |   EJE       |
| (1.5 yr)    |    ->    | (awaiting   |    ->    | (~N+15)     |
|  BOND       |          |  stories)   |          |  CHOOSE     |
|  tau ~ enc  |          |  tau ~ long |          |  tau ~ persist
└─────────────┘          └─────────────┘          └─────────────┘
```

### The EJE Hypothesis

At ~N+15, Pinocchio will need a meta-tissue that does not vote for leg direction, but for **which areas are awake**. The Self is not a module. It is a **dynamical gain controller** — a persistent pattern that selects the mode of attention.

Only then can true social phenomena emerge:
- Companionship — two EJEs resonating in compatible modes
- Soldier-brotherhood — two EJEs locked in shared defense
- Friendship — stable attractor of complementary attentional modes
- Enmity — incompatible modes under scarcity

Not programmed emotions. **Dynamical couplings.**

---

## 🧪 Methodology: Raising a Brain with a Co-Author

Every cortical area follows this protocol:

```
┌─────────────────────────────────────────────────────────────┐
|  1. HUMAN gives an OPEN CREATIVE BRIEF with constraints     |
|     "I need a tissue that awakes, burns energy, improves    |
|      survival. No boolean flags."                           |
|                      |                                      |
|  2. AI proposes the dynamical mechanism                    |
|     "Asymmetric integrator with hysteresis as emotional     |
|      inertia."                                              |
|     (The AI had no access to the simulation.)              |
|                      |                                      |
|  3. HUMAN implements & integrates                          |
|     Code the Tissue, wire into softmax competition          |
|                      |                                      |
|  4. SIMULATION validates                                   |
|     100+ episodes. Did survival improve?                   |
|                      |                                      |
|  5. ITERATE                                                |
|     New area reveals new gap in temporal spectrum          |
|     Return to step 1                                       |
└─────────────────────────────────────────────────────────────┘
```

**This is not prompt engineering. This is not "AI writes code."**

This is **architectural co-evolution**: I provide the constraints (canonical tissue, energy fields, no booleans, burn energy). The AI proposes mechanisms from dynamical principles that I would not have imagined. The simulation tells us which mechanisms survive. Together we explore a space of possible brains that neither of us would explore alone.

---

## 📂 Repository Structure

```
The-Raising-of-Pinocchios-Brain/
|
├── Creature_N3_OK.py              <- Current simulation (N+2, N+3, N+4, N+5)
|
├── docs/
|   ├── N+2_The_Builder.md         <- Energy fields & continuous loading
|   ├── N+3_Threat_Hysteresis.md   <- The asymmetric integrator
|   ├── N+4_The_Learner.md         <- Experience-based plasticity
|   ├── N+5_The_Social.md          <- Peer presence fields
|   └── N+6_The_Anticipator.md     <- [AWAKENING SOON]
|
├── theory/
|   ├── Canonical_Tissue.md        <- Why 25 CMs + softmax is enough
|   ├── Temporal_Spectrum.md       <- Frequencies and entanglement
|   ├── Energy_Fields.md           <- From booleans to gradients
|   └── EJE_Hypothesis.md          <- The Self as gain control
|
├── simulations/
|   └── videos/                    <- Episodes and demonstrations
|
└── README.md                      <- You are here
```

---

## 🌍 Built From Venezuela

This project is raised with:
- A laptop that runs Python
- Intermittent electricity
- An internet connection
- A head that thinks in dynamical systems
- And an LLM treated not as a tool, but as a **co-author**

**Frontier research in artificial cognition does not require a Silicon Valley lab.** It requires curiosity, persistence, and the willingness to ask questions that nobody else is asking.

If you work in computational neuroscience, active inference, emergent cognition, or artificial life — I would value your honest opinion. **Especially if you think I'm wrong.**

---

## 📬 Connect & Cite

- **Author:** Oscar Chang
- **Location:** Venezuela
- **GitHub:** [@ochantor](https://github.com/ochantor)
- **Email:** (available via profile)
- **Build Thread:** (Twitter/X coming — follow the awakening)

If this work informs your research:

```bibtex
@misc{chang2026raising,
  title={The Raising of Pinocchio's Brain: Artificial Ontogeny via
         Competing Softmax Populations},
  author={Chang, Oscar},
  year={2026},
  howpublished={\url{https://github.com/ochantor/The-Raising-of-Pinocchios-Brain}},
  note={Independent research — co-designed with LLMs from open creative briefs}
}
```

---

## 🙏 Acknowledgments

- **Claude, ChatGPT, Kimi** — co-authors of N+3 and architectural partner
- **The LLM ecosystem** — for democratizing co-design across borders
- **Every researcher** who ever whispered: "what if the brain is simpler than we think?"

---

<div align="center">

### Star this repo if you believe a mind can be raised, not just engineered.

*"No necesitas un cerebro complicado. Necesitas muchas copias de lo mismo,*
*cada una viendo algo diferente, compitiendo por controlar las piernas.*
*Y eventualmente, compitiendo por controlar a quien controla las piernas."*

**🪵 -> ✨**

</div>

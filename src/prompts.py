"""
Test prompts for the Horn experiment.

Designed to force long-form generation across different domains.
Each prompt asks for depth, which is where degeneration typically kicks in —
the model runs out of "things to say" and starts looping.

Add your own prompts here to test domain-specific behavior.
"""

DEFAULT_PROMPTS = [
    (
        "Explain the relationship between entropy and information theory, "
        "starting from Shannon's original formulation and extending to "
        "modern applications in machine learning."
    ),
    (
        "Write a detailed analysis of how gravitational lensing works, "
        "including the mathematics of light bending around massive objects "
        "and what this reveals about dark matter."
    ),
    (
        "Describe the evolution of programming paradigms from assembly "
        "language through functional programming, analyzing what each "
        "paradigm reveals about how humans think about computation."
    ),
    (
        "Analyze the economic implications of artificial general intelligence, "
        "considering labor markets, wealth distribution, geopolitical power "
        "dynamics, and potential policy responses."
    ),
    (
        "Explain the concept of emergence in complex systems, using examples "
        "from biology, physics, economics, and consciousness research to "
        "illustrate how macro-level properties arise from micro-level interactions."
    ),
    (
        "Write a comprehensive exploration of the measurement problem in "
        "quantum mechanics, covering the Copenhagen interpretation, many-worlds, "
        "decoherence theory, and objective collapse models."
    ),
    (
        "Discuss the mathematical foundations of neural networks, from "
        "perceptrons through transformers, explaining why certain architectures "
        "work better for specific problems."
    ),
    (
        "Analyze the history and philosophy of zero, tracing its development "
        "across Babylonian, Indian, and European mathematics, and explaining "
        "how a symbol for nothing transformed human thought."
    ),
    (
        "Explain how CRISPR gene editing works at the molecular level, then "
        "discuss the ethical landscape of germline editing including arguments "
        "from disability rights advocates, transhumanists, and bioethicists."
    ),
    (
        "Describe the physics of the black hole information paradox, covering "
        "Hawking radiation, the holographic principle, the firewall paradox, "
        "and recent developments involving quantum error correction."
    ),
]


# ── Domain-specific prompt sets (uncomment or extend as needed) ──

MATH_PROMPTS = [
    (
        "Derive the proof that Gabriel's Horn has finite volume but infinite "
        "surface area. Then discuss the philosophical implications — the "
        "Painter's Paradox — and how this connects to measure theory."
    ),
    (
        "Explain the Riemann Hypothesis and why it matters for the distribution "
        "of prime numbers. Walk through the key ideas from Euler's product "
        "formula through the critical strip."
    ),
]

SYSTEMS_PROMPTS = [
    (
        "Design a distributed inference system for running large language models "
        "across heterogeneous hardware nodes with different GPU capabilities. "
        "Cover load balancing, latency hiding, and fault tolerance."
    ),
    (
        "Explain how modern CPU cache hierarchies work, from L1 through L3, "
        "including coherence protocols, prefetching strategies, and how cache "
        "behavior affects algorithm design."
    ),
]

CREATIVE_PROMPTS = [
    (
        "Write an extended scene set in a cyberpunk city where an AI researcher "
        "discovers that their neural network has developed an internal geometry "
        "that mirrors the topology of spacetime. Focus on technical plausibility."
    ),
    (
        "Compose a philosophical dialogue between Leibniz and Turing about "
        "whether computation can capture the full richness of human thought. "
        "Include their actual positions but extend them to modern AI."
    ),
]

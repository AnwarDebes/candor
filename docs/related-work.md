# Related work: the literature CANDOR builds on

This note maps the prior work behind CANDOR and states plainly what I borrow from each
line and what is new. It mirrors the six-stage arc in the paper's Background section. The
full, machine-readable bibliography lives in [`../paper/references.bib`](../paper/references.bib);
this file is the human-readable map. Every component of CANDOR has antecedents. The novelty
I claim is their conjunction, operating jointly and emitted at runtime.

## The one-line thesis

For a decade interpretability has been practised as archaeology: train an opaque model,
then dig for structure after the fact. CANDOR makes legibility an architectural invariant
the model is required to satisfy, and turns the unexplained remainder into a per-forward-pass,
sound, re-checkable certificate. The map below is the intellectual lineage that makes that
possible, read as a sequence of attempts to make neural computation legible, each born from
the failure of the last.

## Stage 0: the substrate (superposition)

Networks pack many more sparse, near-linear features than they have neurons, which both
motivates a sparse legible basis and bounds how complete one can be. If computation, not
just storage, happens in superposition, no fixed-width legible channel is complete. I concede
this up front, and it is exactly why CANDOR measures the leak rather than assuming it away.

- Elhage et al. (2022), Toy Models of Superposition (`elhage2022superposition`)
- Elhage et al. (2021), A Mathematical Framework for Transformer Circuits (`elhage2021framework`)
- Olah et al. (2020), Zoom In: An Introduction to Circuits (`olah2020zoomin`)
- Park, Choe, Veitch (2024), The Linear Representation Hypothesis (`park2024linear`)
- Haenni, Mendel, Vaintrob, Chan (2024), Mathematical Models of Computation in Superposition (`hanni2024compsuperposition`)

## Stage 1: recover features post hoc (sparse autoencoders)

Dictionary learning disentangles superposition into nameable features, with a long line of
architectural refinements and open releases. By 2025 the crisis is explicit: no SAE objective
is causally faithful by construction, reconstruction error is large and behaviourally
load-bearing, and SAEs often fail to beat linear probes. CANDOR borrows the sparse typed
channel and the reconstruction objective, and departs by making the channel causal and
certified rather than post hoc, and by anchoring identity.

- Bricken et al. (2023), Towards Monosemanticity (`bricken2023monosemanticity`)
- Cunningham et al. (2023), Sparse Autoencoders Find Highly Interpretable Features (`cunningham2023sparse`)
- Templeton et al. (2024), Scaling Monosemanticity (`templeton2024scaling`)
- Gao et al. (2024), Scaling and Evaluating Sparse Autoencoders (`gao2024scaling`)
- Rajamanoharan et al. (2024), JumpReLU SAEs (`rajamanoharan2024jumprelu`)
- Bussmann, Leask, Nanda (2024), BatchTopK SAEs (`bussmann2024batchtopk`)
- Bussmann et al. (2025), Matryoshka SAEs (`bussmann2025matryoshka`)
- Lieberum et al. (2024), Gemma Scope (`lieberum2024gemmascope`)
- Chanin et al. (2024), A is for Absorption (`chanin2024absorption`)
- Chanin et al. (2025), Feature Hedging (`chanin2025hedging`)
- Leask et al. (2025), SAEs Do Not Find Canonical Units of Analysis (`leask2025canonical`)
- Kantamneni et al. (2025), Are Sparse Autoencoders Useful? (`kantamneni2025useful`)
- Karvonen et al. (2025), SAEBench (`karvonen2025saebench`)

## Stage 2: trace computation (transcoders, attribution graphs)

Transcoders approximate a layer's input-output map; cross-layer transcoders plus replacement
models yield per-prompt attribution graphs. This is the closest prior art to "the explanation
is the forward pass." Its limits (unbounded error nodes, per-prompt cost, post-hoc construction)
are exactly what a runtime certificate must fix. CANDOR routes into the trained forward pass
and bounds the error-node mass at runtime.

- Dunefsky, Chlenski, Nanda (2024), Transcoders Find Interpretable Circuits (`dunefsky2024transcoders`)
- Marks et al. (2024), Sparse Feature Circuits (`marks2024sparse`)
- Ameisen et al. (2025), Circuit Tracing (`ameisen2025circuit`)
- Lindsey et al. (2025), On the Biology of a Large Language Model (`lindsey2025biology`)
- Wang et al. (2022), Indirect Object Identification in GPT-2 Small (`wang2022ioi`)
- Olsson et al. (2022), In-context Learning and Induction Heads (`olsson2022induction`)
- Miller, Chughtai, Saunders (2024), Circuit Faithfulness Metrics Are Not Robust (`miller2024faithfulnessnotrobust`)

## Stage 3: make faithfulness causal

The field converges on intervention as the only credible test: activation and path patching,
automated discovery, causal scrubbing, and the unifying theory of causal abstraction.
Decisively, faithfulness can be a training objective via interchange intervention training and
distributed alignment search. This is CANDOR's core borrowed engine. CANDOR generalises it
from aligning a given causal variable to co-training a typed, persistent, complete bottleneck
plus a runtime certificate; it does not invent training-time causal faithfulness.

- Geiger et al. (2021), Causal Abstractions of Neural Networks (`geiger2021causalabstractions`)
- Geiger et al. (2022), Interchange Intervention Training (`geiger2022iit`)
- Geiger et al. (2024), Distributed Alignment Search (`geiger2024das`)
- Wu et al. (2023), Interpretability at Scale / Boundless DAS (`wu2023boundlessdas`)
- Chan et al. (2022), Causal Scrubbing (`chan2022causalscrubbing`)
- Goldowsky-Dill et al. (2023), Localizing Model Behavior with Path Patching (`goldowskydill2023pathpatching`)
- Conmy et al. (2023), Towards Automated Circuit Discovery (`conmy2023acdc`)
- Syed, Rager, Conmy (2023), Attribution Patching Outperforms ACDC (`syed2023eap`)
- Zhang, Nanda (2024), Best Practices of Activation Patching (`zhang2024bestpractices`)
- Geiger et al. (2025), Causal Abstraction: A Theoretical Foundation (`geiger2025causalabstraction`)
- Makelov, Lange, Nanda (2023), An Interpretability Illusion for Subspace Patching (`makelov2023illusion`)

## Stage 4: build legibility in (and its leakage)

Force computation through a privileged basis: concept bottleneck models, self-explaining
networks, prototypes, B-cos alignment, codebooks, and the information bottleneck, plus the
call to use inherently interpretable models for high-stakes decisions. The recurring failure
is concept leakage: the bottleneck silently re-becomes a black box, and identifiability is
unattainable without inductive bias. CANDOR is a concept bottleneck hardened with the
guarantees this line proved necessary. Leakage is dark computation under another name, and the
certificate is what bounds it.

- Koh et al. (2020), Concept Bottleneck Models (`koh2020concept`)
- Espinosa Zarlenga et al. (2022), Concept Embedding Models (`espinosa2022concept`)
- Alvarez-Melis, Jaakkola (2018), Self-Explaining Neural Networks (`alvarezmelis2018senn`)
- Chen et al. (2019), This Looks Like That / ProtoPNet (`chen2019thislooks`)
- Boehle, Fritz, Schiele (2022), B-cos Networks (`bohle2022bcos`)
- van den Oord et al. (2017), Neural Discrete Representation Learning / VQ-VAE (`oord2017vqvae`)
- Tishby, Zaslavsky (2015), Deep Learning and the Information Bottleneck (`tishby2015deep`)
- Rudin (2019), Stop Explaining Black Box Models for High-Stakes Decisions (`rudin2019stop`)
- Mahinpei et al. (2021), Promises and Pitfalls of Black-Box Concept Learning (`mahinpei2021promises`)
- Havasi, Parbhoo, Doshi-Velez (2022), Addressing Leakage in Concept Bottleneck Models (`havasi2022addressing`)
- Locatello et al. (2019), Challenging Common Assumptions in Disentanglement (`locatello2019challenging`)

## Stage 5: certify

Neural-network verification is mature for robustness, but robustness is not interpretability.
Formal XAI gives per-input logical guarantees; interpretable-by-construction-and-verified
models exist only at small scale. No prior method emits a sound, cheap, per-forward-pass bound
on bypassed computation. CANDOR borrows the sound-over-approximation form and the safety-case
framing, and supplies the runtime certificate that monitoring and chain-of-thought approaches
cannot, moving the certificate from the model's outputs to the model's own computation.

- Katz et al. (2017), Reluplex (`katz2017reluplex`)
- Zhang et al. (2018), CROWN (`zhang2018crown`)
- Wang et al. (2021), Beta-CROWN (`wang2021betacrown`)
- Xu et al. (2020), auto_LiRPA (`xu2020autolirpa`)
- Ignatiev, Narodytska, Marques-Silva (2019), Abduction-Based Explanations (`ignatiev2019abduction`)
- Przybysz et al. (2023), Verifying Properties of Tsetlin Machines (`przybysz2023verifying`)
- Granmo (2018), The Tsetlin Machine (`granmo2018tsetlin`)
- Lin et al. (2020), GOSDT (`lin2020gosdt`)

## Oversight, evaluation, and the case for runtime guarantees

CANDOR's certificate is meant to be an artefact a safety case can rest on, and a constructive
answer to eliciting latent knowledge and mechanistic-anomaly detection: dark computation is
surfaced as a number rather than hidden in an error node. The chain-of-thought line shows why
a measurement beats a learned monitor: optimisation pressure makes learned monitors actively
misleading, while a measured certificate cannot be Goodharted into silence.

- Turpin et al. (2023), Unfaithful CoT Explanations (`turpin2023language`)
- Chen et al. (2025), Reasoning Models Don't Always Say What They Think (`chen2025reasoning`)
- Baker et al. (2025), Monitoring Reasoning Models for Misbehavior (`baker2025monitoring`)
- Korbak et al. (2025), Chain of Thought Monitorability (`korbak2025cot`)
- Christiano, Cotra, Xu (2021), Eliciting Latent Knowledge (`christiano2021elk`)
- Roger et al. (2023), Benchmarks for Detecting Measurement Tampering (`roger2023measurement`)
- Jacovi, Goldberg (2020), Towards Faithfully Interpretable NLP Systems (`jacovi2020faithfully`)
- Clymer et al. (2024), Safety Cases (`clymer2024safetycases`)
- Amodei (2025), The Urgency of Interpretability (`amodei2025urgency`)
- European Union (2024), EU AI Act, Article 13 (`euaiact2024`)

## Where CANDOR sits

CANDOR is the conjunction of Stages 3, 4, and 5 performed jointly and emitted at runtime: a
sparse, typed, anchored concept channel (Stage 4) that is co-trained to be causally sufficient
(Stage 3) and that emits a sound, re-checkable Faithfulness Certificate on every forward pass
(Stage 5). I make no completeness claim, the certificate is a sound measurement and not a
scale proof, and the positive evidence is on synthetic ground truth and a one-layer Transformer
rather than a frontier model. The paper states each borrowed ingredient and each boundary
explicitly.

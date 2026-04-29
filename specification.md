Here is the specification and plan formatted in a Markdown codeblock so you can easily copy and paste it:

# Software Requirements Specification and Incremental Implementation Plan for Coarse-Grained Transformer-Based Protein Structure Prediction

## 1. Introduction and Architectural Philosophy

Determining the three-dimensional topology of a protein strictly from its one-dimensional amino acid sequence remains one of the most formidable computational challenges in modern biophysics.[1] For decades, structural biology has depended on experimental methodologies such as X-ray crystallography, nuclear magnetic resonance (NMR) spectroscopy, and cryogenic electron microscopy (cryo-EM).[1, 2] While these methods provide ground-truth atomic coordinates, they are severely bottlenecked by high financial costs, low throughput, and an inherent inability to crystallize highly dynamic or intrinsically disordered proteins.[1] 

The introduction of all-atom deep learning frameworks, most notably AlphaFold2, revolutionized this domain by achieving near-experimental accuracy.[1] These systems primarily succeed by extracting co-evolutionary signals from Multiple Sequence Alignments (MSAs), effectively using evolutionary history to infer structural proximity.[1] However, this MSA-dependent architecture carries massive computational overhead and fundamentally struggles when presented with orphan proteins, rapidly mutating viral targets, or synthetic *de novo* designed sequences that lack evolutionary homologs entirely.[1, 3] 

To circumvent this evolutionary dependency and the computational burden of all-atom models, an emerging paradigm advocates for single-sequence methodologies operating over abstract, coarse-grained geometric representations.[1, 4] By utilizing the global receptive field of self-attention mechanisms inherent to Transformer architectures, it is possible to bypass the autoregressive limitations of Recurrent Neural Networks (RNNs) and predict relative geometries directly.[1, 5]

### 1.1 Purpose of the Specification

The purpose of this document is to establish a comprehensive Software Requirements Specification (SRS) and Implementation Plan for a novel neural network architecture designed to predict the tertiary structure of coarse-grained proteins directly from single amino acid sequences. This document strictly conforms to the structural intent of the IEEE 830-1998 standard and the modernized ISO/IEC/IEEE 29148 standard, which define the full requirements engineering lifecycle from elicitation through validation.[6, 7] 

Crucially, this SRS is architected to respect a rigid incremental extension plan.[1] Because deep learning algorithms applied to structural biology often struggle to converge when forced to learn complex physical rules simultaneously with feature extraction, attempting to build a monolithic architecture usually results in vanishing gradients and training collapse.[3, 8] Therefore, this specification isolates the system into maskable modules. This allows development teams to build, verify, and baseline the core Transformer architecture independently, while preserving seamless, configuration-driven integration pathways for future biophysical enhancements, such as differentiable physics engines and global coordinate loss functions.[1, 9]

### 1.2 Document Conventions and Masking Strategy

To support incremental development, the requirements detailed herein are categorized by an implementation phase tag (Phase 1, Phase 2, or Phase 3). The software design utilizes a feature-toggled, component-based masking strategy.[10, 11] 

*   **Phase 1 (Core Baseline):** Represents the minimum viable architecture. It maps sequences to continuous local angles using Absolute Positional Encoding and localized trigonometric loss.[1] Phase 2 and Phase 3 requirements must be programmatically masked out of the compilation and runtime execution during Phase 1 deployment to ensure training stability.[1, 11]
*   **Phase 2 (Post-Processing):** Represents validation mechanics. It translates angular predictions into 3D Cartesian space via external algorithms outside the active backpropagation loop.[1]
*   **Phase 3 (Structural Extensions):** Represents deep algorithmic interventions inside the training loop, including Relative Positional Encoding (RPE), global dRMSD loss, and differentiable Lennard-Jones steric clash penalties.[1]

### 1.3 Intended Audience

This document is intended for a cross-functional audience of computational biologists, deep learning engineers, and high-performance computing (HPC) systems architects.[12] Developers will reference it for software design and tensor dimensionalities, test engineers will derive validation plans based on the traceability matrices, and biophysicists will utilize it to verify that the mathematical loss functions accurately represent underlying macromolecular mechanics.[7, 12]

## 2. Overall System Description and Context

### 2.1 Product Perspective

The proposed predictive system operates as a standalone computational pipeline deployed within a high-performance computing environment equipped with GPU acceleration.[13] It is not a follow-on to AlphaFold; rather, it is a divergence intended for rapid, single-sequence coarse-grained evaluation.[1] The system relies heavily on the PyTorch ecosystem for autograd-based tensor manipulation and deep learning mechanics.[1, 14]

The primary external data interface is the SidechainNet library, an all-atom protein structure dataset designed specifically for machine learning that extends the original ProteinNet dataset.[14, 15] By interfacing natively with SidechainNet, the software avoids the extensive, error-prone custom parsing algorithms traditionally required to ingest raw `.pdb` files from the Protein Data Bank.[1] 

### 2.2 Biological and Geometric Abstractions

To achieve high computational throughput, the system employs a united-residue or coarse-grained representation of the polypeptide chain.[16] Rather than modeling every heavy atom, the polypeptide chain is reduced to a sequence of alpha-carbon ($C_{\alpha}$) atoms, and occasionally a single characteristic side-chain (SC) atom located at the centroid of the side-chain heavy atoms.[16, 17] 

Within this abstract space, the atoms are connected using virtual bonds.[16] The location and orientation of any residue $i$ in the chain relative to its neighbors can be completely defined by three internal coordinates:
1.  **Virtual Separation Distance ($d$):** The Euclidean distance between adjacent $C_{\alpha}$ atoms.[1, 16]
2.  **Virtual Bond Angle ($\theta$):** The planar angle formed by three consecutive $C_{\alpha}$ atoms.[1, 16]
3.  **Pseudotorsion Angle ($\tau$):** The dihedral angle defining the rotation out of the plane defined by four consecutive $C_{\alpha}$ atoms.[1, 16, 18]

The fundamental machine learning objective is to train a Transformer to predict $\theta$, $\tau$, and $d$ directly from the primary sequence data, establishing a latent grammar of protein folding.[1]

## 3. Phase 1 Specific Requirements: The Core Predictive Architecture

Phase 1 constitutes the foundational build. It strictly maps one-dimensional integer-encoded sequences to continuous local angles and separation distances.[1]

### 3.1 Data Ingestion Subsystem

The data ingestion subsystem is responsible for fetching, batching, and transforming protein sequence data into PyTorch-compatible tensors.[14] The system delegates the heavy lifting of dataset management to SidechainNet, treating it as the authoritative source of truth.

| Requirement ID | Requirement Description | Phase | Maskable |
| :--- | :--- | :--- | :--- |
| **REQ-DI-1.01** | The system shall utilize the `sidechainnet.load()` API to instantiate PyTorch `DataLoader` objects mapped to training, validation, and testing splits.[14, 15] | 1 | No |
| **REQ-DI-1.02** | The data loader must enforce a dynamic batching mechanism that groups proteins of similar length.[15] This prevents massive computational waste caused by excessive zero-padding in sequences of disparate sizes.[15, 19] | 1 | No |
| **REQ-DI-1.03** | The system shall ingest the one-dimensional primary amino acid sequence as an integer-encoded tensor of length $L$, representing the 20 canonical amino acids plus a masking token for missing residues.[14] | 1 | No |
| **REQ-DI-1.04** | The system must extract the ground-truth internal coordinates—specifically backbone torsion angles, backbone bond angles, and $C_{\alpha}$ spatial coordinates—from the SidechainNet `ProteinBatch` objects.[14, 15] | 1 | No |
| **REQ-DI-1.05** | The system must dynamically calculate the ground truth virtual bond angle ($\theta$), pseudotorsion ($\tau$), and virtual distance ($d$) from the raw $C_{\alpha}$ coordinates to establish the coarse-grained targets.[1, 16] | 1 | No |

#### 3.1.1 Data Structure Comparison: ProteinNet vs. SidechainNet
To justify the dependency on SidechainNet, the system acknowledges its superior dimensionality for structural metrics. SidechainNet explicitly includes oxygen atoms as part of the backbone coordinate data, whereas legacy tools like ProteinNet omitted them.[14] The following table outlines the tensor dimensionalities natively ingested by the pipeline.

| Attribute Name | Dimensionality ($L$ = sequence length) | Present in ProteinNet | Present in SidechainNet |
| :--- | :--- | :--- | :--- |
| Primary Sequence | $L$ | Yes | Yes |
| Missing Residue Mask | $L$ | Yes | Yes |
| Backbone Coordinates | $L \times 4 \times 3$ | No (Only $L \times 3 \times 3$) | Yes (Includes Oxygen) |
| Backbone Torsion Angles | $L \times 3$ | No | Yes |
| Backbone Bond Angles | $L \times 3$ | No | Yes |
| Evolutionary PSSMs | $L \times 21$ | Yes | Yes |

*Source: SidechainNet Documentation.[14, 15] Note: The core pipeline will selectively slice this data to isolate the $C_{\alpha}$ trace for the coarse-grained formulation.*

### 3.2 Transformer Neural Network Subsystem

The neural network replaces traditional Recurrent Neural Networks (RNNs). RNNs process sequences one amino acid at a time, sequentially passing gradient information step-by-step. This architecture suffers from the "bucket brigade" problem, wherein gradients vanish or explode over long sequences, severely diminishing the network's capacity to recognize the long-range tertiary interactions that drive protein folding.[1] 

The Transformer bypasses this by analyzing the entire sequence simultaneously through self-attention, enabling the direct correlation of distant amino acids.[1]

| Requirement ID | Requirement Description | Phase | Maskable |
| :--- | :--- | :--- | :--- |
| **REQ-NN-1.01** | The system shall implement a multi-layer self-attention encoder utilizing `torch.nn.TransformerEncoder` as the foundational architecture.[1, 19] | 1 | No |
| **REQ-NN-1.02** | The network shall project the integer-encoded sequence tokens into a dense continuous embedding space of dimension $D_{model}$ via a `torch.nn.Embedding` lookup table.[20] | 1 | No |
| **REQ-NN-1.03** | The system must inject Absolute Sinusoidal Positional Encoding into the sequence embeddings prior to the first attention block.[1] Because attention is natively permutation invariant, absolute encoding ensures the network comprehends the linear physical order of the polymer chain.[1, 21] | 1 | No |
| **REQ-NN-1.04** | The final linear projection layers of the network must map the $D_{model}$ hidden states to a localized continuous parameter space representing the internal geometry of the protein.[1, 16] | 1 | No |

### 3.3 Output Representation and Localized Loss Subsystem

A critical engineering challenge in angle prediction involves circular discontinuity. An angle of $-179^{\circ}$ and an angle of $179^{\circ}$ are physically only $2^{\circ}$ apart. However, naively applying a standard Mean Squared Error (MSE) loss function to the raw scalar values would penalize the model for a mathematically massive $358^{\circ}$ difference, generating false gradients that prevent network convergence.[22] 

To resolve this, the Phase 1 architecture maps the angles to the unit circle.[1] 

| Requirement ID | Requirement Description | Phase | Maskable |
| :--- | :--- | :--- | :--- |
| **REQ-LF-1.01** | The Transformer's output projection must yield a feature vector $\mathbf{v}_i = [\hat{x}_{\theta}, \hat{y}_{\theta}, \hat{x}_{\tau}, \hat{y}_{\tau}, \hat{d}]$ for every residue $i$.[1, 22] | 1 | No |
| **REQ-LF-1.02** | The system shall enforce an internal $L_2$ normalization layer to the trigonometric pairs $(\hat{x}, \hat{y})$ to ensure they represent valid coordinates on the unit circle corresponding to the sine and cosine of the target angle.[22, 23] | 1 | No |
| **REQ-LF-1.03** | The localized training loss function shall evaluate the Mean Squared Error (MSE) directly on the sine and cosine components of the predicted angles.[1, 23] The formula implemented must be: $MSE_{trig} = \frac{1}{n} \sum ((\sin(\theta_{pred}) - \sin(\theta_{true}))^2 + (\cos(\theta_{pred}) - \cos(\theta_{true}))^2)$.[1, 23] | 1 | No |
| **REQ-LF-1.04** | The total Phase 1 training loss must combine the trigonometric MSE with the MSE of the predicted separation distance, utilizing a configurable hyperparameter $\lambda$ to balance the gradients: $Loss_{total} = MSE_{trig} + \lambda \cdot MSE_{distance}$.[1] | 1 | No |

By constraining the Phase 1 training loop strictly to this localized continuous loss function, the network trains with high computational efficiency, free from the heavy processing overhead of Cartesian coordinate reconstruction.[1]

## 4. Phase 2 Specific Requirements: Post-Processing and Spatial Reconstruction

Once the Phase 1 baseline model successfully maps sequences to local angles, validation issues emerge. Raw trigonometric vectors cannot be evaluated visually by biologists.[1] To translate these internal coordinates into physical 3D space, the system relies on the Natural Extension Reference Frame (NeRF) algorithm.[1] 

Phase 2 specifies the development of an isolated post-processing module. Crucially, in Phase 2, NeRF operates strictly *outside* the active training loop.[1] It does not pass gradients back to the model; it merely validates the forward-pass predictions.

### 4.1 Natural Extension Reference Frame (NeRF) Subsystem

The NeRF algorithm converts the parameterization of polymers from internal coordinates (bond lengths, angles, and torsions) to Cartesian coordinates.[24] Standard NeRF sequentially calculates the position of the next atom utilizing the positions of three previous atoms combined with the newly predicted bond length, bond angle, and torsion angle.[25] 

However, sequential NeRF poses a severe computational bottleneck when executing inference on large batches of proteins.[26]

| Requirement ID | Requirement Description | Phase | Maskable |
| :--- | :--- | :--- | :--- |
| **REQ-PP-2.01** | The system shall integrate a parallelized NeRF Python library, specifically `pNeRF` or `MP-NeRF`, to mitigate the sequential processing bottleneck during structure reconstruction.[1, 24, 26] | 2 | Yes |
| **REQ-PP-2.02** | The integrated MP-NeRF module must execute reconstruction through three main phases: (1) parallel composition of the minimal repeated backbone structure, (2) assembly of backbone monomers via efficient roto-translation operations, and (3) parallel elongation of sidechains (if applicable to the specific coarse-grained trace).[26, 27] | 2 | Yes |
| **REQ-PP-2.03** | The system shall provide an export interface that pipes the fully reconstructed all-atom or $C_{\alpha}$ Cartesian coordinates into standard biological file formats, specifically `.pdb` and `.gltf`.[1, 15, 28] | 2 | Yes |
| **REQ-PP-2.04** | The exported structural files must be natively compatible with open-source visualization platforms such as PyMOL and py3Dmol for immediate downstream validation.[1, 15, 28] | 2 | Yes |

### 4.2 The "Lever-Arm" Diagnostic Module

Because the Phase 1 model optimizes local angles independently, it is highly susceptible to the "lever-arm effect." The lever-arm effect is a geometric phenomenon where minuscule angular deviations at the beginning of a polymer chain compound exponentially as the chain extends, resulting in massive spatial inaccuracies at the opposite terminus.[1]

| Requirement ID | Requirement Description | Phase | Maskable |
| :--- | :--- | :--- | :--- |
| **REQ-PP-2.05** | The post-processing module must include a diagnostic logging system that calculates the divergence between localized angular accuracy (low Trig-MSE) and global spatial inaccuracy (high spatial deviation) to empirically quantify the severity of the lever-arm effect in the Phase 1 model.[1] | 2 | Yes |

## 5. Phase 3 Specific Requirements: Advanced Structural Extensions

Phase 3 introduces invasive modifications to the core architecture to address the fundamental flaws revealed during Phase 1 and Phase 2. Absolute Positional Encoding limits length generalization, the lever-arm effect destroys global topology, and unconstrained angular models frequently predict physically impossible steric overlaps.[1] 

These advanced features alter the attention mechanism and actively pass gradients through physical simulators during backpropagation. To maintain incremental stability, Phase 3 requirements must be governed by a configuration registry (e.g., `config.yaml`), allowing them to be toggled independently without breaking the underlying Phase 1 framework.[10, 11]

### 5.1 Relative Positional Encoding (RPE) Subsystem

Absolute Positional Encodings inject fixed indices into the sequence, causing the network to memorize absolute positions and biasing it toward the sequence lengths represented in the training data.[1, 20] By contrast, Relative Positional Encoding (RPE) considers the information between pairwise positions dynamically.[29] It signals the network regarding the distance offsets between tokens, which guarantees length-invariant sequence modeling and theoretical extrapolation to unseen, longer chains.[1, 20]

| Requirement ID | Requirement Description | Phase | Maskable |
| :--- | :--- | :--- | :--- |
| **REQ-EX-3.01** | The architecture must support dynamic swapping between the baseline Absolute Positional Encoding module and a newly injected Relative Positional Encoding mechanism.[1, 21] | 3 | Yes |
| **REQ-EX-3.02** | The RPE implementation must eschew adding positional information element-wise directly to the token embeddings. Instead, relative positional information must be added on the fly to the `keys` and `values` during the attention calculation.[29] | 3 | Yes |
| **REQ-EX-3.03** | The system must implement memory-optimized RPE algorithms. The original Shaw et al. formulation requires an additional $O(L^2 D)$ in memory, which is computationally prohibitive for long proteins.[29, 30] The system shall utilize optimized formulations (e.g., Rotary Positional Embeddings - RoPE, or Music Transformer variants) that extrapolate longer sequences via geometric progressions of angles while preserving vector norms.[20, 29] | 3 | Yes |

### 5.2 Global End-to-End Coordinate Loss (dRMSD) Subsystem

To correct the lever-arm effect, the network must be forced to optimize global distance metrics dynamically inside the active training loop.[1] This requires pulling the NeRF reconstruction algorithm into the PyTorch autograd graph, transforming it from a post-processing script into a differentiable layer.[31] 

Once differentiable, the system minimizes the distance-based Root Mean Square Deviation (dRMSD).[32]

| Requirement ID | Requirement Description | Phase | Maskable |
| :--- | :--- | :--- | :--- |
| **REQ-EX-3.04** | The system must wrap the parallel MP-NeRF algorithm in PyTorch differential primitives, maintaining a continuous chain of gradients from the Cartesian coordinates back to the Transformer weights.[26, 31] | 3 | Yes |
| **REQ-EX-3.05** | The loss function module must dynamically compute the pairwise distance matrix $D_s(F(S))$ for the newly reconstructed predicted coordinates and compare it to the ground-truth experimental distance matrix $D_s(X)$.[33] | 3 | Yes |
| **REQ-EX-3.06** | The system shall compute the dRMSD using the formulation $\ell_{fold} = \sqrt{\frac{1}{n_M} \| M \odot (D_s(F(S)) - D_s(X)) \|_2^2}$, where $M$ is a masking matrix.[33] | 3 | Yes |
| **REQ-EX-3.07** | The masking matrix $M$ must filter out zero elements corresponding to missing data in the native structure, as well as apply a spatial threshold cutoff.[33] Naively minimizing the $L_2$ distance across all atomic pairs causes the loss function to over-focus on large-scale distances at the expense of local structure. The system shall mimic AlphaFold and RGN algorithms by applying a default 22.8Å cutoff threshold, above which pairwise distances are masked out of the loss calculation.[33] | 3 | Yes |
| **REQ-EX-3.08** | **Curriculum Learning Integration:** The system must not initiate training exclusively on dRMSD. Backpropagating through the sequential NeRF reconstruction algorithm induces severe vanishing/exploding gradients in an uninitialized network.[8] The loss function must dynamically interpolate from the localized Phase 1 trigonometric MSE loss toward the Phase 3 global dRMSD loss as epochs progress.[1, 8] | 3 | Yes |

### 5.3 Differentiable Physics Penalty Subsystem

Traditional machine learning algorithms applied to structural biology struggle when forced to infer complex physical rules purely from sparse data, often resulting in overfitting and structural violations.[3] Predicting physically impossible proteins with overlapping amino acids is a hallmark failure of unconstrained deep learning models.[1] 

To solve this, Phase 3 defines the integration of a set of deterministic biophysical rules—a 'force-field'—directly into the deep learning algorithm, freeing the neural network from the burden of deducing basic physics.[3] Because state-of-the-art force fields (Rosetta, CHARMM) are typically written in non-differentiable languages, the system requires a custom PyTorch-native implementation.[3, 34]

| Requirement ID | Requirement Description | Phase | Maskable |
| :--- | :--- | :--- | :--- |
| **REQ-EX-3.09** | The system shall implement an end-to-end differentiable physical simulator module (e.g., inspired by MadraX, TorchMD, or OpenMM-Loss) that evaluates the potential energy of the predicted coordinates.[3, 35, 36] | 3 | Yes |
| **REQ-EX-3.10** | The simulator must enforce a steric clash penalty modeled after the Lennard-Jones (LJ) 6-12 potential, penalizing predictions where atoms overlap.[1, 34, 37] | 3 | Yes |
| **REQ-EX-3.11** | **Repulsive Quenching / Gradient Clipping:** Severe initial clashes yield astronomically high values for the $r^{-12}$ repulsive term of the Lennard-Jones potential.[37, 38] If passed unconstrained into the autograd graph, this results in massively exploding gradients, destroying the network weights.[38] The physical simulator must dynamically cap, scale, or quench the repulsive potential forces during early iterations, acting akin to high heat exchange rates in Langevin dynamics.[38, 39] | 3 | Yes |

## 6. Non-Functional Requirements

To ensure the system is not only theoretically sound but functionally deployable, a series of non-functional requirements govern the code execution and infrastructure dependencies.

### 6.1 Performance and Hardware Scalability
The underlying goal of coarse-graining is to expand the tractable length and time scales by 2–3 orders of magnitude compared to all-atom representations.[4] 
*   **Throughput:** By reducing the coordinate calculations to a $C_{\alpha}$ trace, the memory footprint per sequence must be small enough to allow the Transformer to natively process full sequences exceeding 1,000 amino acids on standard GPU hardware (e.g., NVIDIA A100/H100 instances) without encountering Out-Of-Memory (OOM) exceptions.[1, 15]
*   **Hardware Acceleration:** All custom tensor operations, specifically the Phase 3 differentiable physical simulators and MP-NeRF roto-translations, must execute via optimized `torch.cuda` backends.[15, 36] 

### 6.2 Modularity and Maintainability
*   **Configuration Management:** The deployment architecture must isolate core and extension dependencies.[10] Executing the Phase 1 baseline architecture must not require the environment to compile heavy external MD simulation suites (like OpenMM or CHARMM bindings).[14]
*   **Reproducibility:** A centralized random seed must govern the initialization of model weights, dynamic data loader batching mechanisms, and stochastic physical perturbations (like Langevin noise) to ensure that academic benchmarking metrics (like pLDDT or dRMSD) remain completely deterministic and reproducible across distributed training runs.[32, 39]

### 6.3 Security, Backup, and Reliability
While the system is a research pipeline, basic infrastructure integrity must be maintained.
*   **Data Integrity:** The system must execute incremental backups of model checkpoints daily and full backups of the tensor databases weekly to geographically separate cloud instances.[40]
*   **Recovery Objective:** The infrastructure shall maintain a Recovery Time Objective (RTO) of 4 hours for training job resumption in the event of hardware failure or catastrophic gradient explosions.[40]

## 7. Traceability and Requirements Mapping

In alignment with standard requirements engineering practices, the following traceability matrix maps high-level biological and algorithmic objectives directly to the verifiable software requirements defined in the sections above.[6, 11] This ensures that no code is developed without a corresponding biophysical rationale.

| Biological / Project Objective | Addressed By | Software Verification Method |
| :--- | :--- | :--- |
| Bypass MSA dependency [1] | REQ-NN-1.01 (Self-Attention) | System runs inference on orphan proteins successfully. |
| Overcome RNN bucket-brigade [1] | REQ-NN-1.01 (Transformer) | Attention weights map long-range tertiary contacts. |
| Prevent discontinuous gradients [22] | REQ-LF-1.01 to 1.03 | Loss converges without spiking at -180/180 degrees. |
| Ensure Incremental Masking [1] | Configuration Registry (Sec 1.2) | Phase 1 runs independent of NeRF/Physics imports. |
| Solve the Lever-Arm effect [1] | REQ-EX-3.05 (dRMSD Loss) | Global validation score improves over epochs. |
| Prevent Steric Overlaps [1] | REQ-EX-3.10 (Lennard-Jones) | Final PDB structures exhibit no atomic clashes. |
| Prevent gradient explosions [38] | REQ-EX-3.08 & REQ-EX-3.11 | Gradients clipped; model stabilizes during clash penalization. |

## 8. Incremental Implementation Plan

The project methodology specifically dictates an incremental implementation capable of masking complex future extensions.[1] Consequently, the development lifecycle completely abandons the monolithic "big bang" integration approach in favor of strict, staged, configuration-driven agile sprints.[10] 

### Sprint Strategy: Phase 1
The objective of the first phase is strictly limited to sequence-to-angle tensor mapping.
1.  **Environment and Data Ingestion:** Initialize the PyTorch ecosystem. Deploy the SidechainNet library.[14] The engineering team will configure the custom dynamic batching algorithms to group sequences of similar length, ensuring optimal memory layout on the GPU.[15] Data preprocessing scripts will slice the $L \times 14 \times 3$ sidechain tensors to extract the isolated $C_{\alpha}$ geometric ground truths: virtual bond angles ($\theta$), pseudotorsions ($\tau$), and separation distances ($d$).[28]
2.  **Core Model Construction:** Construct the `torch.nn.TransformerEncoder`. Integrate Absolute Sinusoidal Positional Encoding to manage linear sequence indexing.[1]
3.  **Loss Formulation:** Configure the output projection heads to emit geometric matrices. Apply the normalization layer to project predictions onto continuous sine and cosine domains, solving the $-180^{\circ}$ discontinuity.[1, 22] Implement the localized Trigonometric Mean Squared Error backpropagation loop.[23]
4.  **Verification Checkpoint:** The core architecture is completely walled off from physical spatial realities at this stage.[1] The masking system is tested to confirm the software builds and trains without any dependencies on external coordinate simulators.

### Sprint Strategy: Phase 2
The objective of the second phase is to evaluate the physical manifestation of the Transformer's raw angular predictions without altering the Phase 1 training mechanism.
1.  **NeRF Integration:** Import the massively parallelized MP-NeRF algorithms into an isolated, non-differentiable post-processing script.[26] The pipeline maps the angular predictions into spatial vectors, assembling the polymer backbone hierarchically.[27]
2.  **Visualization Pipeline:** Route the generated 3D Cartesian coordinates into SidechainNet's visualization API to generate `.pdb` and `.gltf` files.[15]
3.  **Verification Checkpoint:** Conduct extensive 3D visual validation using PyMOL.[1] The engineering team will statistically evaluate the divergence between local angle accuracy (from Phase 1) and downstream spatial drift to empirically quantify the "lever-arm effect".[1]

### Sprint Strategy: Phase 3
Phase 3 represents the final structural evolution. The system configuration toggles are activated, injecting highly sophisticated biophysical and mathematical rules directly into the active backpropagation framework.[1]
1.  **Length Bias Correction:** The Absolute Positional Encoding module is masked out and replaced with Relative Positional Encoding (RPE).[1, 29] The attention matrices are refactored to account for sequence distance offsets via Rotary Positional Embeddings (RoPE), ensuring $O(L)$ computational stability.[20, 29] Validation is executed by feeding the network artificially elongated sequences to confirm length extrapolation.[1, 20]
2.  **Global Optimization Injection:** The MP-NeRF reconstruction pipeline is absorbed directly into the PyTorch autograd graph, rendering the spatial calculations fully differentiable.[31] The optimization loop computes the pairwise distance matrix and calculates the dRMSD using the 22.8Å cutoff threshold.[33] The curriculum learning script is deployed, gradually shifting the loss from trigonometric MSE to dRMSD to safely force topological correction without vanishing gradients.[8]
3.  **Biophysical Constraints:** A PyTorch-native physical simulator is injected.[3] The Lennard-Jones 6-12 potential executes dynamically over the predicted spatial coordinates. Gradient clipping mechanisms are actively calibrated to suppress the violently high loss values generated by the $r^{-12}$ potential term during extreme steric clashes.[37, 38]
4.  **Final Verification Checkpoint:** A holistic evaluation of the unmasked system is executed. The neural network is challenged with orphan proteins and sequences lacking evolutionary homologs to definitively prove that a single-sequence, coarse-grained Transformer natively integrating physical force fields is viable and superior to the limitations of MSA-dependent architectures.[1, 3]

# References
Single_Sequence_Prediction_of_Coarse_Grained_Protein_Structure_via_Transformer_Architecture-1.pdf
Protein Structure Prediction Using a Maximum Likelihood Formulation of a Recurrent Geometric Network | bioRxiv, hämtad april 29, 2026, https://www.biorxiv.org/content/10.1101/2021.09.03.458873.full
Integrating physics in deep learning algorithms: A force field as a PyTorch module - bioRxiv, hämtad april 29, 2026, https://www.biorxiv.org/content/10.1101/2023.01.12.523724v1.full.pdf
Structure-Preserving Coarse-Grained Simulation of Proteins in Explicit Solvent - bioRxiv, hämtad april 29, 2026, https://www.biorxiv.org/content/biorxiv/early/2025/08/23/2025.08.20.671185.full.pdf
End-to-End Differentiable Learning of Protein Structure - bioRxiv, hämtad april 29, 2026, https://www.biorxiv.org/content/10.1101/265231v1.full
SRS_Template.doc - Google Docs, hämtad april 29, 2026, https://docs.google.com/document/d/1mbBZ9oEcBwKDFcT5tvFbaKDshp_a4yG0-w8n1TIrK6s/
How to Write a System Requirements Specification (SRS) Document - Jama Software, hämtad april 29, 2026, https://www.jamasoftware.com/requirements-management-guide/writing-requirements/system-requirements-specification/
kleinhenz/psifold: Transformer model for protein structure prediction - GitHub, hämtad april 29, 2026, https://github.com/kleinhenz/psifold
How to Write an SRS Document That Actually Gets Read | Clearly, hämtad april 29, 2026, https://www.clearlyreqs.com/blog/how-to-write-srs-document
Software requirement document template: free SRS steps [2026] - Asana, hämtad april 29, 2026, https://asana.com/resources/software-requirement-document-template
jam01/SRS-Template: A markdown template for Software Requirements Specification based on IEEE 830 and ISO/IEC/IEEE 29148:2011 - GitHub, hämtad april 29, 2026, https://github.com/jam01/SRS-Template
Appendix C: IEEE 830 Template – Requirements Engineering - Rebus Press, hämtad april 29, 2026, https://press.rebus.community/requirementsengineering/back-matter/appendix-c-ieee-830-template/
sidechainnet_walkthrough_v3 - Colab, hämtad april 29, 2026, https://colab.research.google.com/drive/178vGN5aMD_gmS0Z4XbFWMbUZu3xHAWmD?usp=sharing
jonathanking/sidechainnet: An all-atom protein structure dataset for machine learning. - GitHub, hämtad april 29, 2026, https://github.com/jonathanking/sidechainnet
SidechainNet: An All-Atom Protein Structure Dataset for Machine Learning - MLSB 2025 Workshop, hämtad april 29, 2026, https://www.mlsb.io/papers/MLSB2020_SidechainNet:_An_All-Atom_Protein.pdf
Integration of side-chain orientation and global distance-based measures for improved evaluation of protein structural models - PMC, hämtad april 29, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC7018003/
Coarse-Grained Protein Model, hämtad april 29, 2026, https://a-site.vmhost.psu.edu/coarse-grained/
The pseudo-torsional angles. ( A ) The virtual bond scheme for RNA... | Download Scientific Diagram - ResearchGate, hämtad april 29, 2026, https://www.researchgate.net/figure/The-pseudo-torsional-angles-A-The-virtual-bond-scheme-for-RNA-nucleotides-B-The_fig5_233397015
sidechainnet/version1_notes.md at master - GitHub, hämtad april 29, 2026, https://github.com/jonathanking/sidechainnet/blob/master/version1_notes.md
Positional Encodings in Transformer Models - MachineLearningMastery.com, hämtad april 29, 2026, https://machinelearningmastery.com/positional-encodings-in-transformer-models/
Understanding How Positional Encodings Work in Transformer Model - ACL Anthology, hämtad april 29, 2026, https://aclanthology.org/2024.lrec-main.1478.pdf
Custom loss function for discontinuous angle calculation - vision - PyTorch Forums, hämtad april 29, 2026, https://discuss.pytorch.org/t/custom-loss-function-for-discontinuous-angle-calculation/58579
Implementation of all Loss Functions (Deep Learning) in NumPy, TensorFlow, and PyTorch, hämtad april 29, 2026, https://arjun-sarkar786.medium.com/implementation-of-all-loss-functions-deep-learning-in-numpy-tensorflow-and-pytorch-e20e72626ebd
aqlaboratory/pnerf - GitHub, hämtad april 29, 2026, https://github.com/aqlaboratory/pnerf
Learning Correlations between Internal Coordinates to Improve 3D Cartesian Coordinates for Proteins - ACS Publications, hämtad april 29, 2026, https://pubs.acs.org/doi/10.1021/acs.jctc.2c01270
MP-NeRF: A massively parallel method for accelerating protein structure reconstruction from internal coordinates - PubMed, hämtad april 29, 2026, https://pubmed.ncbi.nlm.nih.gov/34709663/
MP-NeRF: A Massively Parallel Method for Accelerating Protein Structure Reconstruction from Internal Coordinates | bioRxiv, hämtad april 29, 2026, https://www.biorxiv.org/content/10.1101/2021.06.08.446214.full
SidechainNet: An All-Atom Protein Structure Dataset for Machine Learning - PMC, hämtad april 29, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC8492522/
What is Relative Positional Encoding | by Ngieng Kianyew - Medium, hämtad april 29, 2026, https://medium.com/@ngiengkianyew/what-is-relative-positional-encoding-7e2fbaa3b510
[1803.02155] Self-Attention with Relative Position Representations - arXiv, hämtad april 29, 2026, https://arxiv.org/abs/1803.02155
End-to-End Differentiable Learning of Protein Structure - PubMed - NIH, hämtad april 29, 2026, https://pubmed.ncbi.nlm.nih.gov/31005579/
Adaptive gradient scaling: integrating Adam and landscape modification for protein structure prediction - PMC, hämtad april 29, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC12210780/
Mimetic Neural Networks: A Unified Framework for Protein Design and Folding - Frontiers, hämtad april 29, 2026, https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2022.715006/pdf
Automatically differentiable atomistic potentials for molecular simulations - GitHub, hämtad april 29, 2026, https://github.com/google/differentiable-atomistic-potentials
Interpreting forces as deep learning gradients improves quality of predicted protein structures - PMC, hämtad april 29, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC11393680/
DesmondZhong/diff_sim_improve_grads: PyTorch and Taichi implementations of our paper on improving gradient computation - GitHub, hämtad april 29, 2026, https://github.com/DesmondZhong/diff_sim_improve_grads
Introductory Tutorial: Lennard-Jones Liquid, hämtad april 29, 2026, https://espressomd.github.io/tutorials/lennard_jones/lennard_jones.html
Automated Minimization of Steric Clashes in Protein Structures - PMC - NIH, hämtad april 29, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC3058769/
Learning Protein Structure with a Differentiable Simulator - cs.Princeton, hämtad april 29, 2026, https://www.cs.princeton.edu/courses/archive/fall23/cos597N/lectures/2023_09_14_lec2_differentiable_simulator.pdf
Software Requirements Specification (SRS) | Enabel, hämtad april 29, 2026, https://www.enabel.be/app/uploads/2025/06/Annex-A-Detailed-Software-Requirements-Specification-SRS.pdf
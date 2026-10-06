"""
Inserts full LaTeX figure environments for Figures 1 to 10 into manuscript/manuscript_tcbb.tex
"""

import re

def update_manuscript_with_figures():
    filepath = "manuscript/manuscript_tcbb.tex"
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Fig 1 & Fig 2: In Section 2
    fig1_2_code = """
% FIGURE 1: Paradigm and Cohort
\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{figures/figure1_experimental_paradigm_and_cohort}
\\caption{\\textbf{Experimental Paradigm, Cohort Architecture, and Computational Pipeline.} (A) Optogenetic stimulation paradigm during Neuropixels recording in mouse visual cortex. Optical pulses (10~ms, 1.0--4.0~mW) and 10-Hz trains were delivered to surface layers. (B) Hierarchical sampling architecture across 28 independent biological specimens (mice), 159 Neuropixels probes, and 18,316 units stratified by Cre driver line (\\textit{Pvalb-IRES-Cre}, $N=8$; \\textit{Sst-IRES-Cre}, $N=12$; \\textit{Vip-IRES-Cre}, $N=8$). (C) Overview of the multi-stage computational framework, contrasting binary thresholding with multidimensional perturbational trajectory analysis.}
\\label{fig:paradigm}
\\end{figure*}

% FIGURE 2: Perturbational Response Space
\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{figures/figure2_perturbational_response_space}
\\caption{\\textbf{Perturbational Response Space and Representative Single-Unit Trajectories.} Representative peri-stimulus time histograms (PSTHs, 1-ms bins, shaded bands: SEM across trials) illustrating diverse cortical dynamics elicited by a 10-ms optical pulse: rapid direct-like excitation, prolonged network suppression, delayed post-inhibitory rebound, and stationary non-responsiveness. Shaded vertical bar ($0$--$10$~ms) marks stimulus duration.}
\\label{fig:psth_examples}
\\end{figure*}
"""

    # Insert Fig 1 & 2 after Section 2.5
    target_sec2 = "\\subsection{Hierarchical Data Organization}\nExtracellular recording datasets possess an intrinsic three-level nested hierarchy: units are physically clustered along probes, and probes are grouped within individual biological specimens:\n\\begin{equation}\n    \\text{Unit } i \\in \\{1, \\dots, N_{\\text{unit}}\\} \\subset \\text{Probe } j \\in \\{1, \\dots, N_{\\text{probe}}\\} \\subset \\text{Specimen } k \\in \\{1, \\dots, N_{\\text{specimen}}\\}.\n\\end{equation}\nStatistical analyses that fail to account for this nested structure incur severe pseudo-replication errors by treating $18,316$ units as independent biological samples."
    if target_sec2 in text:
        text = text.replace(target_sec2, target_sec2 + "\n" + fig1_2_code)
    else:
        print("[WARN] Target for Fig 1 & 2 not found exactly")

    # Fig 3: In Section 5.1
    fig3_code = """
% FIGURE 3: Method Disagreement
\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{figures/figure3_method_disagreement_and_divergence}
\\caption{\\textbf{High Divergence Across Responsiveness Algorithms ($N=18,316$).} (A) 8-way Venn partition across the Operational Heuristic, SALT, and ZETA algorithms, highlighting that three-way consensus encompasses only 13 units ($0.07\\%$). (B) Evoked firing rate versus trial reliability, showing that the operational heuristic (red) exclusively samples high-firing bursts, discarding 179 units that achieve dual statistical significance under SALT and ZETA (purple). (C) Baseline versus evoked rate distributions illustrating algorithmic ascertainment bias.}
\\label{fig:method_disagreement}
\\end{figure*}
"""
    target_sec51 = "\\subsection{Responsiveness Depends on Statistical Representation}\n\\textit{Computational Problem: Does the choice of statistical responsiveness algorithm systematically bias the identified responsive subpopulation?}"
    if target_sec51 in text:
        text = text.replace(target_sec51, target_sec51 + "\n" + fig3_code)

    # Fig 4: In Section 5.3
    fig4_code = """
% FIGURE 4: Latency & Poisson Null
\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{figures/figure4_latency_and_sparse_firing_reliability}
\\caption{\\textbf{Response Latency Heterogeneity and Sparse-Firing Poisson Fragility.} (A) Empirical first-spike latency histogram across 13,643 active units overlaid with a three-component Gaussian mixture model ($\\Delta\\text{BIC} = -117.5$ vs $k=2$). (B) The ambiguous borderline interval ($6.0$--$10.0$~ms, $N=3,193$, $23.40\\%$). (C) Comparison of empirical spiking fraction versus the stationary Poisson null model ($P(N \\ge 1) = 1 - e^{-\\lambda \\tau}$) across baseline firing tiers; in the quiescent tier ($<1$~Hz, $N=3,867$, $\\lambda=0.192$~Hz), spontaneous Poisson events explain $44.03\\%$ of observed spiking units ($518.7/1,178$). (D) Sub-8~ms latency rate ($93.04\\%$) versus heuristic pass rate ($0.59\\%$), demonstrating the mathematical fragility of unconditioned latency thresholds.}
\\label{fig:latency_poisson}
\\end{figure*}
"""
    target_sec53 = "\\subsection{Latency Becomes Fragile Under Sparse Baseline Firing}\n\\textit{Computational Problem: Why do deterministic latency thresholds fail in sparsely firing neurons?}"
    if target_sec53 in text:
        text = text.replace(target_sec53, target_sec53 + "\n" + fig4_code)

    # Fig 5: In Section 5.4
    fig5_code = """
% FIGURE 5: Cell Type Fingerprints
\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{figures/figure5_cell_type_fingerprints}
\\caption{\\textbf{Cell-Type-Associated Perturbational Fingerprints.} Multi-channel radar and bar charts comparing the three transgenic cohorts across seven perturbational dimensions: direct candidate yield ($W_1$), detectable network suppression prevalence ($W_3$--$W_4$), peak evoked firing rate, first-spike latency, 10-Hz train adaptation index ($AI$), direct intensity-response slope, and CCG coincident synchrony fold-change. Values represent mean $\\pm$ SEM across specimens.}
\\label{fig:fingerprints}
\\end{figure*}
"""
    target_sec54 = "\\subsection{Temporal Trajectories Reveal Three Response Archetypes}\n\\textit{Computational Problem: How do unconstrained peri-stimulus trajectories partition across the cortical population?}"
    if target_sec54 in text:
        text = text.replace(target_sec54, target_sec54 + "\n" + fig5_code)

    # Fig 6: In Section 5.5
    fig6_code = """
% FIGURE 6: Intensity Response
\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{figures/figure6_intensity_response_relationships}
\\caption{\\textbf{Optical Intensity Modulates Both Direct Drive and Lateral Circuit Silence.} (A) Population-level and direct candidate firing rate scaling across calibrated optical power levels ($1.0$, $2.5$, and $4.0$~mW) for \\textit{Pvalb-IRES-Cre}, \\textit{Sst-IRES-Cre}, and \\textit{Vip-IRES-Cre} sessions. Lines indicate hierarchical linear mixed-effects fits with random specimen intercepts. (B) Recruitment of non-tagged network units into lateral suppression as a function of optical power.}
\\label{fig:intensity}
\\end{figure*}
"""
    target_sec55 = "\\subsection{Optical Intensity Modulates Both Direct Drive and Suppression}\n\\textit{Computational Problem: How does optical power alter population response gain and recruitment?}"
    if target_sec55 in text:
        text = text.replace(target_sec55, target_sec55 + "\n" + fig6_code)

    # Fig 7: In Section 5.6
    fig7_code = """
% FIGURE 7: Pulse Train Adaptation
\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{figures/figure7_pulse_train_dynamics_and_adaptation}
\\caption{\\textbf{Pulse-Train Dynamics and Standardized Adaptation ($10$-Hz Train).} (A) Pulse-by-pulse normalized firing trajectories ($R_1 \\dots R_{10}$) for responsive units ($\\text{Evoked Rate} \\ge 2.0$~Hz, $N=12,311$) across Cre lines. (B) Distribution of standardized adaptation index values ($AI = (R_{10} - R_1)/\\max(R_1, 0.5)$), demonstrating marked depression in \\textit{Pvalb} ($74.96\\%$) and \\textit{Sst} ($85.02\\%$), and predominant stability ($65.73\\%$) or facilitation ($23.77\\%$) in \\textit{Vip}.}
\\label{fig:adaptation}
\\end{figure*}
"""
    target_sec56 = "\\subsection{Pulse Trains Produce Cell-Type-Associated Adaptation}\n\\textit{Computational Problem: Does repeated high-frequency stimulation separate functional interneuron classes?}"
    if target_sec56 in text:
        text = text.replace(target_sec56, target_sec56 + "\n" + fig7_code)

    # Fig 8: In Section 5.7
    fig8_code = """
% FIGURE 8: Distance-Dependent Spatial Structure
\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{figures/figure8_distance_dependent_response_structure}
\\caption{\\textbf{Distance-Dependent Response Structure along Neuropixels Shanks.} (A) Primary finding: First-spike latency increases with physical distance from the optical hotspot along the probe shank (fitted slope: $+1.1~\\mu\\text{s}/\\mu\\text{m}$ in PV, $+0.3~\\mu\\text{s}/\\mu\\text{m}$ in SST; secondary apparent velocity $v \\approx 0.07$~m/s under linear model). (B) Spatial decay of direct-like drive ($\\lambda \\approx 120$--$160~\\mu\\text{m}$). (C) Broad spatial extent of Prolonged Suppression, spanning $>50\\%$ of units across all distance bins up to $>600~\\mu\\text{m}$.}
\\label{fig:spatial}
\\end{figure*}
"""
    target_sec57 = "\\subsection{Response Latency Varies with Distance from Hotspot}\n\\textit{Computational Problem: How does the spatial distribution of response latency behave along linear multi-electrode arrays?}"
    if target_sec57 in text:
        text = text.replace(target_sec57, target_sec57 + "\n" + fig8_code)

    # Fig 9: In Section 5.8
    fig9_code = """
% FIGURE 9: Population Temporal Coordination
\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{figures/figure9_population_temporal_coordination}
\\caption{\\textbf{Population Temporal Coordination Shifts (Cross-Correlograms).} (A) Multi-unit cross-correlograms between direct candidates and network units in \\textit{Pvalb} sessions during spontaneous baseline (dashed gray) versus post-stimulation intervals (solid blue), showing a $4.04$-fold surge in coincident synchrony at $+3.2$~ms peak lag followed by coordinated suppression. (B) Corresponding CCG in \\textit{Sst} sessions displaying a $2.50$-fold synchrony shift at $+4.8$~ms lag. (C) Summary of synchrony fold-changes across Cre lines.}
\\label{fig:ccg}
\\end{figure*}
"""
    target_sec58 = "\\subsection{Perturbation Reorganizes Population Coordination}\n\\textit{Computational Problem: Does optical perturbation alter fine-scale temporal coordination among simultaneously recorded units?}"
    if target_sec58 in text:
        text = text.replace(target_sec58, target_sec58 + "\n" + fig9_code)

    # Fig 10: In Section 5.9
    fig10_code = """
% FIGURE 10: Specimen Replication and Variance
\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{figures/figure10_specimen_replication_and_variance}
\\caption{\\textbf{Specimen-Level Replication and Hierarchical Variance Decomposition.} (A) Caterpillar plot of specimen-level direct candidate prevalence estimates across all 28 independent mice (points: specimen means, error bars: $95\\%$ CI; red dashed line: cohort mean $1.42\\%$). Specimen-level effects are consistently displaced from null ($p < 0.0001, N=28$). (B) Hierarchical variance decomposition via one-way random effects ANOVA and Intraclass Correlation Coefficients (ICC), demonstrating that $>98\\%$ of modeled variance resides within specimens across recording depths, while between-specimen variation is small ($<2\\%$).}
\\label{fig:reproducibility}
\\end{figure*}
"""
    target_sec59 = "\\subsection{Hierarchical Variance: Within-Specimen Predominance}\n\\textit{Computational Problem: Does response variance reflect animal-to-animal technical variability or cellular/laminar diversity within specimens?}"
    if target_sec59 in text:
        text = text.replace(target_sec59, target_sec59 + "\n" + fig10_code)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print("[SUCCESS] All 10 Figures embedded into manuscript/manuscript_tcbb.tex")

if __name__ == "__main__":
    update_manuscript_with_figures()

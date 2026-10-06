# Data dictionary: `optotagging_units.{csv,parquet}`

19,005 units x 147 columns. One row per unit passing the Allen default QC (isi_violations < 0.5, amplitude_cutoff < 0.1, presence_ratio > 0.9, quality = good) in the 28 Pvalb/Sst/Vip x Ai32 sessions. Every value is computed from NWB spike times by the scripts in scripts/reanalysis/ and scripts/calibration/. Windows are relative to light (or flash) onset.

| column | units | description |
|---|---|---|
| `unit_id` | - | Allen ecephys unit id |
| `session_id` | - | Allen ecephys session id |
| `specimen_id` | - | Allen specimen (mouse) id; one session per mouse in this cohort |
| `cre_line` | - | Cre driver line crossed to Ai32 (ChR2(H134R)-EYFP): Pvalb-, Sst- or Vip-IRES-Cre |
| `sex` | - | sex of the mouse (Allen metadata) |
| `age_in_days` | days | age at recording (Allen metadata) |
| `probe_name` | - | Neuropixels probe label within the session (probeA-F) |
| `probe_id` | - | Allen probe id |
| `structure` | - | CCF structure acronym of the unit's peak channel |
| `region_group` | - | coarse region: visual cortex / hippocampal formation / thalamus / midbrain / other |
| `ccf_ap` | um | CCF anterior-posterior coordinate of the peak channel |
| `ccf_dv` | um | CCF dorsal-ventral coordinate of the peak channel |
| `ccf_lr` | um | CCF left-right coordinate of the peak channel |
| `probe_vertical_position` | um | peak-channel position along the shank (0 = tip) |
| `cortical_depth_um` | um | distance below the highest VIS* channel on the probe (cortical units only; NaN otherwise) |
| `waveform_duration` | ms | trough-to-peak duration of the mean waveform (Allen metric) |
| `waveform_halfwidth` | ms | spike half-width (Allen metric) |
| `PT_ratio` | - | peak-to-trough amplitude ratio (Allen metric) |
| `firing_rate` | Hz | session-wide mean firing rate (Allen metric) |
| `snr` | - | waveform signal-to-noise ratio (Allen metric) |
| `lam_spont_hz` | Hz | spontaneous rate from the session's spontaneous epochs (invalid intervals removed) |
| `spont_duration_s` | s | valid spontaneous-epoch exposure used for lam_spont_hz |
| `p10_n` | trials | valid trials, 10-ms single pulses |
| `p10_nb` | spikes | spike count in the baseline window [-480, -20) ms summed over trials, 10-ms single pulses |
| `p10_ne` | spikes | spike count in the evoked window [+1, +9) ms summed over trials, 10-ms single pulses |
| `p10_lam0_hz` | Hz | baseline rate from [-480, -20) ms of the same trials, 10-ms single pulses |
| `p10_p_exc` | - | exact conditional (binomial) test p, evoked > baseline, pi0 = 0.008/0.468, 10-ms single pulses |
| `p10_p_sup` | - | exact conditional test p, evoked < baseline, 10-ms single pulses |
| `p10_z` | SD | mid-p excitation z-score of the exact test (input to the lfdr), 10-ms single pulses |
| `p10_lrr` | log ratio | half-count-corrected log rate ratio evoked/baseline, 10-ms single pulses |
| `p10_lrr_se` | log ratio | approximate SE of p10_lrr |
| `p10_k` | trials | trials with >= 1 spike in [+1, +9) ms, 10-ms single pulses |
| `p10_rel_raw` | fraction | raw trial reliability k/n, 10-ms single pulses |
| `p10_rho` | probability | chance-corrected reliability (r - p0)/(1 - p0), 10-ms single pulses |
| `p10_rho_se` | probability | SE of p10_rho |
| `p10_p0` | probability | chance probability of >= 1 spike in 8 ms at the baseline rate, 10-ms single pulses |
| `p10_p_rel` | - | one-sided binomial p for reliability > p0, 10-ms single pulses |
| `p10_median_first_ms` | ms | median first-spike latency within [+1, +9) ms (includes spontaneous spikes), 10-ms single pulses |
| `p10_salt_p` | - | SALT p (port of Kvitsiani et al. 2013; 8-ms test window, 1-ms bins), 10-ms single pulses |
| `p10_salt_I` | bits | SALT information difference, 10-ms single pulses |
| `p10_fit_rho` | probability | ML first-spike model: probability of an evoked spike (fitted only if p_exc < 0.01), 10-ms single pulses |
| `p10_fit_delta` | ms | ML first-spike model: evoked latency, 10-ms single pulses |
| `p10_fit_sigma` | ms | ML first-spike model: evoked jitter (SD), 10-ms single pulses |
| `p10_fit_se_rho` | probability | Wald SE of fit_rho (under-covers for weak responses, see estimator validation) |
| `p10_fit_se_delta` | ms | Wald SE of fit_delta |
| `p10_fit_se_logsigma` | log ms | Wald SE of log(fit_sigma) |
| `p10_fit_converged` | bool | optimiser convergence flag of the latency-model fit |
| `p5_n` | trials | valid trials, 5-ms single pulses |
| `p5_nb` | spikes | spike count in the baseline window [-480, -20) ms summed over trials, 5-ms single pulses |
| `p5_ne` | spikes | spike count in the evoked window [+1, +9) ms summed over trials, 5-ms single pulses |
| `p5_lam0_hz` | Hz | baseline rate from [-480, -20) ms of the same trials, 5-ms single pulses |
| `p5_p_exc` | - | exact conditional (binomial) test p, evoked > baseline, pi0 = 0.008/0.468, 5-ms single pulses |
| `p5_p_sup` | - | exact conditional test p, evoked < baseline, 5-ms single pulses |
| `p5_z` | SD | mid-p excitation z-score of the exact test (input to the lfdr), 5-ms single pulses |
| `p5_lrr` | log ratio | half-count-corrected log rate ratio evoked/baseline, 5-ms single pulses |
| `p5_lrr_se` | log ratio | approximate SE of p5_lrr |
| `p5_k` | trials | trials with >= 1 spike in [+1, +9) ms, 5-ms single pulses |
| `p5_rel_raw` | fraction | raw trial reliability k/n, 5-ms single pulses |
| `p5_rho` | probability | chance-corrected reliability (r - p0)/(1 - p0), 5-ms single pulses |
| `p5_rho_se` | probability | SE of p5_rho |
| `p5_p0` | probability | chance probability of >= 1 spike in 8 ms at the baseline rate, 5-ms single pulses |
| `p5_p_rel` | - | one-sided binomial p for reliability > p0, 5-ms single pulses |
| `p5_median_first_ms` | ms | median first-spike latency within [+1, +9) ms (includes spontaneous spikes), 5-ms single pulses |
| `p5_salt_p` | - | SALT p (port of Kvitsiani et al. 2013; 8-ms test window, 1-ms bins), 5-ms single pulses |
| `p5_salt_I` | bits | SALT information difference, 5-ms single pulses |
| `p5_fit_rho` | probability | ML first-spike model: probability of an evoked spike (fitted only if p_exc < 0.01), 5-ms single pulses |
| `p5_fit_delta` | ms | ML first-spike model: evoked latency, 5-ms single pulses |
| `p5_fit_sigma` | ms | ML first-spike model: evoked jitter (SD), 5-ms single pulses |
| `p5_fit_se_rho` | probability | Wald SE of fit_rho (under-covers for weak responses, see estimator validation) |
| `p5_fit_se_delta` | ms | Wald SE of fit_delta |
| `p5_fit_se_logsigma` | log ms | Wald SE of log(fit_sigma) |
| `p5_fit_converged` | bool | optimiser convergence flag of the latency-model fit |
| `p10_low_level` | NWB 'level' units | light level of the low setting in this session (sets differ between sessions) |
| `p10_low_n` | trials | 10-ms trials at the low light level |
| `p10_low_k` | trials | trials with >= 1 evoked spike at the low level |
| `p10_low_rho` | probability | chance-corrected reliability at the low level |
| `p10_low_rho_se` | probability | SE of p10_low_rho |
| `p10_low_median_first_ms` | ms | median first-spike latency at the low level |
| `p10_mid_level` | NWB 'level' units | light level of the mid setting in this session (sets differ between sessions) |
| `p10_mid_n` | trials | 10-ms trials at the mid light level |
| `p10_mid_k` | trials | trials with >= 1 evoked spike at the mid level |
| `p10_mid_rho` | probability | chance-corrected reliability at the mid level |
| `p10_mid_rho_se` | probability | SE of p10_mid_rho |
| `p10_mid_median_first_ms` | ms | median first-spike latency at the mid level |
| `p10_high_level` | NWB 'level' units | light level of the high setting in this session (sets differ between sessions) |
| `p10_high_n` | trials | 10-ms trials at the high light level |
| `p10_high_k` | trials | trials with >= 1 evoked spike at the high level |
| `p10_high_rho` | probability | chance-corrected reliability at the high level |
| `p10_high_rho_se` | probability | SE of p10_high_rho |
| `p10_high_median_first_ms` | ms | median first-spike latency at the high level |
| `dose_delta_rho` | probability | p10_high_rho - p10_low_rho |
| `train_n` | trials | valid 10-Hz train trials (10 x 2.5-ms pulses, 100-ms period) |
| `train_lam0_hz` | Hz | baseline rate before train trials, [-480, -20) ms |
| `train_k1` | trials | train trials with >= 1 spike in [+1, +9) ms after pulse 1 |
| `train_k2` | trials | train trials with >= 1 spike in [+1, +9) ms after pulse 2 |
| `train_k3` | trials | train trials with >= 1 spike in [+1, +9) ms after pulse 3 |
| `train_k4` | trials | train trials with >= 1 spike in [+1, +9) ms after pulse 4 |
| `train_k5` | trials | train trials with >= 1 spike in [+1, +9) ms after pulse 5 |
| `train_k6` | trials | train trials with >= 1 spike in [+1, +9) ms after pulse 6 |
| `train_k7` | trials | train trials with >= 1 spike in [+1, +9) ms after pulse 7 |
| `train_k8` | trials | train trials with >= 1 spike in [+1, +9) ms after pulse 8 |
| `train_k9` | trials | train trials with >= 1 spike in [+1, +9) ms after pulse 9 |
| `train_k10` | trials | train trials with >= 1 spike in [+1, +9) ms after pulse 10 |
| `train_following` | probability | mean over the 10 pulses of the chance-corrected per-pulse response |
| `zeta9_p` | - | zetapy ZETA p, 10-ms pulses, window [1, 9) ms, seeded (2.5e2 resamples) |
| `zeta9_sign` | - | sign of the ZETA deviation (+1 excitation, -1 suppression), window [1, 9) ms |
| `zeta50_p` | - | zetapy ZETA p, 10-ms pulses, window [1, 51) ms, seeded (2.5e2 resamples) |
| `zeta50_sign` | - | sign of the ZETA deviation (+1 excitation, -1 suppression), window [1, 51) ms |
| `lfdr` | probability | local false discovery rate of light activation (empirical null from sham windows) |
| `evidence` | probability | 1 - lfdr |
| `q_exact` | - | Benjamini-Hochberg q of p10_p_exc across all 19,005 units |
| `onset_artifact` | bool | light-onset artifact: fit_delta < 1.5 ms and fit_sigma < 0.3 ms |
| `driven` | bool | calibrated light-activated: lfdr < 0.05 and not onset_artifact |
| `direct_like` | bool | driven and fit_delta < 5 ms and fit_sigma < 1.5 ms (the criterion tested in §4A) |
| `label_allen_lakunina` | bool | criterion label on the real 10-ms pulse trials: >= 4 of first 5 train pulses with significant per-pulse response, raw reliability >= 0.30, median latency < 8 ms |
| `sham_allen_lakunina` | bool | same criterion evaluated on sham trials placed in spontaneous epochs (should be False) |
| `label_allen_lakunina_raw` | bool | criterion label on the real 10-ms pulse trials: as allen_lakunina but per-pulse raw probability >= 0.30 at >= 4 of 5 pulses |
| `sham_allen_lakunina_raw` | bool | same criterion evaluated on sham trials placed in spontaneous epochs (should be False) |
| `label_latency_lt8` | bool | criterion label on the real 10-ms pulse trials: median first-spike latency in [1, 9) ms < 8 ms |
| `sham_latency_lt8` | bool | same criterion evaluated on sham trials placed in spontaneous epochs (should be False) |
| `label_rel030_mod2` | bool | criterion label on the real 10-ms pulse trials: raw reliability >= 0.30 and modulation ratio (original definition) > 2 |
| `sham_rel030_mod2` | bool | same criterion evaluated on sham trials placed in spontaneous epochs (should be False) |
| `label_heuristic_full` | bool | criterion label on the real 10-ms pulse trials: original operational rule: rel >= 0.30, latency < 8 ms, modulation > 2, permutation p < 0.05, Cohen's d > 0.1 |
| `sham_heuristic_full` | bool | same criterion evaluated on sham trials placed in spontaneous epochs (should be False) |
| `label_salt_p01` | bool | criterion label on the real 10-ms pulse trials: SALT p < 0.01 |
| `sham_salt_p01` | bool | same criterion evaluated on sham trials placed in spontaneous epochs (should be False) |
| `label_salt_bh` | bool | criterion label on the real 10-ms pulse trials: SALT BH q < 0.05 |
| `sham_salt_bh` | bool | same criterion evaluated on sham trials placed in spontaneous epochs (should be False) |
| `label_zeta9_bh` | bool | criterion label on the real 10-ms pulse trials: ZETA [1, 9) ms BH q < 0.05, positive deviation |
| `sham_zeta9_bh` | bool | same criterion evaluated on sham trials placed in spontaneous epochs (should be False) |
| `label_zeta51_bh` | bool | criterion label on the real 10-ms pulse trials: ZETA [1, 51) ms BH q < 0.05, any sign |
| `sham_zeta51_bh` | bool | same criterion evaluated on sham trials placed in spontaneous epochs (should be False) |
| `label_exact_bh` | bool | criterion label on the real 10-ms pulse trials: exact conditional test BH q < 0.05 |
| `sham_exact_bh` | bool | same criterion evaluated on sham trials placed in spontaneous epochs (should be False) |
| `label_lfdr05` | bool | criterion label on the real 10-ms pulse trials: lfdr < 0.05 (lfdr curve fitted on real data) |
| `sham_lfdr05` | bool | same criterion evaluated on sham trials placed in spontaneous epochs (should be False) |
| `modulation_ratio_original` | - | evoked [1,9) ms rate / (baseline [-20,-5) ms rate + 1 Hz), original definition |
| `train_sig_pulses` | pulses | number of the first 5 train pulses with a significant per-pulse response |
| `opto_lrr_20_200` | log ratio | log rate ratio 20-200 ms after 10-ms pulses vs baseline |
| `flash_lrr_20_200` | log ratio | log rate ratio 20-200 ms after full-field flash onset vs baseline |
| `opto_suppressed` | bool | suppressed 20-50 or 50-200 ms after 10-ms pulses (exact test, BH q < 0.05; not driven) |
| `flash_suppressed` | bool | suppressed 20-50 or 50-200 ms after flash onset (same test) |
| `flash_excited` | bool | excited 20-50 or 50-200 ms after flash onset (same test) |
| `ccg_role` | - | CCG reference role: 'driven', 'control' (strict match), 'control_relaxed', or empty |
| `ccg_n_partners` | pairs | same-probe partners within 300 um with valid spontaneous CCGs |
| `ccg_source_rate` | fraction | fraction of partners with a causal short-latency trough (unit inhibits partner) |
| `ccg_target_rate` | fraction | fraction of partners with an anticausal trough (partner inhibits unit) |

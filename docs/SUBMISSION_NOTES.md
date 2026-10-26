# Submission note: Journal of Causal Inference

- **Journal:** Journal of Causal Inference (De Gruyter), Research Article, Open Access
- **Submitted:** 2026-10-26 (submitted by Quang-Vinh Dang; corresponding author in the manuscript: Dat Le)
- **Title:** What can real cause-effect benchmarks resolve? Sampling noise, marginal shortcuts and assumption regimes
- **Authors:** Quang-Vinh Dang (British University Vietnam), Dat Le* (Ho Chi Minh City University of Economics and Finance), Minh Ngoc Dinh (Millennia Education)
- **Repository state at submission:** commit `558f8f3` on `main`
- **Manuscript source:** `paper/resolution_jci/main_jci.tex` (De Gruyter template, 6 sections + appendix, 29 pages); PDF `main_jci.pdf`
- **Other submitted files:** `cover_letter.txt` (the plain-text version given in the conversation is the one used), `JCI_Ethical_and_Legal_Declarations.docx`, `JCI_License_to_Publish_filled.pdf` (signature, printed name and date to be completed by the signatory), `submission_title_abstract.txt`
- **Number of experiments / provenance:** 47 experiment scripts (17-61), each committed before it was run; table-to-script map in `docs/TABLE_MAP.md`; negative results in `docs/NEGATIVE_RESULTS.md`
- **Funding / conflict of interest:** none declared. arXiv: not posted. APC: EUR 1050 after acceptance per the journal page (waiver possible); not yet discussed with the editorial office.

## Open points to check after submission
1. **AI-use disclosure.** The declarations file states the use of an LLM (Claude, Anthropic) for code, experiments, drafting and cross-checking. The journal also asks that this be documented in the manuscript; the manuscript currently has no such sentence. Make the manuscript and the declarations agree (the journal says only the declarations file is used in the final article).
2. **CRediT roles** in the declarations were filled in as a plausible draft; confirm with the co-authors.
3. **License to Publish:** must be signed and uploaded by the signatory (the corresponding author, or the submitter with authorisation from all co-authors).
4. **Repository** must be public so reviewers can open the data-availability link.
5. Minh Ngoc Dinh's affiliation string is the short form supplied; replace with the full one if provided.

## Known weak points a reviewer may raise (from the internal reviews)
- No new method; contribution is evaluation plus mechanism evidence.
- The round255 compression convention was chosen because it reproduces a published figure; all 18 conventions are in the appendix.
- New chamber collections have 4-8 distinct edges (case studies); cluster intervals degenerate there.
- The factorial simulator does not cover the real collections (kNN predictive validation failed, reported in the appendix).
- Several rules are our own implementations (cdt cross-check in Section 4).

## If a revision is requested
- Keep `results/` and the scripts unchanged; add new analyses as new numbered experiments committed before running.
- Rebuild the paper with the template files in `paper/resolution_jci/` (needs `texlive-lang-german` for the `ngerman` babel setting; set `TEXINPUTS` and `TFMFONTS` to `fonts/mmathdgm`).

# AgenticRL AAAI-27 Overleaf Package

## Compile

- Upload the ZIP archive to a new Overleaf project.
- Set `main.tex` as the main document.
- Select **pdfLaTeX** as the compiler.
- The source uses the official `aaai2027.sty` and `aaai2027.bst`.

## Included submission figures

- `figures/fig1_motivation_submission.png`
- `figures/fig2_agenticrl_architecture.pdf`
- `figures/fig3_transition_final.pdf`
- `figures/fig4_clinical_breakdown_final.pdf`

Figures 3 and 4 are vector PDFs with embedded fonts and no Type 3 fonts.
Figure 1 is a 300 dpi PNG. Figure 2 is supplied as a PDF.

## Final-result anchors

- Closed-book: 40.59%
- One-shot RAG: 38.85%
- Single-round agent: 40.12%
- Iterative agent without memory: 39.56%
- AgenticRL Full: 42.84%
- Full versus single-round: +2.72 percentage points
- Corrected single-round errors: 436 / 4,091 (10.66%)
- Regressed single-round correct answers: 250 / 2,741 (9.12%)

The title and abstract are retained as submitted. The body distinguishes the
complete fixed-budget strategy from isolated memory attribution.

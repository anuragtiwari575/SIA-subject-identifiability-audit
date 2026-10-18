"""SIA: Subject Identifiability Auditing for patient-level leakage in medical imaging benchmarks.

Three stages (see the paper, Section 3):
  Stage 1  composition analysis       -> sia.audit.composition
  Stage 2  paired protocol ablation   -> sia.audit.leakage (Leakage Index)
  Stage 3  Subject Identifiability    -> sia.audit.sis (SIS, control task, selectivity)
"""

__version__ = "1.0.0"

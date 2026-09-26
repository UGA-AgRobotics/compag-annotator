# 1.0.0rc5: consecutive-tile GPU memory check

Generate masks now releases unused PyTorch allocator cache only when driver-free
memory falls below the configured reserve, then checks the selected GPU again.
Previously, a successful first tile could leave a large reusable cache and cause
the next tile to fail the preflight check. A real remaining shortage still stops
with the free and required MiB shown in the error.

The sampling option is called **Default settings**. The generation dialog uses
general image-annotation wording without paper-specific labels. Numerical SAM
defaults, the selected checkpoint, precision, tiling and quality thresholds are
unchanged. No automatic CPU fallback or quality reduction is introduced.

The generation recipe identity is unchanged: existing partial jobs can use
**Resume** or **Retry** with their saved settings, subject to the existing image,
model and recipe validation. Completed tiles and reviewed annotations stay intact.

Tests cover a reusable cache, a genuine shortage, sufficient-memory/no-flush,
explicit reserve and CPU behavior. A bounded real GPU check uses three consecutive
crops from the same previously failing image with the same SAM2 Base+ checkpoint,
64 x 64 point grid, batch 512 and Float32. Its evidence is reported separately;
this check does not establish complete-image coverage or general quality parity.

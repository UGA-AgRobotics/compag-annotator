# 1.0.0rc7: independent SAM points and explicit review decisions

SAM assistance defaults to **One mask per point · multiple objects**. Choose a
SAM2 or local-weight SAM3 model, use **SAM positive point (+)** and click once
inside each desired object anywhere in the full image. **Preview N masks** starts
one job containing N independent positive-point requests (maximum 64). Their
previews appear together. **Add all N masks as drafts** adds the returned masks
in one revision-checked transaction and one undo step, preserving separate
instances and every pre-existing annotation.

The decoder chooses its highest-scoring nonempty candidate for each point.
Points returning no mask are reported, never filled with fabricated masks.
Different points may cover the same object; no merging or deduplication is
performed. Cached image encoding is reused where the provider permits it; point
requests run serially within the job to bound GPU memory. This does not imply
parallel GPU inference or guarantee correct segmentation.

Use **Refine one object · points / box** for multiple positive/negative points or
a box describing one object. **Refine with SAM** selects this mode for an existing
annotation. Switching prompt modes clears current points/previews. Changing the
image, class, model, device, settings or saved revision invalidates a preview.
The saved batch is bound to the server job, image bytes and original revision.
Cancelled, failed and stale previews cannot partially add masks.

Review status now distinguishes **Labeled · needs review**, **Accepted**,
**Rejected**, and **Needs label & review**. Automated-QA acceptance remains
explicitly marked. The Selection panel reports pending/accepted/rejected counts
for all objects and for hidden objects. Hiding or assigning a class never accepts
a mask. **Review N pending masks** reveals pending objects (clearing other view
filters) and hides completed decisions; **Show all active masks** exits that
review filter. Acceptance, rejection, labels, exports and persistent undo retain
their existing semantics.

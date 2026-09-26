# Mask display controls — 1.0.0rc9

In Annotate → Selection, enable **Hide generated SAM masks** to hide objects originating in Generate masks, including labeled, accepted or refined objects from those automatic layers. New manual objects and new SAM point/box assistance masks remain visible unless another display filter hides them. Hidden objects are removed from current selection and cannot intercept clicks or be selected through the object/layer lists. Uncheck to show them again. Review pending masks resets this filter to expose unfinished review.

In **Layers & display**, uncheck **Show mask fill** to remove the translucent fill over saved annotations. Objects stay selectable, and hovering/selecting a raster mask shows its bounding rectangle. Vector outlines remain visible. Temporary SAM previews and active brush feedback remain visible so editing is still reviewable. The opacity slider's value is preserved.

Defaults preserve the previous appearance: generated masks are visible and mask fill is enabled. These controls are temporary editor display settings, like the existing filters. They do not change labels, geometry, acceptance, exports, training or saved project history. The interface remains English-only.

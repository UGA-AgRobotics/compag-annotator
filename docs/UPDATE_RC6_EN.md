# 1.0.0rc6: hide labeled masks while reviewing

In the annotation inspector, enable **Hide assigned masks** under **Selection**.
Existing labeled objects disappear, and each newly labeled mask disappears after
its assignment is saved. Uncheck the option to see them again. The option starts
off whenever an image is opened and remains active while editing that image.

This is a view filter. It neither deletes nor accepts annotations and does not
change exports, training eligibility, saved geometry or undo history. Hidden
objects leave the active selection and overlap picker, and cannot intercept
canvas clicks or be selected from the object list while hidden. Their rows stay
in the list, marked Hidden, so the visibility state is apparent.

To hide only one class, uncheck its name under **Layers & display**. To hide one
object, uncheck its **Show object** checkbox. These filters combine: disabling
Hide assigned masks does not override a separately hidden class or object.

Undoing a class assignment reveals an unassigned mask while the filter is on;
redoing the assignment hides it again. Turn the filter off for acceptance or
correction of already labeled objects. Assigning a class still does not accept
an object; the existing **Assign and accept** action remains explicit.

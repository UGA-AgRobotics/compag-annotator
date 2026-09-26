import { processingDialog } from "./tiling.js";
import {
  el,
  button,
  input,
  select,
  field,
  card,
  row,
  heading,
  badge,
  details,
  form,
  notify,
} from "./ui.js";
export async function helpPage(app) {
  let doctor;
  try {
    doctor = await app.api.get("/api/doctor");
  } catch (e) {
    doctor = { error: e.message };
  }
  const guide = (title, steps) =>
    card(
      title,
      el(
        "ol",
        {},
        steps.map((text) => el("li", {}, text)),
      ),
    );
  return el(
    "div",
    {},
    heading(
      "Help & settings",
      "An offline guide to your local annotation workspace.",
    ),
    el(
      "div",
      { class: "help-grid" },
      guide("What to do after SAM creates masks", [
        "Follow the Next step guide above the image. It updates from the saved annotation state as you work.",
        "Generate masks saves proposals. In Jobs, choose Open annotation results to continue reviewing the correct image.",
        "SAM point or box assistance first shows a preview. Inspect it, then save the preview as a draft. Leaving without saving discards the preview.",
        "Choose Open labeling & review to reveal pending masks, including hidden ones. Click objects or Shift/Ctrl-click several, choose or create a class, then Assign class.",
        "Review each mask and use Accept selected or Reject selected. A class label alone is not a review decision.",
        "If Mark reviewed says Image review is not finished, your annotations are saved. The dialog counts labeled and unassigned masks that still need a decision, including hidden ones. Choose Show remaining masks; Review next pending mask selects another object without accepting it.",
        "Once no masks are pending, check the full image for missing objects. Choose Confirm image review and explicitly confirm the review dialog.",
        "Continue with Review next image, or Export annotations. Prepare training is optional; it still requires separate reviewed validation images. The guide never changes review decisions, dataset roles or model settings for you.",
      ]),
      guide("Devices, SAM settings and night mode", [
        "Choose Theme → Dark / night in the header. Image pixels and annotation colors do not change.",
        "Generate masks and Train detect CPU/GPU in their respective model runtimes. Choose the named GPU, or CPU; Refresh devices checks again.",
        "Jobs shows byte-based download percentages and completed runtime setup steps. Unknown download totals stay indeterminate.",
        "New SAM2 automatic jobs use the paper recipe: 64 points per side, batch 512, predicted IoU 0.80 and stability 0.88. Edit Advanced automatic mask settings for your hardware.",
        "In Annotate → SAM settings, edit device, logit threshold, alternatives and precision for SAM2 or local-weight SAM3. SAM3 requires CUDA and still supports point/box assistance only.",
      ]),
      guide("Start with manual annotation", [
        "Create a project in Projects. Choose a name and optional local storage folder.",
        "Add your object classes in Labels. Colors and shortcuts are optional; names can change without changing class identities.",
        "Import full images in Images. Choose image files, a browser folder, or an approved local path. “Include subfolders” controls recursion.",
        "Open an image. Draw boxes or polygons, or paint a new mask. Edits autosave after each gesture.",
        "Accept each reviewed object and reject unwanted proposals. Mark the whole image reviewed only after checking for missing objects.",
        "Export a native backup or a compatible annotation format. AI setup and rounds are optional.",
      ]),
      guide("Edit full images", [
        "Scroll to zoom around the pointer. Hold Space and drag, or use Pan. Fit shows the whole image; 1:1 shows original resolution.",
        "Draw box: drag between two corners. In Edit geometry, drag the body to move it or drag a corner to resize.",
        "Draw polygon: click vertices and press Enter, double-click, or click the first vertex to finish. Escape discards an unfinished outline.",
        "In Edit geometry, drag polygon vertices; double-click an edge to add a vertex; Shift-click a vertex to remove it. Crossing edges are rejected with a repair message.",
        "Add mask pixels paints a new or selected mask. Subtract removes pixels, including holes. Separate components remain one instance. Convert a vector shape to a mask explicitly before brushing.",
        "Use the object list or Alt-click to select overlapping objects. Toggle class/proposal layers and adjust opacity in the inspector.",
        "Undo and redo use persistent history. A failed save stays visible: retry it, or reload the saved state.",
      ]),
      guide("Generate, label and merge masks", [
        "Create a project with zero classes if you prefer. Import full images, open one and choose Generate masks. SAM2 automatic generation does not require YOLO or class names.",
        "Choose whole image or overlapping tiles: 256, 512, 1024, or a custom rectangle such as 640 × 384. Use percent or pixel overlap independently for horizontal and vertical axes. The server preview shows exact tile count, stride, edge coverage and shaded overlap.",
        "512 × 512 with 25% overlap is the new-project default. Changing a job dialog affects that job; Save as project default changes future jobs only. Existing jobs and their saved tile results keep their original settings.",
        "Sampling density controls point prompts per crop. Internal crop layers and small-region cleanup are currently disabled by the supported runtime. Model scores are quality estimates, not class probabilities or guarantees of recall.",
        "Generated regions are unassigned proposals in a separate layer. Partial jobs retain completed tiles. Resume unfinished tiles continues that layer; New layer and Replace unreviewed proposals are deliberate separate choices. Replacement protects accepted or edited objects.",
        "Select mode only selects. Hover highlights the exact mask. Shift/Ctrl-click toggles multi-selection, Alt-click cycles overlapping masks, and object-list checkboxes offer the same control. Create a class in the picker and assign it to all selected objects while retaining separate instances.",
        "Paint class mode labels the mask you click with the chosen class. Assign class does not accept geometry. Assign and accept, Merge and accept, individual acceptance and image-complete review are explicit separate actions.",
        "Merge selected masks previews the exact pixel union on the original image. Holes and disconnected components remain. Choose a target class explicitly if parents disagree. Boxes need explicit conversion to masks first. The parents become superseded and the child starts as a draft unless you choose Merge and accept. Persistent Undo restores all parents together.",
        "Merge cannot invent missing pixels between fragments. Refine with SAM or brush correction is a separate action with its own review. Hiding objects is display-only; rejection and deletion affect the active annotation set and can be undone.",
        "Tile-edge recovery is OFF unless explicitly enabled for a new generation job or a selected annotation layer/proposal scope. It applies only to internal crop seams, not physical image edges; whole-image generation has no internal seam. Recovered geometry is staged for comparison. Training preparation retains its own separate opt-in. Ordinary selection, assignment, merge, upload and export do not run recovery.",
      ]),
      guide("Use optional SAM assistance", [
        "SAM points are added directly, even over existing or hidden masks. Saved masks are not scanned to prevent another prompt. Use Refine one object when you want to correct an existing mask.",
        "There is no fixed SAM point-count limit. One mask per point processes prompts sequentially with progress and cancellation; time and saved preview size grow with the number of points.",
        "Label & accept SAM Assist drafts lets you choose selected drafts or all SAM Assist drafts in the current image, choose an existing class and confirm that you inspected them. It preserves separate instances and supports persistent Undo/Redo. It does not complete full-image review.",
        "Selection provides Color for all masks and separate display colors for selected masks, including masks generated earlier. Reset all mask colors restores class colors. Colors stay in this browser for the project image; annotation data and exports are unchanged. Show mask fill controls saved overlays. No separate overlap computation runs.",
        "In Models & AI, install a chosen SAM2 runtime/checkpoint, or register trusted local weights. Review source and license information first.",
        "SAM3 has a runtime installation action and a separate local checkpoint path. SAM3 weight downloads are disabled.",
        "One mask per point: place a positive point inside each object, choose Preview N masks, then Add all N masks as drafts. Separate instances are preserved; the whole addition supports Undo.",
        "Choose Refine one object for multiple positive/negative points or a box describing one instance. Refine with SAM selects this mode for an existing mask.",
        "Hidden counts distinguish pending review, accepted and rejected masks. A class assignment is not acceptance. Review N pending masks brings unfinished decisions into view.",
        "Draw a SAM box prompt, or select an existing shape and choose Refine with SAM.",
        "Request Preview mask. Inspect alternative masks. Use preview as draft saves geometry; it does not mark the object human verified.",
        "Changing the image, object, prompts, class, model or revision invalidates old previews. Accept the saved object only after review.",
      ]),
      guide("Rounds & training", [
        "For disconnected or hole-containing masks blocked by strict YOLO conversion, use Exclude incompatible objects in Train. Select objects and confirm the background-label acknowledgement. Original annotations and exports stay intact, and lossy conversion is not enabled. Review training exclusions restores selected objects. These choices apply to future training and validation snapshots across supported YOLO models; resumed jobs retain their original frozen data.",
        "Omitted objects become background in YOLO labels, which can affect training and validation metrics. To avoid partially labeled images, set the whole image role to excluded instead. An image with all its accepted objects excluded is omitted, not treated as an intentional negative. Separate reviewed training and validation images are still required.",
        "Train opens its settings while Data readiness checks image review and mask geometry. Large masks can take longer; an indeterminate indicator and elapsed time show that checking continues. A failed check offers Retry data check without clearing settings. Training still waits for validation.",
        "In Images, choose Dataset role, then Confirm review under each chosen image. The dialog shows accepted and pending counts; explicitly approve labeled drafts only after inspecting them, and confirm the full-image review. Unassigned objects need a class or rejection in Annotate. Empty images need a separate intentional-negative confirmation.",
        "Use Continue to Train to load fresh readiness. Other unfinished images may remain in the project and are excluded. Confirmation preserves masks/classes and starts no training job; Undo in Annotate reverses it.",
        "Assign pool/train/validation/test roles in Images. Related-image groups must stay in one split. Fixed validation/test images are excluded from round planning.",
        "If no validation image is assigned, Train shows Choose image roles before training. Use Set image roles, then the Dataset role dropdown under each image. Keep at least one reviewed image as train or pool and a different reviewed image as validation. Mark reviewed does not assign roles. Role changes save automatically and preserve completed review; return to Train and Refresh readiness.",
        "Choose a round count between 1 and the eligible pool size. Preview import order, deterministic shuffle or a manual image order before confirmation.",
        "Start a round, review its full images and finish review. Round completion does not imply training ran.",
        "Train shows readiness, class counts, negative images and exclusions. Complete accepted polygons/masks are required; boxes alone are not segmentation truth.",
        "Choose a base model, epochs, image size, batch, device and seed. The job freezes cumulative reviewed data and excludes incomplete images.",
        "New YOLO training jobs adapt batches to small datasets and verify actual learnable-weight changes. Inspect Weight updates verified and the validation metrics in Jobs. For an old ineffective short run, start a new job from the original base model. Every object class you want the model to learn needs reviewed training examples; extra epochs cannot replace missing examples.",
        "Tile-edge checking defaults off. If explicitly enabled during training preparation, review proposed changes and retry training after resolving them.",
        "Inspect real training logs, checkpoints and validation in Jobs. Activate a validated model explicitly in Models & AI and map its classes. Roll back restores a prior activation.",
      ]),
      guide("Import, export & recovery", [
        "Native backup preserves canonical masks, class identities, review provenance, rounds and model metadata. Weight inclusion is not enabled. Restore creates a separate project.",
        "COCO RLE and per-instance PNG preserve holes/components. Flat-polygon exports may not preserve topology. Strict mode blocks unsupported geometry.",
        "Explicit approximate export records changed geometry. Box-only formats do not carry segmentation boundaries. Single-plane PNG requires an overlap conflict policy.",
        "Imported annotations begin unreviewed. Review the reported class mapping and conversion details before treating them as ground truth.",
        "Reference-mode images need their original source files. Image Details offers relinking to an approved matching original. Removing an image from the project preserves its original source.",
        "Jobs shows failures, cancellation and retry. Training resume requires a real supported checkpoint; it is not an arbitrary pause/resume button.",
      ]),
      card(
        "Keyboard shortcuts",
        el(
          "div",
          { class: "data-list" },
          [
            ["V", "Select"],
            ["G", "Edit geometry"],
            ["L", "Paint class"],
            ["Shift / Ctrl + click", "Toggle selection"],
            ["H / Space + drag", "Pan"],
            ["Right mouse button + drag", "Pan, including while placing SAM points"],
            ["Ctrl + D", "Remove the last SAM prompt point (saved masks stay unchanged)"],
            ["B", "Draw box"],
            ["P", "Draw polygon"],
            ["D", "Add mask pixels"],
            ["E", "Subtract mask pixels"],
            ["Enter", "Finish polygon"],
            ["Escape", "Cancel current gesture / polygon"],
            [
              "Delete / Backspace",
              "Delete selected object / last draft vertex",
            ],
            ["Ctrl / ⌘ + Z", "Undo"],
            ["Ctrl / ⌘ + Shift + Z", "Redo"],
            ["Alt + click", "Cycle overlapping objects"],
          ].map(([key, label]) =>
            row(el("kbd", { class: "kbd" }, key), el("span", {}, label)),
          ),
        ),
      ),
    ),
    el(
      "div",
      { class: "columns", style: "margin-top:20px" },
      card(
        "Local settings",
        app.project &&
          button("Project processing defaults", () => processingDialog(app)),
        form(
          [
            field(
              "Default AI device",
              input("device", app.device),
              "Examples: cpu or cuda:0. Availability is reported by the provider; no silent fallback is requested.",
            ),
            field(
              "Interface language",
              select("language", [["en", "English"]], "en"),
            ),
            el(
              "p",
              {},
              "Device preference is saved only in this browser. No image, annotation, checkpoint path or session token is stored in browser preferences.",
            ),
          ],
          async (v) => {
            app.device = v.device.trim() || "cpu";
            localStorage.setItem("compag-device", app.device);
            notify("Local preference saved.");
          },
          "Save preferences",
        ),
        details("Storage, device & core diagnostics", doctor, true),
        button("Create redacted diagnostic export", async () =>
          app.showJob(await app.api.post("/api/diagnostics")),
        ),
      ),
      card(
        "Privacy, licenses & troubleshooting",
        el(
          "p",
          {},
          "Images, annotations and model training stay on this computer. The interface is bundled locally and needs no CDN or npm runtime. Optional downloads require explicit consent.",
        ),
        el(
          "p",
          {},
          "SAM2, SAM3 and Ultralytics have separate code/weight terms. Inspect the catalog record at installation. Original-source redistribution rights and human release acceptance remain separate gates.",
        ),
        el(
          "p",
          {},
          "If AI setup fails, inspect Jobs and device diagnostics; manual editing and export remain available. Missing GPU, network or checkpoint files must not be interpreted as successful provider validation.",
        ),
        el(
          "p",
          {},
          "If a revision conflict appears, another operation changed the project. Keep the failure visible and reload before making a new edit.",
        ),
        el(
          "p",
          {},
          "This application supports 2-D still-image annotation. Video, medical imaging formats, 3-D and remote team hosting are outside this release.",
        ),
        row(
          badge(`Version ${app.api.session.version}`),
          badge(
            app.api.session.qa_mode
              ? "Automated QA session"
              : "Local review session",
          ),
        ),
      ),
    ),

  );
}

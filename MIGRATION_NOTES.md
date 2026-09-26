# Migration and preservation

COMPAG Annotator 1.0.0rc1 is a separate product and data root. It does not modify or downgrade a historical research installation, migrate old scientific state in place, or borrow an old model's learned class meanings.

Create a new project and import supported image/annotation copies. Classes may be defined before annotation or after automatic proposal generation; zero classes is a valid starting state. Standard imports retain source provenance and begin unreviewed until explicit attestation. There is no advertised general legacy database importer. Preserve any original source labels when preparing an explicit, versioned conversion; review its class/geometry loss report.

Class IDs are stable; trained models keep their saved contiguous class mapping. Changes to project classes require compatibility review before activation/inference. Existing binary probabilities cannot become multiclass predictions by renaming labels.

Back up projects before application/schema updates. Restore into a new location. Application rollback restores code selection, not project schema or data. Model activation/rollback is separate and must not overwrite historical annotations or original research model pointers.

No research parity, B/C acceleration integration, automatic color cropping, perspective warping or resumed benchmark is claimed. Reused geometry concepts and historical defaults are documented in SOURCE_REUSE_MAP.md; the narrow changes below take precedence for new addendum jobs.

## Addendum 01 preservation rules

The implementation is being integrated into the same unreleased 1.0.0rc1 candidate. Original specifications, preserved research inputs, completed model runs and their evidence remain historical inputs. The addendum does not validate an old artifact against new UI/API behavior; the coordinator must rebuild and check the frozen candidate before delivery.

New projects use actual 512×512 source crops with 25% overlap as an initial default. Existing recorded tile grids/jobs are not rewritten. Legacy projects with no recorded processing choice remain unset until the user explicitly chooses; a migration must not backfill a fabricated historical grid. New job manifests freeze settings, source/image identity and transforms. Changing presets, independent Custom dimensions or overlap affects new jobs only, never accepted masks or completed/running jobs. Cache entries bind image/model/preprocessing/crop identity. Inference tiling does not change full-image training defaults, historical snapshots or separate training-crop settings.

Generated unassigned masks use the existing canonical annotation state and stable IDs; they do not create a second dataset or an implicit background class. New generation layers preserve completed tile receipts and explicit partial status. Replace only a deliberately selected unreviewed layer, retaining accepted/edited geometry, rejections and merge history. Explicit merge creates a new draft child and supersedes parents without deleting lineage; undo/redo restores the appropriate active set. Class changes and geometry edits require fresh completeness review for future snapshots. Existing models keep their own class-index maps.

The old **training-only** heavy recovery restriction is superseded by two contexts: **training preparation** and **automatic-mask annotation**. Recovery is still **OFF by default**, with explicit consent for each scoped job and review before geometry changes. No migration, saved checkbox or earlier job authorizes a new recovery run. Import, image opening, manual edits/labels/prompts/merge, ordinary inference, rounds and export do not trigger it automatically. Manual merge is exact pixel union; it is not permission to enable Fast-Stitching, automatic neighbor union or research-specific recovery.

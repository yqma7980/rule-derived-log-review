# Public release checklist

- [x] MIT selected and included; author requested public software release.
- [x] Preserve baseline scientific Python files and original source evidence.
- [x] Keep controlled data, research tables, third-party snapshots and private
  author records out of the public payload.
- [x] Provide data-free examples, input schemas and chain-specific environments.
- [x] Identify the intended repository and tag without claiming publication.
- [x] Publisher: review VALIDATION_SUMMARY.json and final manifest against the
  audited payload; do not include sibling author_notes/ or qa/ directories.
- [x] Publisher: finalize README/LICENSE_STATUS and release_metadata.json with
  truthful publication state, then regenerate hashes before fixing the final tag.
  Preserve .gitattributes so Git does not alter the audited file bytes.
- Release access and the resolved commit are checked externally after publishing
  the fixed tag, not self-certified inside its immutable payload. Downloaders can
  resolve the tag and verify every file against PACKAGE_MANIFEST.json.
- Manuscript/SI use the actual repository, fixed tag and MIT scope; controlled
  data access remains a separate mechanism.
- [ ] Author: give final approval of the manuscript and submission materials.
  This remains pending; public MIT code release was authorized separately.
- [ ] Parent/author: complete journal-submission approval independently.

Missing controlled inputs
are an access/reproduction boundary, not a pending software-license choice.
Fresh installation, cross-platform exact replay and full model retraining are
not certified by interface tests. Research outputs must not be added to this
software-only release without separate permission and an explicit data license.

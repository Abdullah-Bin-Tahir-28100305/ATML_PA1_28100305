# shared/splits/

This folder holds the **shared PACS train/val split** used by BOTH Task 2 and
Task 3, so the two tasks train on identical source splits (an assignment
requirement).

- `pacs_sketch_seed6304.json` is created **automatically** the first time you run
  training (`shared/pacs_protocol.build_or_load_splits`). You do not create it by
  hand.
- It stores, for each source domain (photo, art_painting, cartoon), the stratified
  80/20 train/val **image indices** (into that domain's ImageFolder), generated
  with **seed 6304**. Saving indices (not pixels) makes the split perfectly
  reproducible and shareable across tasks.
- Once it exists, it is reused verbatim on every subsequent run and by Task 3.
  Delete it only if you deliberately want to regenerate the splits.

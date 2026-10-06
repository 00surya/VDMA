# Training a replacement weapon detector

Open [the Colab notebook](../notebooks/train_vdma_weapons_colab.ipynb) at [Google Colab](https://colab.research.google.com/) using **File → Upload notebook**, then select **Runtime → Change runtime type → T4 GPU**. The notebook is standalone; it does not need the VDM repository, a Roboflow key, or Twilio credentials.

The current model is Assalim Normal_Compressed YOLOv8n, with `0: guns`, `1: knife`. Inference requests candidates at a 0.90 threshold, and public weapon detections must be strictly above 0.90. The mapping is checked when loading; inspection did not identify a swapped-class bug. A high confidence score is not proof that an object is a gun. The reported phones/screens are consistent with a model/data generalization problem, but no labelled sample from those failures has been evaluated in this update.

## Dataset choices

| Source | Use in this notebook | Limitations |
| --- | --- | --- |
| [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html) | Automatic, no-key download of Handgun, Rifle, Shotgun, Knife and Kitchen knife boxes. Phone, laptop, tablet, remote, television, drill and scissors images supply **candidates** for negative review. | General photographs, not a CCTV benchmark. Missing weapon annotations do **not** prove an image is weapon-free. Human review is required before candidate negatives enter training. V7 reuses the official V6/V5 box files. |
| [YouTube-GDD](https://github.com/UCAS-GYX/YouTube-GDD) | Recommended additional gun positives; download separately and import reviewed YOLO labels through the notebook's local-positive folder layout. | Authors describe 5,000 images from 343 videos. Their original class IDs are `0: person`, `1: gun`; remap gun to `0`, omit person, and label any knives. Some source videos span adjacent published folds: split by original video/session, not individual frame. The repository links pre-extracted images because many video links are dead. Repository code licensing does not settle the rights to source videos. |
| [CCTV-Gun](https://github.com/srikarym/CCTV-Gun/blob/main/dataset_instructions.md) | Recommended later external surveillance evaluation, with independent footage and manually checked class remapping. | Aggregates MGD, USRT and UCF sources with separate acquisition instructions. It is not automatically downloaded, converted, or mixed into the notebook's test split. Check each source's terms. |
| Your camera's weapon-free footage | Most relevant negatives: phones at different angles, laptop screens, remotes, bottles, hands, chairs, desks, low light and movement. The notebook extracts sparse frames and requires review. | Keep an entire recording session in one split. Use different rooms, days and people in held-out sessions. Do not label a visible gun or knife as background. Do not put a real weapon in a scene just to collect training footage. |

Open Images [publishes annotation and image-license information](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#licenses): annotations are CC BY 4.0; images are listed as CC BY 2.0, with per-image verification advised by the publisher. The notebook saves selected image attribution metadata. Ultralytics and optional baseline weights have their own licenses; source and license references accompany the export. Do not publish private camera data with the notebook.

## What the notebook does

1. Pins Ultralytics to the app's installed version, saves configuration and package versions, optionally mounts Drive, and checks GPU availability.
2. Downloads official annotations and a bounded image subset. The training box CSV alone is about **2.26 GB**; allow download/preparation time and several GB of scratch disk. Filtered annotation caches and reviews persist to Drive; bulk CSVs and images use Colab scratch storage.
3. Preserves public train/validation/test splits and local recording-session boundaries, validates boxes, and removes exact/visually near duplicates across splits conservatively.
4. Presents an explicit negative-review UI. Only images marked **no gun and no knife** become empty YOLO label files. There is no `background` class and no automatic conversion of unlabeled images into negatives. Ambiguous images can be excluded; positive labels must be corrected before importing them.
5. Fine-tunes a fresh COCO-pretrained **YOLOv8s**, retaining the existing `guns`, `knife` class order. This avoids initializing from the problematic specialist. `TRAIN_FROM_SCRATCH=True` is available for a controlled comparison; it is not the recommended first Colab run. Change `MODEL_SIZE` to `n` if runtime latency matters more. Use a new run name for each experiment.
6. Saves recoverable checkpoints to Drive and supports explicit resume. Evaluates per-class detection metrics and weapon-free-image false-alarm rates. Selects a gun threshold on **validation only**, requiring a false-positive target, minimum gun recall and separate handgun recall; it reports failure rather than recommending a threshold that hides every gun. The checked Open Images validation annotations have only 17 usable handgun images before duplicate filtering. That small subset is a starting point: add independent labelled handgun examples rather than letting rifle performance hide handgun misses.
7. Evaluates the fixed threshold on untouched test images, optionally compares the pinned current detector, and exports weights, SHA-256, dataset/provenance manifest and reports. Frame false-positive rates are not operational false alerts per hour; test longer camera sequences before deployment.

The notebook does not change the running app or make calls. Training and held-out evaluation must finish before replacing its detector. Bring back `vdma-weapon-model.zip`; integration must update the pinned filename/checksum and both threshold gates in `vmd/objects.py`, retain restricted checkpoint loading/class-map checks, and verify CPU latency. A matching class map alone does not make a new checkpoint deployable.

## Local footage layout

The configuration cell prints `USER_DATA`. It is a folder in your mounted Drive by default. Put files there before the local-data cell:

```text
user_data/
  negative_clips/
    train/session_a/phone_and_laptop.mp4
    val/session_b/different_day.mp4
    test/session_c/different_room.mp4
  negative_images/
    train/session_d/photo.jpg
  positives/
    train/images/session_e/frame.jpg
    train/labels/session_e/frame.txt
    val/images/session_f/frame.jpg
    val/labels/session_f/frame.txt
    test/images/session_g/frame.jpg
    test/labels/session_g/frame.txt
```

Each positive label must explicitly contain **all** visible guns/knives using normalized `class cx cy width height` lines (`0: guns`, `1: knife`). Empty/missing labels in the positive folder are errors, not accepted negatives. Use the reviewable negative folders for genuinely weapon-free examples. Different clips from one recording session must use the same session folder and split. Training does not learn automatically from the live app or its review buttons.

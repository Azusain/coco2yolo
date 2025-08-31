# COCO to YOLO Converter

Converts COCO format annotations to YOLO format for training YOLO models.
Supports both standard COCO format and DAMM dataset custom format.

## Usage

```bash
cargo build --release
cargo run -- --input /path/to/dataset --output /path/to/output
```

### Options

- `--input`, `-i`: Input directory with COCO JSON files
- `--output`, `-o`: Output directory for YOLO files
- `--format`: `damm` (default) or `standard`
- `--create-classes`: Generate classes.txt (default: true)

### DAMM Dataset Example

```bash
cargo run -- --input C:\Downloads\detection_datasets --output .\yolo_output
```

## Output

- One `.txt` file per image: `class_id x_center y_center width height` (normalized 0-1)
- `classes.txt`: Class definitions

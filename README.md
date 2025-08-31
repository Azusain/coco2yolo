# COCO to YOLO Format Converter

A fast Rust-based tool to convert COCO format annotations to YOLO format, with support for both standard COCO and custom DAMM dataset formats.

## ✨ Features

- 🔄 **Multiple Format Support**: Handles both standard COCO format and DAMM dataset format
- 📁 **YOLO Directory Structure**: Creates proper `images/` and `labels/` folders with train/val splits
- 🎲 **Random Train/Val Split**: Configurable split ratio (default 80% train, 20% validation)
- 🔍 **Image File Discovery**: Automatically finds and copies corresponding image files
- 📝 **Class File Generation**: Creates `classes.txt` with detected class names
- ⚡ **Fast Processing**: Written in Rust for optimal performance
- 📂 **Flexible Input**: Recursively processes multiple JSON files in directory structure
- 📊 **Progress Bars**: Beautiful progress indicators for JSON parsing and image processing

## 🚀 Usage

### Build
```bash
cargo build --release
```

### Run
```bash
./target/release/coco_to_yolo --input /path/to/dataset --output /path/to/output
```

### 📋 Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | Input directory with COCO JSON files | Required |
| `--output` | `-o` | Output directory for YOLO files | Required |
| `--format` | | Dataset format: `damm` or `standard` | `damm` |
| `--train-split` | | Training split ratio (0.0-1.0) | `0.8` |
| `--yolo-structure` | | Create YOLO directory structure | `true` |
| `--create-classes` | | Generate classes.txt file | `true` |

### 💡 Examples

**DAMM Dataset:**
```bash
./target/release/coco_to_yolo --input ./datasets --output ./yolo_dataset --format damm --train-split 0.8
```

**Standard COCO:**
```bash
./target/release/coco_to_yolo --input ./coco_data --output ./yolo_data --format standard --train-split 0.9
```

## 📤 Output Structure

```
output_directory/
├── classes.txt                 # Class definitions
├── images/
│   ├── train/                  # Training images
│   └── val/                    # Validation images
└── labels/
    ├── train/                  # Training labels (.txt files)
    └── val/                    # Validation labels (.txt files)
```

**Label Format:** Each `.txt` file contains one line per object:
```
class_id x_center y_center width height
```
*All coordinates are normalized (0.0-1.0)*

## 🎯 Progress Display

The tool shows three progress bars during conversion:
1. **JSON Parsing** - Processing metadata files
2. **Image Processing** - Copying images and generating labels
3. **Completion Summary** - Final statistics

## 🔧 Development

**Requirements:**
- Rust 1.70+
- Cargo

**Dependencies:**
- `clap` - Command line argument parsing
- `serde` - JSON serialization/deserialization
- `walkdir` - Directory traversal
- `rand` - Random shuffling for train/val split
- `indicatif` - Progress bars
- `anyhow` - Error handling

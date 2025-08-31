use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use rand::seq::SliceRandom;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Parser)]
#[command(name = "coco-to-yolo")]
#[command(about = "Convert COCO format annotations to YOLO format")]
struct Args {
    /// Input directory containing COCO JSON files
    #[arg(short, long)]
    input: PathBuf,

    /// Output directory for YOLO format files
    #[arg(short, long)]
    output: PathBuf,

    /// Create classes.txt file with class names
    #[arg(long, default_value_t = true)]
    create_classes: bool,

    /// Format type: 'standard' for standard COCO format, 'damm' for DAMM dataset format
    #[arg(long, default_value = "damm")]
    format: String,
    
    /// Training split ratio (0.0 to 1.0)
    #[arg(long, default_value = "0.8")]
    train_split: f64,
    
    /// Create YOLO directory structure (images/labels with train/val splits)
    #[arg(long, default_value_t = true)]
    yolo_structure: bool,
}

// DAMM format annotation (custom format)
#[derive(Debug, Deserialize)]
struct DammAnnotation {
    bbox: Vec<Vec<f64>>, // [[x1, y1], [x2, y2]] format
    category_id: u32,
    #[serde(default)]
    bbox_mode: Option<String>, // BoxMode.XYXY_ABS
    #[serde(default)]
    segmentation: Option<Vec<Vec<f64>>>,
}

// DAMM format image structure
#[derive(Debug, Deserialize)]
struct DammImage {
    file_name: String,
    height: u32,
    width: u32,
    image_id: u32,
    annotations: Vec<DammAnnotation>,
}

// DAMM format dataset
#[derive(Debug, Deserialize)]
struct DammDataset {
    annotations: Vec<DammImage>,
}

// Standard COCO format annotation
#[derive(Debug, Deserialize)]
struct CocoAnnotation {
    id: u32,
    image_id: u32,
    category_id: u32,
    bbox: Vec<f64>, // [x, y, width, height] format (standard COCO)
    area: f64,
    #[serde(default)]
    iscrowd: u32,
    #[serde(default)]
    segmentation: Option<serde_json::Value>,
}

// Standard COCO format image
#[derive(Debug, Deserialize)]
struct CocoImageInfo {
    id: u32,
    file_name: String,
    height: u32,
    width: u32,
}

// Standard COCO format dataset
#[derive(Debug, Deserialize)]
struct CocoDataset {
    images: Vec<CocoImageInfo>,
    annotations: Vec<CocoAnnotation>,
    #[serde(default)]
    categories: Option<Vec<serde_json::Value>>,
}

// Unified annotation format for processing
#[derive(Debug)]
struct UnifiedAnnotation {
    bbox: Vec<f64>, // Always in [x1, y1, x2, y2] format
    category_id: u32,
}

// Unified image format for processing
#[derive(Debug)]
struct UnifiedImage {
    file_name: String,
    height: u32,
    width: u32,
    annotations: Vec<UnifiedAnnotation>,
}

#[derive(Debug)]
struct YoloAnnotation {
    class_id: u32,
    x_center: f64,
    y_center: f64,
    width: f64,
    height: f64,
}

impl YoloAnnotation {
    fn from_unified(ann: &UnifiedAnnotation, img_width: u32, img_height: u32) -> Self {
        // Unified bbox format: [x1, y1, x2, y2] where (x1,y1) is top-left, (x2,y2) is bottom-right
        let x1 = ann.bbox[0];
        let y1 = ann.bbox[1];
        let x2 = ann.bbox[2];
        let y2 = ann.bbox[3];

        // Convert to YOLO format (normalized coordinates)
        let bbox_width = x2 - x1;
        let bbox_height = y2 - y1;
        let x_center = (x1 + bbox_width / 2.0) / img_width as f64;
        let y_center = (y1 + bbox_height / 2.0) / img_height as f64;
        let norm_width = bbox_width / img_width as f64;
        let norm_height = bbox_height / img_height as f64;

        YoloAnnotation {
            class_id: ann.category_id,
            x_center,
            y_center,
            width: norm_width,
            height: norm_height,
        }
    }

    fn to_string(&self) -> String {
        format!(
            "{} {:.6} {:.6} {:.6} {:.6}",
            self.class_id, self.x_center, self.y_center, self.width, self.height
        )
    }
}


fn parse_damm_format(content: &str) -> Result<Vec<UnifiedImage>> {
    let dataset: DammDataset = serde_json::from_str(content)?;
    let mut unified_images = Vec::new();
    
    for damm_image in dataset.annotations {
        let mut unified_annotations = Vec::new();
        
        for damm_ann in damm_image.annotations {
            // Convert DAMM [[x1, y1], [x2, y2]] to unified [x1, y1, x2, y2]
            let unified_ann = UnifiedAnnotation {
                bbox: vec![damm_ann.bbox[0][0], damm_ann.bbox[0][1], damm_ann.bbox[1][0], damm_ann.bbox[1][1]],
                category_id: damm_ann.category_id,
            };
            unified_annotations.push(unified_ann);
        }
        
        let unified_image = UnifiedImage {
            file_name: damm_image.file_name,
            height: damm_image.height,
            width: damm_image.width,
            annotations: unified_annotations,
        };
        unified_images.push(unified_image);
    }
    
    Ok(unified_images)
}

fn parse_standard_format(content: &str) -> Result<Vec<UnifiedImage>> {
    let dataset: CocoDataset = serde_json::from_str(content)?;
    let mut unified_images = Vec::new();
    
    // Create a map of image_id to image info
    let mut image_map: HashMap<u32, &CocoImageInfo> = HashMap::new();
    for image in &dataset.images {
        image_map.insert(image.id, image);
    }
    
    // Group annotations by image_id
    let mut annotations_by_image: HashMap<u32, Vec<&CocoAnnotation>> = HashMap::new();
    for annotation in &dataset.annotations {
        annotations_by_image.entry(annotation.image_id)
            .or_insert_with(Vec::new)
            .push(annotation);
    }
    
    // Convert to unified format
    for (image_id, image_info) in image_map {
        let mut unified_annotations = Vec::new();
        
        if let Some(annotations) = annotations_by_image.get(&image_id) {
            for coco_ann in annotations {
                // Convert COCO [x, y, width, height] to unified [x1, y1, x2, y2]
                let x1 = coco_ann.bbox[0];
                let y1 = coco_ann.bbox[1];
                let x2 = x1 + coco_ann.bbox[2];
                let y2 = y1 + coco_ann.bbox[3];
                
                let unified_ann = UnifiedAnnotation {
                    bbox: vec![x1, y1, x2, y2],
                    category_id: coco_ann.category_id,
                };
                unified_annotations.push(unified_ann);
            }
        }
        
        let unified_image = UnifiedImage {
            file_name: image_info.file_name.clone(),
            height: image_info.height,
            width: image_info.width,
            annotations: unified_annotations,
        };
        unified_images.push(unified_image);
    }
    
    Ok(unified_images)
}

fn find_image_file(input_dir: &Path, image_filename: &str) -> Option<PathBuf> {
    // Common image extensions to search for
    let extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "tif"];
    
    // Try with the exact filename first
    for entry in WalkDir::new(input_dir).into_iter().filter_map(|e| e.ok()) {
        if let Some(file_name) = entry.path().file_name() {
            if file_name.to_str().unwrap_or("") == image_filename {
                return Some(entry.path().to_path_buf());
            }
        }
    }
    
    // If not found, try with different extensions
    let base_name = Path::new(image_filename).file_stem()?.to_str()?;
    for ext in &extensions {
        let search_name = format!("{}.{}", base_name, ext);
        for entry in WalkDir::new(input_dir).into_iter().filter_map(|e| e.ok()) {
            if let Some(file_name) = entry.path().file_name() {
                if file_name.to_str().unwrap_or("") == search_name {
                    return Some(entry.path().to_path_buf());
                }
            }
        }
    }
    
    None
}

fn convert_coco_to_yolo(
    input_dir: &Path, 
    output_dir: &Path, 
    create_classes: bool, 
    format: &str,
    train_split: f64,
    yolo_structure: bool
) -> Result<()> {
    fs::create_dir_all(output_dir).context("Failed to create output directory")?;

    let mut all_images = Vec::new();
    let mut class_names = HashMap::new();
    let mut processed_files = 0;
    let mut total_annotations = 0;

    println!("Using format: {}", format);
    println!("Scanning for metadata files...");
    
    // Find all JSON files first
    let mut json_files = Vec::new();
    for entry in WalkDir::new(input_dir).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            json_files.push(path.to_path_buf());
        }
    }
    
    if json_files.is_empty() {
        anyhow::bail!("No JSON files found in input directory");
    }
    
    println!("Found {} JSON files", json_files.len());
    
    // Create progress bar for JSON parsing
    let pb_parse = ProgressBar::new(json_files.len() as u64);
    pb_parse.set_style(
        ProgressStyle::with_template(
            "Parsing JSON    [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}"
        )?
        .progress_chars("#>-")
    );
    
    // Parse all JSON files with progress bar
    for json_file in &json_files {
        let filename = json_file.file_name().unwrap_or_default().to_string_lossy();
        pb_parse.set_message(format!("Processing {}", filename));
        
        let content = fs::read_to_string(json_file)
            .with_context(|| format!("Failed to read file: {}", json_file.display()))?;
        
        let unified_images = match format {
            "standard" => {
                parse_standard_format(&content)
                    .with_context(|| format!("Failed to parse as standard COCO format: {}", json_file.display()))?
            },
            "damm" => {
                parse_damm_format(&content)
                    .with_context(|| format!("Failed to parse as DAMM format: {}", json_file.display()))?
            },
            _ => {
                anyhow::bail!("Invalid format '{}'. Use 'standard' or 'damm'", format);
            }
        };

        all_images.extend(unified_images);
        processed_files += 1;
        pb_parse.inc(1);
    }
    
    pb_parse.finish_with_message("JSON parsing complete");

    let total_images = all_images.len();
    println!("Found {} images total", total_images);
    
    if yolo_structure {
        // Create professional YOLO directory structure
        let train_images_dir = output_dir.join("train").join("images");
        let train_labels_dir = output_dir.join("train").join("labels");
        let val_images_dir = output_dir.join("val").join("images");
        let val_labels_dir = output_dir.join("val").join("labels");
        
        fs::create_dir_all(&train_images_dir)?;
        fs::create_dir_all(&train_labels_dir)?;
        fs::create_dir_all(&val_images_dir)?;
        fs::create_dir_all(&val_labels_dir)?;
        
        // Shuffle images for random split
        let mut rng = rand::thread_rng();
        let mut images = all_images;
        images.shuffle(&mut rng);
        
        let train_count = (images.len() as f64 * train_split) as usize;
        
        println!("Split: {} training, {} validation images", train_count, images.len() - train_count);
        
        // Create progress bar for image processing
        let pb_images = ProgressBar::new(images.len() as u64);
        pb_images.set_style(
            ProgressStyle::with_template(
                "Processing     [{elapsed_precise}] [{bar:40.green/blue}] {pos:>7}/{len:7} {msg}"
            )?
            .progress_chars("#>-")
        );
        
        let mut missing_images = 0;
        
        for (idx, image) in images.iter().enumerate() {
            let is_train = idx < train_count;
            let (images_dir, labels_dir, split_name) = if is_train {
                (&train_images_dir, &train_labels_dir, "train")
            } else {
                (&val_images_dir, &val_labels_dir, "val")
            };
            
            // Extract filename from path
            let image_filename = Path::new(&image.file_name)
                .file_name()
                .context("Invalid image filename")?
                .to_str()
                .context("Non-UTF8 filename")?;
            
            pb_images.set_message(format!("{} - {} ({} ann)", split_name, image_filename, image.annotations.len()));
            
            // Find the actual image file
            if let Some(source_image_path) = find_image_file(input_dir, image_filename) {
                let dest_image_path = images_dir.join(image_filename);
                fs::copy(&source_image_path, &dest_image_path)
                    .with_context(|| format!("Failed to copy image: {}", source_image_path.display()))?;
                
                // Create annotation file
                let base_name = Path::new(image_filename)
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap();
                let annotation_path = labels_dir.join(format!("{}.txt", base_name));
                
                let mut yolo_annotations = Vec::new();
                for annotation in &image.annotations {
                    let yolo_ann = YoloAnnotation::from_unified(annotation, image.width, image.height);
                    yolo_annotations.push(yolo_ann.to_string());
                    class_names.insert(annotation.category_id, format!("class_{}", annotation.category_id));
                    total_annotations += 1;
                }
                
                let content = if yolo_annotations.is_empty() { 
                    String::new() 
                } else { 
                    yolo_annotations.join("\n") + "\n"
                };
                
                fs::write(&annotation_path, content)
                    .with_context(|| format!("Failed to write annotation file: {}", annotation_path.display()))?;
            } else {
                missing_images += 1;
            }
            
            pb_images.inc(1);
        }
        
        pb_images.finish_with_message("Image processing complete");
        
        if missing_images > 0 {
            println!("Warning: {} image files not found", missing_images);
        }
    } else {
        // Legacy flat structure
        for image in &all_images {
            let image_name = Path::new(&image.file_name)
                .file_stem()
                .unwrap_or_default()
                .to_str()
                .unwrap_or("unknown");
            
            let output_file = output_dir.join(format!("{}.txt", image_name));
            let mut yolo_annotations = Vec::new();

            for annotation in &image.annotations {
                let yolo_ann = YoloAnnotation::from_unified(annotation, image.width, image.height);
                yolo_annotations.push(yolo_ann.to_string());
                class_names.insert(annotation.category_id, format!("class_{}", annotation.category_id));
                total_annotations += 1;
            }

            let content = if yolo_annotations.is_empty() { 
                String::new() 
            } else { 
                yolo_annotations.join("\n") + "\n"
            };
            
            fs::write(&output_file, content)
                .with_context(|| format!("Failed to write output file: {}", output_file.display()))?;
            
            println!("  -> Generated: {} ({} annotations)", output_file.display(), image.annotations.len());
        }
    }

    // Create classes.txt file
    if create_classes && !class_names.is_empty() {
        let classes_file = output_dir.join("classes.txt");
        let mut sorted_classes: Vec<_> = class_names.into_iter().collect();
        sorted_classes.sort_by_key(|(id, _)| *id);
        
        let class_content = sorted_classes
            .into_iter()
            .map(|(_, name)| name)
            .collect::<Vec<_>>()
            .join("\n") + "\n";
        
        fs::write(&classes_file, class_content)
            .with_context(|| format!("Failed to write classes file: {}", classes_file.display()))?;
        
        println!("\nGenerated classes file: {}", classes_file.display());
    }

    println!("\nConversion completed!");
    println!("Processed JSON files: {}", processed_files);
    println!("Total images: {}", total_images);
    println!("Total annotations: {}", total_annotations);
    
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.input.exists() {
        anyhow::bail!("Input directory does not exist: {}", args.input.display());
    }

    println!("Converting COCO format to YOLO format...");
    println!("Input directory: {}", args.input.display());
    println!("Output directory: {}", args.output.display());
    println!();

    convert_coco_to_yolo(&args.input, &args.output, args.create_classes, &args.format, args.train_split, args.yolo_structure)?;
    
    Ok(())
}

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

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

fn convert_coco_to_yolo(input_dir: &Path, output_dir: &Path, create_classes: bool, format: &str) -> Result<()> {
    // Create output directory
    fs::create_dir_all(output_dir).context("Failed to create output directory")?;

    let mut class_names = HashMap::new();
    let mut processed_files = 0;
    let mut total_annotations = 0;

    println!("Using format: {}", format);
    
    // Find all JSON files in input directory
    for entry in WalkDir::new(input_dir).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            println!("Processing: {}", path.display());
            
            let content = fs::read_to_string(path)
                .with_context(|| format!("Failed to read file: {}", path.display()))?;
            
            // Parse based on format
            let unified_images = match format {
                "standard" => {
                    parse_standard_format(&content)
                        .with_context(|| format!("Failed to parse as standard COCO format: {}", path.display()))?
                },
                "damm" => {
                    parse_damm_format(&content)
                        .with_context(|| format!("Failed to parse as DAMM format: {}", path.display()))?
                },
                _ => {
                    anyhow::bail!("Invalid format '{}'. Use 'standard' or 'damm'", format);
                }
            };

            // Process all unified images
            for image in unified_images {
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
                    
                    // Store class name for classes.txt
                    class_names.insert(annotation.category_id, format!("class_{}", annotation.category_id));
                    total_annotations += 1;
                }

                // Write YOLO format file (even if empty)
                let content = if yolo_annotations.is_empty() { 
                    String::new() 
                } else { 
                    yolo_annotations.join("\n") 
                };
                
                fs::write(&output_file, content)
                    .with_context(|| format!("Failed to write output file: {}", output_file.display()))?;
                
                println!("  -> Generated: {} ({} annotations)", output_file.display(), image.annotations.len());
            }
            
            processed_files += 1;
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
            .join("\n");
        
        fs::write(&classes_file, class_content)
            .with_context(|| format!("Failed to write classes file: {}", classes_file.display()))?;
        
        println!("Generated classes file: {}", classes_file.display());
    }

    println!("\nConversion completed!");
    println!("Processed files: {}", processed_files);
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

    convert_coco_to_yolo(&args.input, &args.output, args.create_classes, &args.format)?;
    
    Ok(())
}

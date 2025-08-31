@echo off
REM Example usage script for COCO to YOLO converter
REM 
REM This script demonstrates how to use the converter with the DAMM dataset

echo COCO to YOLO Converter - Example Usage
echo =====================================
echo.

REM Build the project first
echo [1/3] Building the project...
cargo build --release
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to build the project
    exit /b 1
)
echo Build completed successfully!
echo.

REM Example paths (adjust these to your actual dataset location)
set INPUT_DIR=C:\Users\azusaing\Downloads\detection_datasets
set OUTPUT_DIR=.\yolo_dataset

echo [2/3] Converter configuration:
echo Input directory: %INPUT_DIR%
echo Output directory: %OUTPUT_DIR%
echo.

REM Check if input directory exists
if not exist "%INPUT_DIR%" (
    echo Warning: Input directory "%INPUT_DIR%" does not exist.
    echo Please download and extract the DAMM dataset first.
    echo.
    echo To download the dataset:
    echo 1. Visit: https://huggingface.co/datasets/gauravkaul/DAMM_mouse_detection
    echo 2. Download the detection_datasets.zip file
    echo 3. Extract it to the current directory
    echo.
    echo Then run this script again.
    exit /b 1
)

REM Run the converter
echo [3/3] Running conversion...
echo.
target\release\coco_to_yolo.exe --input "%INPUT_DIR%" --output "%OUTPUT_DIR%" --format damm

if %ERRORLEVEL% eq 0 (
    echo.
    echo =====================================
    echo Conversion completed successfully!
    echo.
    echo Generated files are in: %OUTPUT_DIR%
    echo - Individual .txt files for each image
    echo - classes.txt file with class definitions
    echo.
    echo You can now use these files to train your YOLO model.
) else (
    echo.
    echo Error: Conversion failed
    exit /b 1
)

pause

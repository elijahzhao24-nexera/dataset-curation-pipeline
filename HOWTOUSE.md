# CLI Quick Use

```powershell
# Show commands
python cli.py --help
```

# Basic Commands

### Ingest Images

```powershell
# Ingest local images into DB/S3

python cli.py ingest-folder --input-dir --output-folder

# Example
python cli.py ingest-folder --input-dir ".\data\raw" --output-folder "datasets/raw-batch-01"
```



### Select K Diverse Images
```powershell
# Select k diverse images from a bucket folder
# --input-folder is the folders that the images are in for the s3 container.

python cli.py select-diverse --k --input-folder --output-folder 

# Example
python cli.py select-diverse --k 500 --input-folder "datasets/raw-batch-01" --output-folder ".\output\diverse"
```

### Select k images similar to candidate images
```powershell
# Select k similar images from a bucket folder compared to images in the candidate folder

python cli.py select-similar --k --candidates-folder --input-folder --output-folder

python cli.py select-similar --k 200 --candidates-folder ".\data\candidates" --input-folder "datasets/raw-batch-01" --output-folder ".\output\similar"
```

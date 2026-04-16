# CLI Quick Use

```powershell
# Show commands
python cli.py --help
```

# Basic Commands

### Ingest Images

```powershell
# Ingest local images into DB/S3.
# Images are stored at bucket root using filename keys.

python cli.py ingest-folder --input-dir

# Example
python cli.py ingest-folder --input-dir ".\data\raw"
```



### Select K Diverse Images
```powershell
# Select k diverse images from all vectors in the configured bucket.

python cli.py select-diverse --k --output-folder

# Example
python cli.py select-diverse --k 500 --output-folder ".\output\diverse"
```

### Select k images similar to candidate images
```powershell
# Select k similar images from all vectors in the configured bucket compared to candidate images.

python cli.py select-similar --k --candidates-folder --output-folder

python cli.py select-similar --k 200 --candidates-folder ".\data\candidates" --output-folder ".\output\similar"
```

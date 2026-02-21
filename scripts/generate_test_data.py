"""
generate_test_data.py — Create test files for the compression experiment.

We generate 4 file types at 3 sizes (12 files total):
  - text  (.log) : realistic server logs — very compressible
  - csv   (.csv) : tabular dataset       — very compressible  
  - pdf   (.pdf) : PDF with text pages    — barely compressible (internal compression)
  - image (.bmp) : BMP bitmap image       — very compressible (no internal compression)

Each file is generated deterministically (same SEED = same output every time).
"""

import os
import random
import struct
import sys
from pathlib import Path

# Add scripts/ to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import TEST_DATA_DIR, FILE_SIZES, SEED, logger


# ══════════════════════════════════════════════════════════════════════════
#  GENERATORS — one function per file type
# ══════════════════════════════════════════════════════════════════════════

def generate_text(path, size_mb):
    """
    Generate a fake server log file.
    These are highly compressible because log lines are repetitive.
    """
    logger.info(f"  Generating text log ({size_mb}MB)...")
    random.seed(SEED)

    levels = ["INFO", "DEBUG", "WARN", "ERROR", "TRACE"]
    modules = ["auth", "api", "db", "cache", "scheduler", "worker"]
    templates = [
        "Request processed in {ms}ms",
        "Connection to database established",
        "Cache miss for key user_{id}",
        "Retry attempt {n}/3 for operation",
        "Health check passed",
        "Rate limit hit for {ip}",
        "Memory usage at {pct}%",
        "Config reloaded from disk",
        "Worker finished batch job",
        "Session expired for user_{id}",
    ]

    target = size_mb * 1024 * 1024
    written = 0

    with open(path, "w") as f:
        while written < target:
            # Build a realistic log line
            ts = (f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
                  f"T{random.randint(0,23):02d}:{random.randint(0,59):02d}:"
                  f"{random.randint(0,59):02d}.{random.randint(0,999):03d}Z")
            level = random.choice(levels)
            module = random.choice(modules)
            msg = random.choice(templates).format(
                ms=random.randint(1, 5000),
                id=random.randint(1000, 9999),
                n=random.randint(1, 3),
                ip=f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}",
                pct=random.randint(30, 95),
            )
            line = f"[{ts}] [{level:5s}] [{module:10s}] {msg}\n"
            f.write(line)
            written += len(line.encode())

    actual = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"    -> {actual:.1f}MB")


def generate_csv(path, size_mb):
    """
    Generate a tabular dataset CSV.
    Compresses well because columns have repeated values.
    """
    logger.info(f"  Generating CSV dataset ({size_mb}MB)...")
    random.seed(SEED)

    categories = ["A", "B", "C", "D", "E"]
    statuses = ["active", "inactive", "pending"]
    target = size_mb * 1024 * 1024
    written = 0

    with open(path, "w") as f:
        header = "id,timestamp,value,score,category,status,description\n"
        f.write(header)
        written += len(header.encode())

        row_id = 0
        while written < target:
            row = (
                f"{row_id},"
                f"{random.randint(1700000000, 1710000000)},"
                f"{random.random() * 1000:.2f},"
                f"{random.randint(0, 100)},"
                f"{random.choice(categories)},"
                f"{random.choice(statuses)},"
                f"description for item number {row_id}\n"
            )
            f.write(row)
            written += len(row.encode())
            row_id += 1

    actual = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"    -> {actual:.1f}MB ({row_id} rows)")


def generate_pdf(path, size_mb):
    """
    Generate a PDF file by writing raw PDF syntax.
    PDFs use internal compression (FlateDecode), so compressing them
    externally yields almost no size reduction — that's the point.
    This lets us see how much energy tools waste on already-compressed data.

    We build the PDF manually to avoid needing external libraries.
    """
    logger.info(f"  Generating PDF ({size_mb}MB)...")
    random.seed(SEED)

    import zlib  # for internal PDF compression (FlateDecode)

    target = size_mb * 1024 * 1024

    # We'll generate many PDF "pages" each containing compressed text
    # until we reach the target size.

    pages_content = []  # list of (stream_bytes, page_obj_num)
    total_bytes = 0
    page_num = 0

    while total_bytes < target:
        # Create text content for one page
        lines = []
        lines.append("BT")  # Begin text
        lines.append("/F1 10 Tf")  # Font size 10
        y = 750
        for i in range(50):  # 50 lines per page
            text = (f"Page {page_num+1} Line {i+1}: "
                    f"Lorem ipsum data entry {random.randint(10000,99999)} "
                    f"value={random.random()*1000:.4f} "
                    f"status={'ACTIVE' if random.random()>0.3 else 'CLOSED'} "
                    f"ref={random.randint(100000,999999)}")
            lines.append(f"1 0 0 1 50 {y} Tm")
            lines.append(f"({text}) Tj")
            y -= 14
        lines.append("ET")  # End text

        page_text = "\n".join(lines).encode()
        # Compress with zlib (this is what real PDFs do internally)
        compressed = zlib.compress(page_text)
        pages_content.append(compressed)
        total_bytes += len(compressed)
        page_num += 1

    # Now build the actual PDF file structure
    # (This is simplified but produces valid PDFs)
    objects = []  # list of (obj_number, bytes)
    obj_num = 1

    # Object 1: Catalog
    catalog_num = obj_num
    objects.append((obj_num, b"<< /Type /Catalog /Pages 2 0 R >>"))
    obj_num += 1

    # Object 2: Pages (parent of all page objects)
    pages_parent_num = obj_num
    obj_num += 1  # we'll fill this in after we know page obj numbers

    # Object 3: Font
    font_num = obj_num
    objects.append((obj_num, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))
    obj_num += 1

    # Generate page objects + stream objects
    page_obj_nums = []
    for compressed_stream in pages_content:
        # Stream object
        stream_num = obj_num
        stream_dict = f"<< /Length {len(compressed_stream)} /Filter /FlateDecode >>".encode()
        stream_obj = stream_dict + b"\nstream\n" + compressed_stream + b"\nendstream"
        objects.append((obj_num, stream_obj))
        obj_num += 1

        # Page object
        page_obj_num = obj_num
        page_obj = (f"<< /Type /Page /Parent {pages_parent_num} 0 R "
                    f"/MediaBox [0 0 612 792] "
                    f"/Contents {stream_num} 0 R "
                    f"/Resources << /Font << /F1 {font_num} 0 R >> >> >>").encode()
        objects.append((obj_num, page_obj))
        page_obj_nums.append(page_obj_num)
        obj_num += 1

    # Now fill in the Pages object
    kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
    pages_obj = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_obj_nums)} >>".encode()
    objects.append((pages_parent_num, pages_obj))

    # Sort objects by number
    objects.sort(key=lambda x: x[0])

    # Write PDF
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4\n")

        offsets = {}
        for num, content in objects:
            offsets[num] = f.tell()
            f.write(f"{num} 0 obj\n".encode())
            f.write(content)
            f.write(b"\nendobj\n")

        # Cross-reference table
        xref_offset = f.tell()
        f.write(b"xref\n")
        f.write(f"0 {obj_num}\n".encode())
        f.write(b"0000000000 65535 f \n")
        for i in range(1, obj_num):
            offset = offsets.get(i, 0)
            f.write(f"{offset:010d} 00000 n \n".encode())

        # Trailer
        f.write(b"trailer\n")
        f.write(f"<< /Size {obj_num} /Root {catalog_num} 0 R >>\n".encode())
        f.write(b"startxref\n")
        f.write(f"{xref_offset}\n".encode())
        f.write(b"%%EOF\n")

    actual = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"    -> {actual:.1f}MB ({page_num} pages)")


def generate_image(path, size_mb):
    """
    Generate a BMP (bitmap) image file.
    BMPs are uncompressed raster images — they compress very well externally.
    This represents the use case of compressing raw image data
    (medical imaging, camera RAW, scientific data).

    We write a valid BMP file header + pixel data directly.
    """
    logger.info(f"  Generating BMP image ({size_mb}MB)...")
    random.seed(SEED)

    target = size_mb * 1024 * 1024

    # BMP stores 3 bytes per pixel (BGR) + row padding to 4-byte boundary.
    # For simplicity: make a square-ish image.
    # Each row = width * 3 bytes, padded to multiple of 4.
    bytes_per_pixel = 3
    # Calculate dimensions
    # total pixels ≈ target / 3
    total_pixels = target // bytes_per_pixel
    width = int(total_pixels ** 0.5)
    height = target // (width * bytes_per_pixel)

    # Row padding: each row must be multiple of 4 bytes
    row_bytes = width * bytes_per_pixel
    padding = (4 - (row_bytes % 4)) % 4
    padded_row = row_bytes + padding

    # Actual image data size
    image_data_size = padded_row * height

    # BMP file header (14 bytes) + DIB header (40 bytes) = 54 bytes
    file_size = 54 + image_data_size

    with open(path, "wb") as f:
        # -- BMP File Header (14 bytes) --
        f.write(b"BM")                              # Magic bytes
        f.write(struct.pack("<I", file_size))        # File size
        f.write(struct.pack("<HH", 0, 0))            # Reserved
        f.write(struct.pack("<I", 54))               # Pixel data offset

        # -- DIB Header (BITMAPINFOHEADER, 40 bytes) --
        f.write(struct.pack("<I", 40))               # Header size
        f.write(struct.pack("<i", width))            # Width
        f.write(struct.pack("<i", height))           # Height
        f.write(struct.pack("<HH", 1, 24))           # Planes=1, BPP=24
        f.write(struct.pack("<I", 0))                # No compression
        f.write(struct.pack("<I", image_data_size))  # Image data size
        f.write(struct.pack("<ii", 2835, 2835))      # Resolution (72 DPI)
        f.write(struct.pack("<II", 0, 0))            # Colors

        # -- Pixel data --
        # We write semi-realistic pixel data: smooth gradients with noise.
        # This gives moderate compressibility (not random, not uniform).
        pad_bytes = b"\x00" * padding
        for y in range(height):
            row = bytearray(row_bytes)
            for x in range(width):
                # Gradient + noise = moderately compressible
                base_r = (x * 255 // max(width - 1, 1)) & 0xFF
                base_g = (y * 255 // max(height - 1, 1)) & 0xFF
                base_b = ((x + y) * 127 // max(width + height - 2, 1)) & 0xFF
                noise = random.randint(-20, 20)
                r = max(0, min(255, base_r + noise))
                g = max(0, min(255, base_g + noise))
                b = max(0, min(255, base_b + noise))
                idx = x * 3
                row[idx] = b      # BMP stores BGR
                row[idx + 1] = g
                row[idx + 2] = r
            f.write(row)
            f.write(pad_bytes)

    actual = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"    -> {actual:.1f}MB ({width}x{height}px)")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN — generate all files
# ══════════════════════════════════════════════════════════════════════════

GENERATORS = {
    "text":  generate_text,
    "csv":   generate_csv,
    "pdf":   generate_pdf,
    "image": generate_image,
}


def main():
    logger.info("=== Generating test data ===\n")

    for size_label, size_mb in FILE_SIZES.items():
        logger.info(f"--- {size_label} ({size_mb}MB) ---")
        for file_type, generator in GENERATORS.items():
            from utils import get_test_file_path
            path = get_test_file_path(file_type, size_label)
            generator(path, size_mb)
        print()

    # Print summary
    logger.info("=== Summary ===")
    for size_label in FILE_SIZES:
        for file_type in GENERATORS:
            from utils import get_test_file_path
            path = get_test_file_path(file_type, size_label)
            mb = os.path.getsize(path) / (1024 * 1024)
            logger.info(f"  {file_type}/{size_label}: {mb:.1f}MB")

    logger.info("\nDone! Files in: test_data/")


if __name__ == "__main__":
    main()

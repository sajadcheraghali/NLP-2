import os
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path

INPUT_DIR = "input_pdfs"
TEXT_OUT = "output_texts"

os.makedirs(TEXT_OUT, exist_ok=True)

# تنظیمات OCR
custom_config = r'--oem 1 --psm 6 -c preserve_interword_spaces=1'

def preprocess_for_ocr(img):
    """بهبود تصویر برای OCR فارسی"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    return gray

def ocr_image(img):
    return pytesseract.image_to_string(img, lang="fas+eng", config=custom_config)

def extract_tables(img):
    """
    تشخیص جدول‌ها با OpenCV و تقسیم به سلول
    خروجی: لیست جدول‌ها -> هر جدول لیست ردیف‌ها -> هر ردیف لیست سلول‌ها
    """
    tables = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # تشخیص خطوط افقی و عمودی
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    horiz_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
    vert_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, vert_kernel, iterations=2)
    
    # ترکیب خطوط
    table_mask = cv2.add(horiz_lines, vert_lines)
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 50 or h < 50:
            continue  # جدول خیلی کوچک نادیده گرفته شود

        table_img = img[y:y+h, x:x+w]
        table_rows = split_table_into_cells(table_img)
        if table_rows:
            tables.append(table_rows)

    return tables

def split_table_into_cells(table_img):
    """تقسیم جدول به سلول‌ها"""
    gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # خطوط افقی و عمودی
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    horiz_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, horiz_kernel, iterations=2)
    vert_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, vert_kernel, iterations=2)
    mask = cv2.add(horiz_lines, vert_lines)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = []

    # مرتب‌سازی سلول‌ها از بالا به پایین، چپ به راست
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1], b[0]))  # sort by y, then x

    row = []
    last_y = -1
    for x, y, w, h in bounding_boxes:
        if last_y != -1 and abs(y - last_y) > 10:
            if row:
                cells.append(row)
            row = []
        cell_img = table_img[y:y+h, x:x+w]
        cell_text = ocr_image(preprocess_for_ocr(cell_img)).strip()
        row.append(cell_text)
        last_y = y
    if row:
        cells.append(row)

    return cells

# -------------------------
# پردازش PDFها
# -------------------------
for pdf_name in os.listdir(INPUT_DIR):
    if not pdf_name.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(INPUT_DIR, pdf_name)
    print(f"📥 Processing: {pdf_name}")

    all_text = ""

    try:
        images = convert_from_path(pdf_path, dpi=400)
    except Exception as e:
        print("❌ خطا در تبدیل PDF به تصویر:", e)
        continue

    for i, img in enumerate(images):
        img_np = np.array(img)
        pre_img = preprocess_for_ocr(img_np)

        # متن کل صفحه
        page_text = ocr_image(pre_img)
        all_text += f"\n\n===== صفحه {i+1} =====\n{page_text}"

        # جدول‌ها
        tables = extract_tables(img_np)
        for t_idx, table in enumerate(tables):
            all_text += f"\n--- جدول {t_idx+1} (صفحه {i+1}) ---\n"
            for row in table:
                all_text += " | ".join(row) + "\n"

    # ذخیره خروجی
    out_txt = os.path.join(TEXT_OUT, pdf_name.replace(".pdf", ".txt"))
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"✅ خروجی TXT ساخته شد → {out_txt}")

print("\n🎉 همه PDFها پردازش شدند.")

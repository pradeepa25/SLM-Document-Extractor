import os
import subprocess
import json
import re
import fitz  # PyMuPDF

# UPDATE THESE PATHS TO YOUR LOCAL GGUF MODEL AND LLAMA.CPP BINARY
LLAMA_CLI = os.path.join("llama-cpp-bin", "llama-mtmd-cli.exe")
MODEL_PATH = os.path.join("output", "qwen2-vl-2b-instruct.Q4_K_M.gguf")
MMPROJ_PATH = os.path.join("output", "qwen2-vl-2b-instruct.F16-mmproj.gguf")

def extract_json_from_output(output_text):
    match = re.search(r'```json\n(.*?)```', output_text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r'\{.*\}', output_text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def process_page_with_llama(image_path, page_num):
    print(f"[{page_num}] Running Map Extraction on page {page_num}...")
    
    system_prompt = "You are a helpful assistant."
    user_prompt = """Extract all Purchase Order data from this page.
Return ONLY valid JSON in the exact following format:
{
  "Headers": {
    "PO Number": "",
    "Date": "",
    "Bill To": "",
    "Ship To": "",
    "Terms": "",
    "Ship Via": "",
    "Shipping Instructions": ""
  },
  "LineItems": [
    {
      "Part Number": "",
      "Description": "",
      "Quantity": "",
      "Price": ""
    }
  ]
}
If a header field is not found on this specific page, leave it empty. Extract ALL tabular line items found on this page. Do not include any other text. ONLY output the JSON."""

    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "--mmproj", MMPROJ_PATH,
        "--image", image_path,
        "-p", prompt,
        "-n", "1024",
        "--temp", "0.1",
        "-c", "4096",
        "-ngl", "0"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        json_data = extract_json_from_output(result.stdout)
        if json_data:
            return json.loads(json_data)
        else:
            print(f"[{page_num}] Warning: VLM returned invalid JSON.")
    except Exception as e:
        print(f"[{page_num}] Error during inference: {e}")
        
    return {"Headers": {}, "LineItems": []}

def process_document(pdf_path):
    print(f"Processing document: {os.path.basename(pdf_path)} using Map-Reduce SLM strategy...")
    doc = fitz.open(pdf_path)
    
    merged_headers = {}
    merged_line_items = []
    
    for i in range(len(doc)):
        page_num = i + 1
        page = doc[i]
        pix = page.get_pixmap(dpi=150)
        temp_img = f"temp_page_{page_num}.jpg"
        pix.save(temp_img)
        
        # Map: Extract page
        page_data = process_page_with_llama(temp_img, page_num)
        
        # Reduce: Merge Headers
        for k, v in page_data.get("Headers", {}).items():
            if v and str(v).strip():
                if k not in merged_headers or not merged_headers[k]:
                    merged_headers[k] = v
                    
        # Reduce: Merge Line Items
        for item in page_data.get("LineItems", []):
            if any(item.values()):
                merged_line_items.append(item)
                
        if os.path.exists(temp_img):
            os.remove(temp_img)
            
    # Final Output Formatting
    out = {
        "file_name": os.path.basename(pdf_path),
        "num_pages": len(doc),
        "headers": merged_headers,
        "line_items": merged_line_items,
        "status": "success"
    }
    
    out_path = pdf_path.replace(".pdf", "_extracted.json").replace(".PDF", "_extracted.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        
    print(f"\nExtraction Complete! Saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to PDF file")
    args = parser.parse_args()
    
    if os.path.exists(args.pdf_path):
        process_document(args.pdf_path)
    else:
        print("File not found.")

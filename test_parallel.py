import base64
import os
import io
import time
from typing import Optional, Dict, Any, Union, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import google.generativeai as genai
from PIL import Image
import fitz  # PyMuPDF for PDF processing
from dotenv import load_dotenv
import threading

load_dotenv()

# Thread-local storage for Gemini models
thread_local = threading.local()

def get_gemini_model(api_key: str) -> genai.GenerativeModel:
    """
    Get or create a Gemini model instance for the current thread.
    
    Args:
        api_key: Google API key for Gemini
        
    Returns:
        Configured Gemini model instance
    """
    if not hasattr(thread_local, 'model'):
        genai.configure(api_key=api_key)
        thread_local.model = genai.GenerativeModel('gemini-1.5-flash')
    return thread_local.model

def process_single_page(args: Tuple[int, bytes, str, str, bool]) -> Dict[str, Any]:
    """
    Process a single page with Gemini API.
    
    Args:
        args: Tuple containing (page_num, image_data, prompt, api_key, extract_charts)
        
    Returns:
        Dictionary with page results
    """
    page_num, image_data, prompt, api_key, extract_charts = args
    
    try:
        # Create PIL Image from bytes
        image = Image.open(io.BytesIO(image_data))
        
        # Get model for this thread
        model = get_gemini_model(api_key)
        
        # Generate content with Gemini
        response = model.generate_content([prompt, image])  # type: ignore
        page_text = response.text
        
        return {
            'page': page_num,
            'text': page_text,
            'success': True
        }
        
    except Exception as e:
        return {
            'page': page_num,
            'text': f"Error extracting text: {e}",
            'success': False
        }

def extract_page_as_image(pdf_path: str, page_num: int, resolution: float = 3.0) -> Tuple[int, bytes]:
    """
    Extract a single page as image data.
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (0-indexed)
        resolution: Image resolution multiplier
        
    Returns:
        Tuple of (page_num, image_bytes)
    """
    pdf_document = fitz.open(pdf_path)
    try:
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(resolution, resolution))
        img_data = pix.tobytes("png")
        return page_num + 1, img_data  # Return 1-indexed page number
    finally:
        pdf_document.close()

def extract_text_from_pdf_parallel(
    pdf_path: str, 
    api_key: str, 
    start_page: int = 10, 
    end_page: int = 12, 
    extract_charts: bool = True,
    max_workers: Optional[int] = None,
    all_pages: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Extract text and charts from PDF using Gemini Vision API with parallel processing.
    
    Args:
        pdf_path: Path to the PDF file
        api_key: Google API key for Gemini
        start_page: First page to extract (1-indexed, default: 10)
        end_page: Last page to extract (1-indexed, default: 12)
        extract_charts: Whether to focus on chart extraction (default: True)
        max_workers: Maximum number of parallel workers (default: min(8, cpu_count))
        all_pages: Whether to extract all pages (default: False)
    Returns:
        Dictionary containing extracted text and metadata, or None if error
    """
    # Set default max_workers if not provided
    if max_workers is None:
        max_workers = min(8, cpu_count())
    
    # Open PDF to get total pages
    pdf_document = fitz.open(pdf_path)
    total_pages_in_pdf = len(pdf_document)
    pdf_document.close()
    
    # Adjust for 0-indexed pages in PyMuPDF
    start_page_idx = start_page - 1
    end_page_idx = end_page - 1
    
    if all_pages:
        start_page_idx = 0
        end_page_idx = total_pages_in_pdf - 1
        # Update start_page and end_page for display purposes
        start_page = 1
        end_page = total_pages_in_pdf
    
    # Validate page range
    if start_page_idx < 0 or end_page_idx >= total_pages_in_pdf:
        print(f"Error: Page range {start_page}-{end_page} is invalid for PDF with {total_pages_in_pdf} pages")
        return None
    
    if start_page_idx > end_page_idx:
        print(f"Error: Start page ({start_page}) cannot be greater than end page ({end_page})")
        return None
    
    print(f"Starting PARALLEL OCR extraction...")
    print(f"Total pages in PDF: {total_pages_in_pdf}")
    print(f"Extracting pages: {start_page} to {end_page}")
    print(f"Chart extraction: {'Enabled' if extract_charts else 'Disabled'}")
    print(f"Max parallel workers: {max_workers}")
    print("-" * 50)
    
    start_time = time.time()
    
    # Comprehensive prompt that extracts both text and charts
    prompt = """
    Please analyze this image and extract ALL information in the following structured format:

    === TEXT CONTENT ===
    [Extract all text content, titles, labels, captions, and any written information]

    === CHARTS AND GRAPHS ===
    [If charts or graphs are present, analyze each one with:]
    - Chart Type: [bar chart, line chart, pie chart, scatter plot, etc.]
    - Chart Title: [exact title]
    - X-Axis Label: [label]
    - Y-Axis Label: [label]
    - Data Points: [list all data points with exact values and labels]
    - Legend: [legend items if present]
    - Trends/Patterns: [describe any visible trends or patterns]
    - Data Source: [if mentioned]
    - Time Period: [if mentioned]

    === TABLES ===
    [If tables are present, preserve the structure and extract all data]

    === IMAGES AND VISUAL ELEMENTS ===
    [Describe any images, icons, or visual elements]

    === ADDITIONAL INFORMATION ===
    - Footnotes: [any footnotes or source information]
    - Color coding: [if colors represent different categories]

    Format the output clearly with sections for each type of content.
    For charts, provide the data in a structured format that could be used to recreate the chart.
    Return only the extracted information without additional commentary.
    """
    
    # Step 1: Extract all pages as images in parallel
    print("Step 1: Extracting pages as images...")
    image_extraction_start = time.time()
    
    with Pool(processes=min(max_workers, cpu_count())) as pool:
        # Create arguments for image extraction
        extraction_args = [
            (pdf_path, page_num, 4.0)  # resolution = 4.0 for better quality
            for page_num in range(start_page_idx, end_page_idx + 1)
        ]
        
        # Extract images in parallel
        image_results = pool.starmap(extract_page_as_image, extraction_args)
    
    image_extraction_time = time.time() - image_extraction_start
    print(f"✓ Image extraction completed in {image_extraction_time:.2f}s")
    
    # Step 2: Process images with Gemini API in parallel
    print("Step 2: Processing images with Gemini API...")
    api_processing_start = time.time()
    
    # Prepare arguments for API processing
    api_args = [
        (page_num, img_data, prompt, api_key, extract_charts)
        for page_num, img_data in image_results
    ]
    
    extracted_text = []
    successful_pages = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_page = {
            executor.submit(process_single_page, args): args[0] 
            for args in api_args
        }
        
        # Process completed tasks
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                result = future.result()
                extracted_text.append({
                    'page': result['page'],
                    'text': result['text']
                })
                if result['success']:
                    successful_pages += 1
                    print(f"✓ Page {result['page']} processed successfully")
                else:
                    print(f"✗ Page {result['page']} failed: {result['text']}")
            except Exception as e:
                print(f"✗ Exception processing page {page_num}: {e}")
                extracted_text.append({
                    'page': page_num,
                    'text': f"Error extracting text: {e}"
                })
    
    api_processing_time = time.time() - api_processing_start
    total_time = time.time() - start_time
    
    # Sort results by page number
    extracted_text.sort(key=lambda x: x['page'])
    
    print("-" * 50)
    print(f"PARALLEL OCR extraction completed!")
    print(f"Image extraction time: {image_extraction_time:.2f}s")
    print(f"API processing time: {api_processing_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Successfully processed: {successful_pages}/{len(extracted_text)} pages")
    
    return {
        'total_pages_extracted': len(extracted_text),
        'pages': extracted_text,
        'start_page': start_page,
        'end_page': end_page,
        'total_pages_in_pdf': total_pages_in_pdf,
        'chart_extraction_enabled': extract_charts,
        'performance': {
            'total_time': total_time,
            'image_extraction_time': image_extraction_time,
            'api_processing_time': api_processing_time,
            'max_workers': max_workers
        }
    }

def save_extracted_text_to_file(extracted_data: Dict[str, Any], output_path: str) -> None:
    """
    Save extracted text and chart data to a file.
    
    Args:
        extracted_data: Dictionary containing extracted text and charts
        output_path: Path where to save the output file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"PARALLEL TEXT + CHART EXTRACTION RESULTS\n")
        f.write(f"Pages extracted: {extracted_data['start_page']} to {extracted_data['end_page']}\n")
        f.write(f"Total pages in PDF: {extracted_data['total_pages_in_pdf']}\n")
        f.write(f"Chart extraction: {'Enabled' if extracted_data.get('chart_extraction_enabled', False) else 'Disabled'}\n")
        
        # Add performance metrics
        if 'performance' in extracted_data:
            perf = extracted_data['performance']
            f.write(f"Performance:\n")
            f.write(f"  Total time: {perf['total_time']:.2f}s\n")
            f.write(f"  Image extraction: {perf['image_extraction_time']:.2f}s\n")
            f.write(f"  API processing: {perf['api_processing_time']:.2f}s\n")
            f.write(f"  Max workers: {perf['max_workers']}\n")
        
        f.write("=" * 60 + "\n\n")
        
        for page_data in extracted_data['pages']:
            f.write(f"PAGE {page_data['page']}\n")
            f.write("=" * 50 + "\n")
            f.write(page_data['text'])
            f.write("\n\n" + "-" * 60 + "\n\n")

# Usage example
if __name__ == "__main__":
    # Set your API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable")
        exit(1)
    
    # Path to your PDF
    pdf_path = "C:/MoovMedia/ocr_exploration/input/The-Future-100-2025_part_1.pdf"
    
    # Extract base filename without extension for output naming
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    print("=" * 60)
    print("PARALLEL TEXT + CHART EXTRACTION")
    print("=" * 60)
    
    # Run comprehensive extraction with both text and charts
    print("\nRunning parallel extraction with text and charts...")
    result = extract_text_from_pdf_parallel(
        pdf_path, 
        api_key, 
        all_pages=True,  # Process all pages
        extract_charts=True,  # Enable chart extraction
        max_workers=4  # Adjust based on your system and API limits
    )
    
    if result:
        output_path = f"C:/MoovMedia/ocr_exploration/output/{pdf_filename}_complete_extraction.txt"
        save_extracted_text_to_file(result, output_path)
        print(f"Complete extraction results saved to {output_path}")
    else:
        print("Extraction failed.")
    
    print("\n" + "=" * 60)
    print("PARALLEL EXTRACTION COMPLETED!")
    print("=" * 60) 
import base64
import os
import io
from typing import Optional, Dict, Any, List, Tuple
import google.generativeai as genai
from PIL import Image
import fitz  # PyMuPDF for PDF processing
from pathlib import Path
import concurrent.futures
import threading
from tqdm import tqdm
import time
from dataclasses import dataclass
from queue import Queue
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class ProcessingConfig:
    """Configuration for parallel processing."""
    max_workers: int = 4  # Number of parallel workers
    batch_size: int = 3   # Number of pages to process in each batch
    retry_attempts: int = 3  # Number of retry attempts for failed pages

def find_all_pdf_files(input_dir: str) -> List[str]:
    """
    Find all PDF files in the input directory and its subdirectories.
    
    Args:
        input_dir: Path to the input directory
        
    Returns:
        List of paths to all PDF files found
    """
    pdf_files = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return pdf_files
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                pdf_files.append(pdf_path)
    
    return pdf_files

def encode_pdf(pdf_path: str) -> Optional[str]:
    """
    Encode the PDF to base64.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Base64 encoded string of the PDF or None if error
    """
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_single_page(args: Tuple[str, int, str, genai.GenerativeModel, ProcessingConfig]) -> Dict[str, Any]:
    """
    Process a single page with retry logic and error handling.
    
    Args:
        args: Tuple containing (pdf_path, page_num, api_key, model, config)
        
    Returns:
        Dictionary with page processing results
    """
    pdf_path, page_num, api_key, model, config = args
    
    # Configure Gemini for this thread
    genai.configure(api_key=api_key)
    
    try:
        # Open PDF and get specific page
        pdf_document = fitz.open(pdf_path)
        page = pdf_document[page_num]
        
        # Convert page to image with higher resolution
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_data = pix.tobytes("png")
        
        # Create PIL Image
        image = Image.open(io.BytesIO(img_data))
        
        # Prepare prompt for Gemini
        prompt = """
        Please extract all text from this image. 
        Maintain the original formatting and structure.
        If there are tables, preserve the table structure.
        Return only the extracted text without any additional commentary.
        """
        
        # Retry logic for API calls
        for attempt in range(config.retry_attempts):
            try:
                response = model.generate_content([prompt, image])
                page_text = response.text
                
                pdf_document.close()
                return {
                    'page': page_num + 1,  # Convert to 1-indexed
                    'text': page_text,
                    'success': True,
                    'attempts': attempt + 1
                }
                
            except Exception as e:
                if attempt == config.retry_attempts - 1:
                    pdf_document.close()
                    return {
                        'page': page_num + 1,
                        'text': f"Error extracting text after {config.retry_attempts} attempts: {e}",
                        'success': False,
                        'attempts': config.retry_attempts
                    }
                time.sleep(1)  # Brief pause before retry
                
    except Exception as e:
        return {
            'page': page_num + 1,
            'text': f"Error processing page: {e}",
            'success': False,
            'attempts': 0
        }

def process_pdf_pages_parallel(pdf_path: str, api_key: str, start_page: int = 1, 
                             end_page: Optional[int] = None, config: Optional[ProcessingConfig] = None) -> Dict[str, Any]:
    """
    Extract text from PDF using parallel processing for multiple pages.
    
    Args:
        pdf_path: Path to the PDF file
        api_key: Google API key for Gemini
        start_page: First page to extract (1-indexed, default: 1)
        end_page: Last page to extract (1-indexed, default: None for all pages)
        config: Processing configuration
        
    Returns:
        Dictionary containing extracted text and metadata
    """
    if config is None:
        config = ProcessingConfig()
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Open PDF to get total pages
    pdf_document = fitz.open(pdf_path)
    total_pages_in_pdf = len(pdf_document)
    pdf_document.close()
    
    # Validate page range
    if end_page is None:
        end_page = total_pages_in_pdf
    
    # Adjust for 0-indexed pages in PyMuPDF
    start_page_idx = start_page - 1
    end_page_idx = end_page - 1
    
    # Validate page range
    if start_page_idx < 0 or end_page_idx >= total_pages_in_pdf:
        print(f"Error: Page range {start_page}-{end_page} is invalid for PDF with {total_pages_in_pdf} pages")
        return None
    
    if start_page_idx > end_page_idx:
        print(f"Error: Start page ({start_page}) cannot be greater than end page ({end_page})")
        return None
    
    print(f"Starting parallel OCR extraction for {os.path.basename(pdf_path)}...")
    print(f"Total pages in PDF: {total_pages_in_pdf}")
    print(f"Extracting pages: {start_page} to {end_page}")
    print(f"Using {config.max_workers} parallel workers")
    print("-" * 50)
    
    # Create list of pages to process
    pages_to_process = list(range(start_page_idx, end_page_idx + 1))
    
    # Process pages in parallel
    extracted_text = []
    successful_pages = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # Prepare arguments for each page
        futures_args = [
            (pdf_path, page_num, api_key, model, config) 
            for page_num in pages_to_process
        ]
        
        # Submit all tasks and track progress
        with tqdm(total=len(pages_to_process), desc=f"Processing {os.path.basename(pdf_path)}") as pbar:
            # Submit all tasks
            future_to_page = {
                executor.submit(process_single_page, args): args[1] 
                for args in futures_args
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_page):
                result = future.result()
                extracted_text.append(result)
                
                if result['success']:
                    successful_pages += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'Success': f"{successful_pages}/{len(pages_to_process)}",
                    'Page': result['page']
                })
    
    # Sort results by page number
    extracted_text.sort(key=lambda x: x['page'])
    
    print("-" * 50)
    print(f"Parallel OCR extraction completed for {os.path.basename(pdf_path)}!")
    print(f"Successfully processed: {successful_pages}/{len(extracted_text)} pages")
    
    return {
        'pdf_path': pdf_path,
        'pdf_name': os.path.basename(pdf_path),
        'total_pages_extracted': len(extracted_text),
        'pages': extracted_text,
        'start_page': start_page,
        'end_page': end_page,
        'total_pages_in_pdf': total_pages_in_pdf,
        'successful_pages': successful_pages
    }

def process_multiple_pdfs_parallel(pdf_files: List[str], api_key: str, start_page: int = 1, 
                                 end_page: Optional[int] = None, config: Optional[ProcessingConfig] = None) -> List[Dict[str, Any]]:
    """
    Process multiple PDF files in parallel.
    
    Args:
        pdf_files: List of PDF file paths
        api_key: Google API key for Gemini
        start_page: First page to extract (1-indexed, default: 1)
        end_page: Last page to extract (1-indexed, default: None for all pages)
        config: Processing configuration
        
    Returns:
        List of processing results for each PDF
    """
    if config is None:
        config = ProcessingConfig()
    
    results = []
    
    print(f"Processing {len(pdf_files)} PDF files in parallel...")
    print(f"Using {config.max_workers} parallel workers")
    print("=" * 60)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(config.max_workers, len(pdf_files))) as executor:
        # Submit all PDF processing tasks
        future_to_pdf = {
            executor.submit(process_pdf_pages_parallel, pdf_path, api_key, start_page, end_page, config): pdf_path
            for pdf_path in pdf_files
        }
        
        # Collect results as they complete
        with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
            for future in concurrent.futures.as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                try:
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    pbar.set_postfix({
                        'PDF': os.path.basename(pdf_path),
                        'Success': f"{result['successful_pages']}/{result['total_pages_extracted']}" if result else "Failed"
                    })
                except Exception as e:
                    print(f"Error processing {pdf_path}: {e}")
                    # Create a failed result entry instead of None
                    failed_result = {
                        'pdf_path': pdf_path,
                        'pdf_name': os.path.basename(pdf_path),
                        'start_page': start_page,
                        'end_page': end_page,
                        'total_pages_in_pdf': 0,
                        'successful_pages': 0,
                        'total_pages_extracted': 0,
                        'pages': [],
                        'error': str(e)
                    }
                    results.append(failed_result)
                    pbar.update(1)
    
    return results

def save_extracted_text_to_file(extracted_data: Dict[str, Any], output_path: str) -> None:
    """
    Save extracted text to a file.
    
    Args:
        extracted_data: Dictionary containing extracted text
        output_path: Path where to save the output file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"OCR Extraction Results\n")
        f.write(f"PDF: {extracted_data['pdf_name']}\n")
        f.write(f"Pages extracted: {extracted_data['start_page']} to {extracted_data['end_page']}\n")
        f.write(f"Total pages in PDF: {extracted_data['total_pages_in_pdf']}\n")
        f.write(f"Successfully processed: {extracted_data['successful_pages']}/{extracted_data['total_pages_extracted']} pages\n")
        f.write("=" * 50 + "\n\n")
        
        for page_data in extracted_data['pages']:
            f.write(f"Page {page_data['page']}")
            if not page_data['success']:
                f.write(" (FAILED)")
            f.write("\n")
            f.write("-" * 30 + "\n")
            f.write(page_data['text'])
            f.write("\n\n")

def process_all_pdfs_in_directory_parallel(input_dir: str, api_key: str, start_page: int = 1, 
                                         end_page: Optional[int] = None, config: Optional[ProcessingConfig] = None) -> None:
    """
    Process all PDF files in the input directory using parallel processing.
    
    Args:
        input_dir: Path to the input directory
        api_key: Google API key for Gemini
        start_page: First page to extract (1-indexed, default: 1)
        end_page: Last page to extract (1-indexed, default: None for all pages)
        config: Processing configuration
    """
    if config is None:
        config = ProcessingConfig()
    
    # Find all PDF files
    pdf_files = find_all_pdf_files(input_dir)
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir} or its subdirectories")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s):")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"{i}. {pdf_file}")
    print("=" * 60)
    
    # Process PDFs in parallel
    results = process_multiple_pdfs_parallel(pdf_files, api_key, start_page, end_page, config)
    
    # Save results
    successful_saves = 0
    for i, result in enumerate(results):
            if result and 'error' not in result:
                try:
                    # Create output filename based on PDF name
                    pdf_name = os.path.splitext(os.path.basename(result['pdf_path']))[0]
                    output_filename = f"{pdf_name}_extracted_text.txt"
                    output_path = os.path.join("output", output_filename)
                    
                    # Ensure output directory exists
                    os.makedirs("output", exist_ok=True)
                    
                    # Save to file
                    save_extracted_text_to_file(result, output_path)
                    print(f"✓ Results saved to {output_path}")
                    successful_saves += 1
                except Exception as e:
                    print(f"✗ Error saving results for {result['pdf_name']}: {e}")
            elif result and 'error' in result:
                print(f"✗ PDF {i+1} failed to process: {result['error']}")
            else:
                print(f"✗ No results to save for PDF {i+1}")
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {len([r for r in results if r])}/{len(results)} PDFs")
    print(f"Successfully saved: {successful_saves}/{len(results)} files")

# Usage example with parallel processing
if __name__ == "__main__":
    # Set your API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable")
        exit(1)
    
    # Path to your input directory
    input_dir = "C:/MoovMedia/ocr_exploration/input"
    
    # Configure parallel processing
    config = ProcessingConfig(
        max_workers=4,      # Number of parallel workers
        batch_size=3,       # Pages per batch
        retry_attempts=3    # Retry attempts for failed pages
    )
    
    # Process all PDF files with parallel processing
    process_all_pdfs_in_directory_parallel(
        input_dir, 
        api_key, 
        start_page=1, 
        end_page=None,
        config=config
    )
    
    print("\nAll PDF files have been processed with parallel processing!")
# --- Imports ---
import os
import glob
import pandas as pd
import google.generativeai as genai
from lxml import html
import time
import re

# --- Configuration ---

# !!! --- API KEY CONFIGURATION --- !!!
# IMPORTANT: The hardcoded API key has been removed for security.
# This script now requires you to set the GEMINI_API_KEY environment variable.
api_key = ""

if not api_key:
    print("Error: Gemini API Key not found.")
    print("Please set the GEMINI_API_KEY environment variable.")
    exit()

try:
    genai.configure(api_key=api_key)
    print("Gemini API Key configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API with the provided key: {e}")
    exit()
# !!! --- END OF API KEY CONFIGURATION --- !!!

# --- Paths and Constants ---

# Adjust the folder path to where your HTML files are located
HTML_FOLDER_PATH = '/Users/ yanha/Downloads/test_degree_page/'

# Output file name for the results
OUTPUT_CSV_PATH = '/Users/yanha/Downloads/extracted_majors_lxml_via_gemini_xpath.csv'

# --- Gemini API Setup ---

# CORRECTED: Using a valid and current model name
MODEL_NAME = 'gemini-2.0-flash'
try:
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"Using Gemini model: '{MODEL_NAME}'")
except Exception as e:
    print(f"Error initializing Gemini model '{MODEL_NAME}': {e}")
    exit()

# This prompt asks Gemini for XPath expressions to find program details.
# Experiment with this prompt for best results.
XPATH_GENERATION_PROMPT = """
Analyze the following HTML content from a university degree page.
Identify the common structure used to list individual degree programs or majors.
Provide the specific XPath expressions that would select the following elements FOR A TYPICAL PROGRAM LISTING on this page:

1.  Major Name Element XPath: The XPath to select the primary element (e.g., <h1>, <h2>, <a>, <span>) containing the text name of the major/degree. This XPath should ideally select the element containing the most complete name.
2.  Major URL Element XPath: The XPath to select the <a> anchor element whose 'href' attribute contains the specific URL for that major/degree. If no direct link element is consistently found, provide 'XPath Not Found'.
3.  Major Description Element XPath: The XPath to select the element(s) (e.g., <p>, <div>) containing the main descriptive text for the major. If no distinct description element is consistently found, provide 'XPath Not Found'.

Please ensure that XPaths can capture multiple majors if present. May use contains() to filter.
Provide *only* the XPath expressions in the following exact format, without explanations:
XPATH_NAME: [XPath expression for Major Name Element]
XPATH_URL: [XPath expression for Major URL Element or 'XPath Not Found']
XPATH_DESC: [XPath expression for Major Description Element or 'XPath Not Found']

HTML Content to Analyze:
```html
{html_content}
```
XPath Expressions:
"""

# --- Helper Functions ---

def get_xpaths_from_gemini(html_content, retries=3, delay=5):
    """
    Sends HTML to Gemini API to get XPath expressions.

    Returns:
        str or None: The text response from the API, or None on failure.
    """
    prompt = XPATH_GENERATION_PROMPT.format(html_content=html_content)
    for attempt in range(retries):
        try:
            print(f"  - Calling Gemini API for XPaths (Attempt {attempt + 1}/{retries})...")
            # Safety settings can be adjusted if the API blocks responses too aggressively
            safety_settings = {
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            }
            response = model.generate_content(prompt, safety_settings=safety_settings)

            if response and hasattr(response, 'text'):
                print("  - API call for XPaths successful.")
                return response.text
            else:
                feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'
                print(f"  - Warning: Received empty/blocked response (Attempt {attempt + 1}/{retries}). Feedback: {feedback}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    return None
        except Exception as e:
            print(f"  - Error calling Gemini API (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return None

def parse_xpath_response(api_response_text):
    """
    Parses the Gemini response to extract XPath strings using regular expressions.

    Returns:
        dict: Dictionary with 'name_xpath', 'url_xpath', 'desc_xpath', or None if parsing fails.
    """
    if not api_response_text:
        print("  - Parsing skipped: Received empty API response.")
        return None

    # Use regex for more robust parsing, ignoring whitespace and minor formatting issues
    name_match = re.search(r"XPATH_NAME:\s*(.+)", api_response_text, re.MULTILINE)
    url_match = re.search(r"XPATH_URL:\s*(.+)", api_response_text, re.MULTILINE)
    desc_match = re.search(r"XPATH_DESC:\s*(.+)", api_response_text, re.MULTILINE)

    name_xpath = name_match.group(1).strip() if name_match else None
    url_xpath = url_match.group(1).strip() if url_match else None
    desc_xpath = desc_match.group(1).strip() if desc_match else None
    
    # The name XPath is required to proceed with extraction
    if not name_xpath or name_xpath == 'XPath Not Found':
        print(f"  - Parsing Failed: A valid XPATH_NAME was not found in the response.")
        return None

    # Handle cases where other XPaths are not found
    if url_xpath == 'XPath Not Found': url_xpath = None
    if desc_xpath == 'XPath Not Found': desc_xpath = None
    
    xpaths = {
        'name_xpath': name_xpath,
        'url_xpath': url_xpath,
        'desc_xpath': desc_xpath
    }
    
    print(f"  - Parsed XPaths: Name='{name_xpath}', URL='{url_xpath}', Desc='{desc_xpath}'")
    return xpaths

def extract_data_with_lxml(tree, xpaths):
    """
    Uses lxml to execute the generated XPaths and extract data.
    
    !! LOGIC WARNING !!
    This function assumes the lists of names, URLs, and descriptions returned by the
    XPaths correspond by index. This is a fragile assumption and can lead to mismatched
    data. A more robust solution would first find a parent/container element for each
-   major and then extract details relative to that container.

    Returns:
        list: A list of dictionaries, each representing an extracted major.
    """
    extracted_data = []
    if not xpaths or not xpaths.get('name_xpath'):
        print("  - LXML Extraction skipped: Invalid or missing Name XPath.")
        return extracted_data

    try:
        # --- Extract all potential elements from the page ---
        name_elements = tree.xpath(xpaths['name_xpath'])
        url_elements = tree.xpath(xpaths['url_xpath']) if xpaths.get('url_xpath') else []
        desc_elements = tree.xpath(xpaths['desc_xpath']) if xpaths.get('desc_xpath') else []

        if not name_elements:
            print(f"  - LXML Warning: Name XPath '{xpaths['name_xpath']}' found 0 elements.")
            return []

        print(f"  - LXML: Found {len(name_elements)} names, {len(url_elements)} URLs, {len(desc_elements)} descriptions.")

        # --- Correlate data by index ---
        for i, name_element in enumerate(name_elements):
            # Clean up whitespace from extracted text
            major_name = " ".join(name_element.text_content().split())
            
            # Get corresponding URL if available
            major_url = "Not Found"
            if i < len(url_elements) and url_elements[i] is not None:
                major_url = str(url_elements[i])
            
            # Get corresponding description if available
            major_description = "Not Found"
            if i < len(desc_elements) and desc_elements[i] is not None:
                major_description = " ".join(desc_elements[i].text_content().split())

            if major_name:
                extracted_data.append({
                    'major_name': major_name,
                    'major_url': major_url,
                    'major_description': major_description
                })
            else:
                 print(f"  - LXML Skipping entry {i+1}: Invalid major name found.")

    except Exception as e:
        print(f"  - LXML Error during XPath execution or data correlation: {e}")

    return extracted_data


# --- Main Processing Logic ---

def main():
    """
    Main function to find HTML files, get XPaths via Gemini, extract data, and save to CSV.
    """
    file_pattern = os.path.join(HTML_FOLDER_PATH, 'degree_page_*.html')
    html_files = glob.glob(file_pattern)

    if not html_files:
        print(f"Error: No HTML files found matching '{file_pattern}'.")
        print(f"Please check the HTML_FOLDER_PATH: {HTML_FOLDER_PATH}")
        return

    print(f"Found {len(html_files)} HTML files to process.")
    all_extracted_data = []

    for file_path in html_files:
        file_name = os.path.basename(file_path)
        print(f"\nProcessing file: {file_name}")

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()

            if not html_content.strip():
                print("  - Warning: HTML file is empty. Skipping.")
                continue

            # Limit content size to avoid exceeding API context limits
            max_chars = 150000
            html_content_for_api = html_content[:max_chars] if len(html_content) > max_chars else html_content

            # 1. Get XPaths from Gemini
            xpath_response = get_xpaths_from_gemini(html_content_for_api)
            xpaths = parse_xpath_response(xpath_response)

            if xpaths:
                try:
                    # 2. Parse the full HTML with LXML
                    tree = html.fromstring(html_content)
                    
                    # 3. Extract data using the generated XPaths
                    extracted_records = extract_data_with_lxml(tree, xpaths)

                    if extracted_records:
                        # Add source file and the XPaths used for extraction to each record
                        for record in extracted_records:
                            record['source_file'] = file_name
                            record['name_xpath'] = xpaths.get('name_xpath', 'N/A')
                            record['url_xpath'] = xpaths.get('url_xpath', 'N/A')
                            record['desc_xpath'] = xpaths.get('desc_xpath', 'N/A')
                            all_extracted_data.append(record)
                        
                        print(f"  - Success: Extracted {len(extracted_records)} program(s) from this file.")
                    else:
                        print("  - Info: LXML extraction yielded no data with the generated XPaths.")

                except Exception as e_lxml:
                    print(f"  - LXML Error: Could not parse HTML file {file_name}: {e_lxml}")
            else:
                print("  - Info: Skipping LXML extraction due to failure in getting/parsing XPaths from Gemini.")

        except FileNotFoundError:
             print(f"  - Error: File not found at {file_path}. Skipping.")
        except Exception as e:
            print(f"  - An unexpected error occurred while processing {file_name}: {e}")

    # --- Save Results to CSV ---
    if all_extracted_data:
        print(f"\nTotal entries extracted across all files: {len(all_extracted_data)}")
        df = pd.DataFrame(all_extracted_data)

        # Reorder columns for better readability in the final CSV
        column_order = [
            'source_file', 'major_name', 'major_url', 'major_description',
            'name_xpath', 'url_xpath', 'desc_xpath'
        ]
        
        # Ensure all columns exist before reordering to prevent errors
        for col in column_order:
            if col not in df.columns:
                df[col] = ''
        df = df[column_order]

        try:
            df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
            print(f"\nExtraction complete. Results saved to: {OUTPUT_CSV_PATH}")
            print("\n--- First 5 Rows ---")
            print(df.head().to_markdown(index=False, tablefmt="grid"))
            print("--------------------")
        except Exception as e:
            print(f"\nError saving results to CSV '{OUTPUT_CSV_PATH}': {e}")
    else:
        print("\nProcessing complete. No data was successfully extracted.")

# --- Script Execution ---
if __name__ == "__main__":
    main()

import configparser
import logging
import os
import pdf2image
import traceback

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.exceptions import HttpResponseError
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from prompt import IMG_JSON_GENERATOR_PROMPT
from pydantic import BaseModel, Field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union
from unstructured.partition.image import partition_image


class ScheduleItem(BaseModel):
    morning: int = Field(
        description="whether to take the drug in the morning, with 1 being 'yes' and 0 being 'no'.")
    afternoon: int = Field(
        description="whether to take the drug in the afternoon, with 1 being 'yes' and 0 being 'no'.")
    evening: int = Field(
        description="whether to take the drug in the evening, with 1 being 'yes' and 0 being 'no'.")
    night: int = Field(
        description="whether to take the drug in the night, with 1 being 'yes' and 0 being 'no'.")


class JSON_SCHEMA(BaseModel):
    drug_name: str = Field(description="Name of the drug")
    uom: str = Field(description="Unit of measurement for the drug")
    dosage: str = Field(description="The prescribed amount to consume")
    content_uom: str = Field(
        description="Amount of active ingredient per unit.")
    frequency: str = Field(description="Frequency to take the drug daily")
    instruction: str = Field(
        description="Additional instructions or reference on how to take the medicine (before or after food) and how long (up to 90 days)")
    condition: str = Field(description="What is the medicine used to treat")
    schedule: List[ScheduleItem] = Field(
        description="Schedule of medication in the morning, afternoon, evening and night")


log_path = 'logs/pdf_processor.log'
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# ---Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# --- Load all configuration settings ---
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), "config.prop")
config.read(config_path)
os.environ["AZURE_OPENAI_ENDPOINT"] = config['azure_openai']['azure_openai_endpoint']
os.environ["AZURE_OPENAI_API_KEY"] = config['azure_openai']['azure_openai_api_key']


class PDFProcessor:
    def __init__(self, chain):
        """
        Initialise PDF processor with Azure OpenAI configuration.

        Args:
            chain:      LLM chain
            di_client:  Document Intelligence client
        """
        self.chain = chain
        self.di_client = DocumentIntelligenceClient(
            endpoint=config['doc_intelligence']['endpoint'], credential=AzureKeyCredential(config['doc_intelligence']['key']))

    def extract_and_format_pdf_content(
        self,
        pdf_path: str,
        model_id: str = "prebuilt-layout"  # Default to prebuilt-layout, can be overridden
    ) -> Optional[str]:
        """
        Reads a PDF file, analyzes it using Azure Document Intelligence,
        and formats the extracted content (lines, selection marks, tables)
        into a single string.

        Args:
            pdf_path: The full path to the local PDF file.
            doc_intel_endpoint: The endpoint URL for the Azure Document Intelligence resource.
                                (Should NOT have a trailing slash).
            doc_intel_key: The API key for the Azure Document Intelligence resource.
            model_id: The model ID to use for analysis (defaults to "prebuilt-layout").

        Returns:
            A formatted string containing the extracted document content,
            or None if any error occurs during file reading or analysis.
        """

        # 1. Read the local file content as bytes
        file_content_bytes = None  # Initialize
        try:
            with open(pdf_path, "rb") as f:
                file_content_bytes = f.read()
            logger.info(f"Successfully read local file: {pdf_path}")
        except FileNotFoundError:
            logger.error(f"ERROR: File not found at path: {pdf_path}")
            return None  # Return None on error
        except Exception as e:
            logger.error(f"ERROR: Failed to read file '{pdf_path}': {e}")
            return None  # Return None on error

        # Ensure file content was read
        if not file_content_bytes:
            logger.error("ERROR: File content is empty or could not be read.")
            return None

        # 3. Analyse the document
        try:
            logger.info(
                f"Submitting document for analysis using model '{model_id}'...")
            poller = self.di_client.begin_analyze_document(
                model_id, AnalyzeDocumentRequest(
                    bytes_source=file_content_bytes)
            )

            logger.info("Waiting for analysis result...")
            result: AnalyzeResult = poller.result()
            logger.info("Analysis complete.")

            # --- Build Output String ---
            output_string = ""

            # 3a. Add Handwritten Status (if available)
            if result.styles:  # Check if styles exist
                # Often there's just one style block for the whole doc, or the first is representative
                style = result.styles[0]
                status = "handwritten" if style.is_handwritten else "no handwritten"
                output_string += f"Document contains {status} content.\n"
            else:
                output_string += "Document style information (handwritten status) not available.\n"

            output_string += "----------------------------------------\n"

            # 3b. Process Pages
            if result.pages:  # Check if pages exist
                for page in result.pages:
                    output_string += f"\n--- Page {page.page_number} ---\n"

                    # Add Lines (Primary content)
                    if page.lines:  # Check if lines exist
                        for line in page.lines:
                            # Use line.content directly (it's already a string)
                            output_string += line.content + "\n"
                    else:
                        output_string += "(No lines detected on this page)\n"

                    # Add Selection Marks (if any)
                    if page.selection_marks:  # Check if selection marks exist
                        output_string += f"\n--- Selection Marks on Page {page.page_number} ---\n"
                        for selection_mark in page.selection_marks:
                            # Use selection_mark.state (it's an enum, convert to string)
                            output_string += f"  Mark: {str(selection_mark.state)}, Confidence: {selection_mark.confidence:.4f}\n"

            else:
                output_string += "\n(No pages found in document analysis)\n"

            output_string += "----------------------------------------\n"

            # 4. Return the final consolidated string
            return output_string.strip()  # Remove leading/trailing whitespace

        except HttpResponseError as e:
            # Handle API errors (like 404, 401, etc.)
            logger.error(
                f"\nERROR during analysis (HTTP Status {e.status_code}): {e.message}")
            if hasattr(e, 'error') and hasattr(e.error, 'code') and hasattr(e.error, 'message'):
                logger.error(f"  Error Code: {e.error.code}")
                logger.error(f"  Error Message: {e.error.message}")
            logger.error(
                "\nPlease check your endpoint, key, model ID, file format, and network connection.")
            return None  # Return None on error
        except Exception as e:
            # Catch other potential errors
            logger.error(
                f"\nAn unexpected error occurred during analysis or processing: {e}")
            return None  # Return None on error

    def di_analyse_document(self, elements: str, selected_language: str) -> Dict[str, Any]:
        """
        Use Langchain with Azure OpenAI to analyze and structure medical document information.

        Args:
            elements: List of extracted text elements

        Returns:
            Structured medical document information
        """
        output_string = ""
        try:
            output_string = elements.strip()  # Remove any trailing whitespace

            # Invoke the chain with raw elements
            response = self.chain.invoke(
                {"context2": output_string, "language": selected_language})

            drug_names = {entry['drug_name'] for entry in response['json']}
            drugs = list(sorted(drug_names))

            logger.info(f"All Drugs: {drugs}")

            logger.info("Structured JSON Output:")
            logger.info(response["json"])

            response['json'] = self.schedule_corrector(response['json'])

            return response

        except Exception as e:
            logger.error(f"Medical document analysis failed: {e}")
            logger.error(traceback.format_exc())
            return {}

    def convert_pdf_to_images(self, pdf_path: str, output_folder: str = 'output_images') -> List[str]:
        """
        Convert PDF to images and save in output folder.

        Args:
            pdf_path: Path to input PDF
            output_folder: Destination for image outputs

        Returns:
            List of image file paths
        """
        try:
            # Create output directory if not exists
            os.makedirs(output_folder, exist_ok=True)

            # Convert PDF to images
            images = pdf2image.convert_from_path(
                pdf_path,
                poppler_path='path/to/poppler/bin' if os.name == 'nt' else None
            )

            # Save images
            image_paths = []
            for i, image in enumerate(images, 1):
                image_path = os.path.join(output_folder, f"page_{i}.jpg")
                image.save(image_path, "JPEG")
                image_paths.append(image_path)

            logger.info(f"Converted {len(images)} pages to images")
            return image_paths

        except Exception as e:
            logger.error(f"PDF to image conversion failed: {e}")
            logger.error(traceback.format_exc())
            return []

    def extract_image_text(self, image_path: str) -> List[dict]:
        """
        Extract text elements from an image.

        Args:
            image_path: Path to image file

        Returns:
            List of extracted text elements
        """
        try:
            elements = partition_image(image_path)
            return [
                {
                    "category": el.category,
                    "text": el.text.strip()
                }
                for el in elements
                if el.text.strip()
            ]
        except Exception as e:
            logger.error(f"Text extraction from {image_path} failed: {e}")
            return []

    def schedule_corrector(self, entries):
        """
        Takes a list of medication entries (each with a 'schedule' key that's a one‐element list of dicts)
        and normalises the schedule based on 'frequency' and 'when_to_take'.
        Returns the modified entries list.
        """
        times_of_day = ['morning', 'afternoon', 'evening', 'night']

        def mentions_any(when_str):
            """Return True if any of the times_of_day words appear in when_str."""
            when_lower = when_str.lower()
            return any(tok in when_lower for tok in times_of_day)

        for entry in entries:
            # 1) Safely unwrap the schedule dict (or create a fresh one)
            sched_list = entry.get('schedule', [])
            if sched_list and isinstance(sched_list, list):
                sched = sched_list[0]
            else:
                sched = {t: 0 for t in times_of_day}
                entry['schedule'] = [sched]

            # 2) Normalize all flags to 0 before setting true ones
            for t in times_of_day:
                sched[t] = 0

            freq = entry.get('frequency', '').upper()
            when = entry.get('when_to_take', '').lower()

            # 3) Apply rules
            if freq == 'OM':
                sched['morning'] = 1

            elif freq == 'ON':
                sched['night'] = 1

            elif 'QDS' in freq:
                for t in times_of_day:
                    sched[t] = 1

            elif 'TDS' in freq:
                # if “when required” or no explicit time words, assume morning/afternoon/evening
                if when == 'when required' or not mentions_any(when):
                    sched['morning'] = sched['afternoon'] = sched['evening'] = 1
                else:
                    for t in times_of_day:
                        if t in when:
                            sched[t] = 1

            elif 'BD' in freq:
                # if “when required” or no explicit time words, assume morning & night
                if when == 'when required' or not mentions_any(when):
                    sched['morning'] = sched['night'] = 1
                else:
                    for t in times_of_day:
                        if t in when:
                            sched[t] = 1

            # 4) put it back
            entry['schedule'][0] = sched

        return entries

    def analyse_document(self, elements: List[dict], selected_language: str) -> Dict[str, Any]:
        """
        Use Langchain with Azure OpenAI to analyze and structure medical document information.

        Args:
            elements: List of extracted text elements

        Returns:
            Structured medical document information
        """
        output_string = ""
        try:

            # Invoke the chain with raw elements
            for item in elements:
                if item['category'] in ['NarrativeText', 'Title', 'Table']:
                    if item['category'] == 'Title':
                        # Title acts as a section header
                        output_string += f"\n{item['text']}\n"
                    else:
                        # NarrativeText follows under the title
                        output_string += f"{item['text']}\n"

            output_string = output_string.strip()

            response = self.chain.invoke(
                {"context2": output_string, "language": selected_language})
            response['json'] = self.schedule_corrector(response['json'])

            return response

        except Exception as e:
            logger.error(f"Medical document analysis failed: {e}")
            logger.error(traceback.format_exc())
            return {}

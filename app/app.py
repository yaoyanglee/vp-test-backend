import asyncio
import base64
import concurrent.futures
import configparser
import copy
import cv2
import json
import io
import langchain
import logging
import numpy as np
import os
import pandas as pd
import re
import tempfile
import time
import typing
import uvicorn


from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential, AzureNamedKeyCredential
from azure.data.tables.aio import TableServiceClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.exceptions import HttpResponseError
from azure.storage.blob import BlobServiceClient
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Form, File, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fuzzywuzzy import fuzz
from io import BytesIO
from langchain_community.document_loaders import AzureBlobStorageFileLoader, DataFrameLoader, UnstructuredPDFLoader, PyPDFLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from multiprocessing import Pool
from operator import itemgetter
from rapidfuzz import process, fuzz
from PIL import Image, ImageEnhance
from app.pdf_processor import PDFProcessor
from app.structs.response.RawEntry import RawEntry
from app.structs.response.RxEntry import RxEntry
from app.structs.request.GetPatientDataRaw import GetPatientDataRaw
from app.routes.sse.sse_routes import sse_router
from app.prompt import CHAT_SYSTEM_PROMPT, STRUCTURED_GENERATOR_PROMPT, JSON_GENERATOR_PROMPT, IMG_EXTRACTION_PROMPT, IMG_OCR_PROMPT, IMG_JSON_GENERATOR_PROMPT
from pydantic import BaseModel, Field
from typing import Any, Iterator, List, Optional, Set, Tuple, Union


class ScheduleItem(BaseModel):
    morning: int = Field(
        description="Number of times for the drug to be taken in the morning")
    afternoon: int = Field(
        description="Number of times for the drug to be taken in the afternoon")
    evening: int = Field(
        description="Number of times for the drug to be taken in the evening")
    night: int = Field(
        description="Number of times for the drug to be taken in the night")


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


# --- Load all configuration settings ---
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), "config.prop")
config.read(config_path)

# --- Configure logger ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()  # Or any other handler you use
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Example format
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


parser = JsonOutputParser(pydantic_object=JSON_SCHEMA)

app = FastAPI()

# Allow all origins for development purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sse_router, prefix="/sse")

drug_mapping = {}

LANGUAGE_MAPPING = {'en': 'English',
                    'zh': 'Chinese', 'ms': 'Malay', 'ta': 'Tamil'}

os.environ["AZURE_OPENAI_ENDPOINT"] = config['azure_openai']['azure_openai_endpoint']
os.environ["AZURE_OPENAI_API_KEY"] = config['azure_openai']['azure_openai_api_key']
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Azure Blob configuration
CONNECTION_STRING = config["azure_storage"]["connection_string"]

# Text file container
CONTAINER_NAME_TXT = config['azure_storage']["container_name_txt"]

# RX pdf container for user testing
CONTAINER_NAME_RX = config['rx_user_testing']["container_name_pdf"]

# Azure Vision configuration
VISION_API_KEY = config['azure_vision']['azure_vision_api_key']
VISION_API_ENDPOINT = config['azure_vision']['azure_vision_endpoint']


try:
    logger.info("Initialising Blob Service Client...")
    blob_service_client = BlobServiceClient.from_connection_string(
        CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(
        CONTAINER_NAME_TXT)
    rx_client = blob_service_client.get_container_client(
        CONTAINER_NAME_RX)
    logger.info(
        f"Successfully connected to txt container: {CONTAINER_NAME_TXT}")
    logger.info(f"Successfully connected to rx container: {CONTAINER_NAME_RX}")
except Exception as e:
    logger.error(f"Failed to initialise Blob Service Client: {e}")
    raise

all_test_rx = list(rx_client.list_blobs())

# Azure Table Service Connection
table_config = config["azure_table"]
account_name, account_key = table_config["account_name"], table_config["account_key"]

credential = AzureNamedKeyCredential(account_name, account_key)
endpoint = f"https://{account_name}.table.core.windows.net"
table_service_client = TableServiceClient(
    endpoint=endpoint, credential=credential)


# Function to get table client
def get_table_client(table_name):
    return table_service_client.get_table_client(table_name)

# Function to fetch all entities from a table


async def get_all_entities(table_name):
    entities = []
    table_client = get_table_client(table_name)

    async for entity in table_client.list_entities():
        entities.append(entity["PartitionKey"])
    return entities


def transform_df(df):
    """
    Transforms the Patients CSV Data into the required format.

    Parameters:
        df (pd.DataFrame): The pandas DataFrame containing patient data.

    Returns:
        pd.DataFrame: The transformed DataFrame with standardized UOM and formatted content.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty.")
        return df

    # Exclude rows where 'uom' is 'EA' and reset the index
    df = df[df['uom'] != 'EA'].reset_index(drop=True)

    # UOM Mapping Dictionary
    uom_mapping = {
        'BTL': 'Bottle',
        'CP': 'Capsule',
        'IJ': 'Injection',
        'PKT': 'Packet',
        'TA': 'Tablet',
        'TBE': 'Tube',
        'VAL': 'Injection'
    }

    # Standardize UOM values
    df['uom'] = df['uom'].map(uom_mapping)

    # Generate a formatted content string
    df['content'] = df.apply(
        lambda row: f"Drug: {row['item_description']}, UOM: {row['uom']}, Dosage instructions: {row['dosage_instr']}", axis=1)

    return df


# Sample anonymised medication structured data
path = os.path.join(os.path.dirname(__file__), "testing_data.csv")
full_df = (lambda x: transform_df(pd.read_csv(x)))(path)

# Create an Image Analysis client
vision_client = ImageAnalysisClient(
    endpoint=VISION_API_ENDPOINT,
    credential=AzureKeyCredential(VISION_API_KEY)
)

# Initialise Azure Vision Client for OCR operations
logger.info("Successfully established connection to Azure Vision Client...")

image_path = "./img/"

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version="2024-10-01-preview",
    temperature=0.0,
    max_retries=3
)

# Medication Summary Prompt
answer_generation_prompt = PromptTemplate(
    template=CHAT_SYSTEM_PROMPT, input_variables=["context", "language"])

# CSV Chain
structured_output_prompt = PromptTemplate(
    template=STRUCTURED_GENERATOR_PROMPT, input_variables=["context", "pdfcontext"])
json_output_prompt = PromptTemplate(template=JSON_GENERATOR_PROMPT, input_variables=[
                                    "sum_context", "language"], partial_variables={"format_instructions": parser.get_format_instructions()})

structured_chain_1 = structured_output_prompt | llm
structured_chain_2 = json_output_prompt | llm | parser

complete_chain = ({
    "sum_context": structured_chain_1,
    "context": itemgetter("context"),
    "pdfcontext": itemgetter("pdfcontext"),
    "language": itemgetter("language"),
}
    | RunnablePassthrough.assign(json=structured_chain_2)
)

# Image Chain
img_ocr_prompt = PromptTemplate(
    template=IMG_OCR_PROMPT, input_variables=["context"])
img_structured_output_prompt = PromptTemplate(
    template=IMG_EXTRACTION_PROMPT, input_variables=["context"])
img_json_output_prompt = PromptTemplate(template=IMG_JSON_GENERATOR_PROMPT, input_variables=[
                                        "context2", "language"], partial_variables={"format_instructions": parser.get_format_instructions()})

ocr_parser = img_ocr_prompt | llm
img_structured_chain_1 = img_structured_output_prompt | llm
img_structured_chain_2 = img_json_output_prompt | llm | parser


img_complete_chain = (
    {
        "context2": itemgetter("context2"),
        "language": itemgetter("language")
    }
    | RunnablePassthrough.assign(json=img_structured_chain_2)
)

pdf_processor = PDFProcessor(img_complete_chain)


def dewarp_text(image: np.ndarray) -> np.ndarray:
    """
    Corrects curved or distorted text in an image using contour detection and perspective transformation.

    Parameters:
    - image : np.ndarray
        Input image as a NumPy array (expected to be in RGB format).

    Returns:
    - np.ndarray
        Dewarped (corrected) image if successful, otherwise returns the original image.
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image

        # Find the largest contour (assumed to be the text area)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Get width and height of the detected rectangle
        width, height = map(int, rect[1])

        if width == 0 or height == 0:
            return image

        # Define source and destination points for perspective transform
        src_pts = box.astype("float32")
        dst_pts = np.array([
            [0, height - 1],
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1]
        ], dtype="float32")

        # Compute perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        dewarped = cv2.warpPerspective(image, matrix, (width, height))

        return dewarped

    except Exception as e:
        logger.warning(f"Dewarping failed: {e}")
        return image


def image_rescaler(
    img: Image.Image,
    min_length: int = 2048,
    max_length: int = 4096,
    min_dpi: int = 400,
    output_format: str = "png",
    quality: int = 100
) -> Tuple[np.ndarray, bytes]:
    """
    Rescales an image for optimal OCR performance, adjusting size based on resolution and text content analysis.

    Parameters:
    - img (Image.Image): The input PIL image to be processed.
    - min_length (int, optional): Minimum length for the longer side of the image. Defaults to 2048.
    - max_length (int, optional): Maximum allowed length for the longer side of the image. Defaults to 4096.
    - min_dpi (int, optional): Minimum DPI required for OCR accuracy. Defaults to 400.
    - output_format (str, optional): Output image format, recommended as 'png' for lossless quality. Defaults to 'png'.
    - quality (int, optional): Image compression quality. Has no effect on PNG but impacts JPEG compression. Defaults to 100.

    Returns:
    - Tuple[np.ndarray, bytes]: A tuple containing:
        1. The processed image as a NumPy array.
        2. The image bytes in the specified output format.
    """
    try:
        width, height = img.size
        max_side = max(width, height)
        min_side = min(width, height)

        logger.info(f"Original image size: {width}x{height}")

        # Convert to NumPy array for analysis
        np_img = np.array(img)

        # Try to correct curved text
        np_img = dewarp_text(np_img)

        # Convert back to PIL Image for further processing
        img = Image.fromarray(np_img)

        # Check image resolution (DPI)
        try:
            dpi_x, dpi_y = img.info.get('dpi', (72, 72))
            current_dpi = min(dpi_x, dpi_y)
        except (AttributeError, TypeError):
            current_dpi = 72

        logger.debug(f"Image DPI: {current_dpi}")

        # Enhanced text detection function
        def analyze_text_regions(image: np.ndarray) -> Tuple[bool, float]:
            """
            Analyzes image for text regions and their characteristics.
            Returns whether image needs upscaling and recommended scale factor.
            """
            try:
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                # Apply adaptive thresholding
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )

                # Find contours
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Analyze text characteristics
                small_components = 0
                text_heights = []

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0

                    # Filter likely text components
                    if 0.1 < aspect_ratio < 15 and h > 3:
                        text_heights.append(h)
                        if h < (height * 0.015):
                            small_components += 1

                # Calculate recommended scale based on text size distribution
                if text_heights:
                    median_height = np.median(text_heights)
                    target_height = height * 0.03
                    recommended_scale = min(
                        target_height / median_height, 3.0) if median_height > 0 else 1.0
                else:
                    recommended_scale = 1.0

                needs_upscaling = small_components > 15 or recommended_scale > 1.5

                return needs_upscaling, recommended_scale

            except Exception as e:
                logger.warning(f"Text analysis failed: {e}")
                return False, 1.0

        # Analyze text and get scaling recommendation
        needs_upscaling, text_scale = analyze_text_regions(np_img)

        # Determine if resizing is needed
        needs_resize = any([
            max_side < min_length,
            current_dpi < min_dpi,
            needs_upscaling
        ])

        if needs_resize:
            # Calculate optimal scale factor
            dpi_scale = min_dpi / current_dpi if current_dpi > 0 else 2
            min_length_scale = min_length / max_side

            # Use the largest necessary scale factor, but don't exceed max_length
            scale = min(max(dpi_scale, min_length_scale,
                        text_scale), max_length / max_side)

            new_width = int(width * scale)
            new_height = int(height * scale)

            # Enhanced resizing with sharpening
            resized_image = img.resize(
                (new_width, new_height), Image.Resampling.LANCZOS)

            # Apply subtle sharpening
            enhancer = ImageEnhance.Sharpness(resized_image)
            resized_image = enhancer.enhance(1.5)  # Increased sharpening

            # Enhance contrast slightly
            contrast_enhancer = ImageEnhance.Contrast(resized_image)
            resized_image = contrast_enhancer.enhance(1.2)

            logger.info(f"Image enhanced to: {new_width}x{
                        new_height} (scale: {scale:.2f})")
        else:
            resized_image = img
            logger.info("Image quality acceptable; no resizing performed")

        # Convert final image to NumPy array
        np_img = np.array(resized_image)

        # Get byte representation
        b = io.BytesIO()
        resized_image.save(b, format=output_format.upper(), quality=quality)
        im_bytes = b.getvalue()

        logger.debug(f"Output size: {len(im_bytes)} bytes")

        return np_img, im_bytes

    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        raise RuntimeError(f"Image processing failed: {e}") from e


def convert_points_to_dict(points):
    """
    Converts a list of ImagePoint objects into a serializable dictionary format.

    Parameters:
    - points (List[ImagePoint]): A list of ImagePoint objects, each containing x and y coordinates.

    Returns:
    - List[dict]: A list of dictionaries, where each dictionary represents a point with 'x' and 'y' keys.
    """
    return [{'x': point.x, 'y': point.y} for point in points]


def azure_ocr_parser(results):
    """
    Parses the OCR results from Azure Vision and extracts text in multiple formats.

    Parameters:
    - results (Azure OCR Result): The OCR output from Azure Vision API.

    Returns:
    - dict: A dictionary containing:
        - 'concatenated_text' (str): All detected text concatenated with spaces.
        - 'newline_text' (str): Text with preserved line breaks.
        - 'structured_text' (dict): A structured representation with:
            - 'lines' (list): A list of detected lines, each containing:
                - 'text' (str): The extracted line text.
                - 'bounding_box' (list of dicts): The bounding polygon of the line.
            - 'words_with_confidence' (list): A list of words, each containing:
                - 'word' (str): The recognized word.
                - 'confidence' (float): The confidence score of recognition.
                - 'bounding_box' (list of dicts): The bounding polygon of the word.

    If an error occurs during processing, logs the error and returns None.
    """
    # Initialize collections
    all_lines = []
    structured_text = {
        'lines': [],
        'words_with_confidence': []
    }

    try:
        if results.read:
            for block in results.read.blocks:
                for line in block.lines:
                    # Store full line text
                    all_lines.append(line.text)
                    structured_text['lines'].append({
                        'text': line.text,
                        'bounding_box': convert_points_to_dict(line.bounding_polygon)
                    })

                    # Store individual words with confidence scores
                    for word in line.words:
                        structured_text['words_with_confidence'].append({
                            'word': word.text,
                            'confidence': word.confidence,
                            'bounding_box': convert_points_to_dict(word.bounding_polygon)
                        })

        # Create different text formats
        concatenated_text = ' '.join(all_lines)
        newline_text = '\n'.join(all_lines)

        return {
            'concatenated_text': concatenated_text,
            'newline_text': newline_text,
            'structured_text': structured_text
        }
    except Exception as e:
        logger.error(f"Error processing OCR results: {e}")
        return None


def fuzzy_dedup(df, similarity_threshold=98):
    """ 
    Checks for similarity and removes duplicates

    Parameters:
    - df (pd.DataFrame): The formatted dataframe of the patients data
    - similarity_threshold (int): The expected similarity percentage to decide which duplicates to drop

    Outputs:
    - df_copy (pd.DataFrame): The dataframe of the patients data without duplicates
    """
    def check_similarity(d):
        dupl_indexes = []
        for i in range(len(d.values) - 1):
            for j in range(i + 1, len(d.values)):
                if fuzz.token_sort_ratio(d.values[i], d.values[j]) >= similarity_threshold:
                    dupl_indexes.append(d.index[j])
        return dupl_indexes

    indexes_to_drop = check_similarity(df['content'])
    df_copy = df.copy()
    df_copy.drop(indexes_to_drop, inplace=True)
    return df_copy


def filter_df(df, patient_id):
    """
    Filter Dataframe based on patient_id

    Parameters:
    - df (pd.DataFrame): The formatted df from transform_df
    - patient_id (str): The ID for each patient (eg. B3144XXXXX)

    Outputs:
    - docs (DataFrameLoader): The drug instructions from csv.
    - prescriptions (list): The list of string of drug names
    """
    prescriptions = []

    condition = df['ANYM_ID_NO'] == patient_id
    filtered_df = df[condition]
    rm_df = fuzzy_dedup(filtered_df)

    logger.debug(rm_df)

    for drug in rm_df["item_description"]:
        prescriptions.append(drug)

    rm_df = rm_df.drop(columns=["item_description", "dosage_instr"])
    rm_df['content'] = rm_df['content'].fillna('')
    data_loader = DataFrameLoader(rm_df, page_content_column="content")
    docs = data_loader.load_and_split()
    return docs, prescriptions


async def extract_conditions(patient_id: str, prescriptions: list, selected_language: str):
    """
    Extracting the conditions from the medication summary to add to the csv data

    Parameters:
    - patient_id (str): The ID for each patient (eg. B3144XXXXX)
    - prescriptions (list): A list of prescriptions for the specified patient

    Outputs:
    - formatted_response (dict): Returns the dictionary of the medication summary 
    - medinfo (dict): Returns the dictionary of the conditions from the medication summary
    """
    formatted_response = await azure_get_data(prescriptions, selected_language)
    medinfo = copy.deepcopy(formatted_response['results'])

    fields_to_drop = ["Administration", "Common side effects", "Storage"]
    for med in medinfo:
        for field in fields_to_drop:
            med.pop(field, None)
    return formatted_response, medinfo


async def azure_get_data(prescriptions: list, selected_language: str):
    """
    Performs parallel calls to obtain the Medication Summary using asyncio

    Parameters:
    - prescriptions (list): A list of prescriptions for the specified patient
    - selected_language (str): Selected language for the response

    Outputs:
    - Dictionary of the Medication Summary
    """
    blobs = list(container_client.list_blobs())
    gmappings = await get_all_entities("gmapping")
    leaflet_names = [blob['name'][:-4].lower() for blob in blobs]
    prescriptions = [prescription.lower() for prescription in prescriptions]

    logger.info(f"*** Prescriptions: {prescriptions}")

    try:
        if not prescriptions:
            raise HTTPException(
                status_code=400, detail="Prescription is empty")

        # Create tasks for all prescriptions
        tasks = [
            azure_get_query_data(
                prescription, "", selected_language, leaflet_names, gmappings)
            for prescription in prescriptions
        ]

        # Gather all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        drug_results = []
        raw_strings = []
        seen_drug_names = set()  # Keep track of drug names we've already added

        # Log the original count
        logger.info(f"*** Original drug_results count: {len(results)}")

        for result_item in results:
            if isinstance(result_item, Exception):
                logger.error(f"Error processing result: {result_item}")
                continue

            # Ensure the item is a tuple with two elements
            if not (isinstance(result_item, tuple) and len(result_item) == 2):
                logger.warning(
                    f"Skipping malformed result item (not a 2-tuple): {result_item}")
                continue

            drug_dict, raw_string = result_item

            # Filter out items where drug_dict is None or not a dict, or raw_string is None/empty
            # This mimics the original `if result and result[0] and result[1]` behavior for such cases.
            if not (drug_dict and isinstance(drug_dict, dict) and raw_string):
                logger.info(
                    "Skipping result item due to missing or invalid components")
                continue

            drug_name = drug_dict.get('drug_name')

            if drug_name and isinstance(drug_name, str) and drug_name.strip():
                # Normalize drug name for consistent checking (e.g., lowercase and strip whitespace)
                normalized_drug_name = drug_name.lower().strip()

                if normalized_drug_name not in seen_drug_names:
                    seen_drug_names.add(normalized_drug_name)
                    drug_results.append(drug_dict)
                    raw_strings.append(raw_string)
                else:
                    logger.info(
                        f"Duplicate drug found and discarded: {drug_name} (Normalized: {normalized_drug_name})")
            else:
                # This case handles entries that have a valid dict and raw_string, but no 'drug_name' key or an empty drug_name.
                # These are not "drug duplicates" so they are kept.
                logger.info(
                    f"Entry has no valid 'drug_name' or 'drug_name' is empty, adding as is: {drug_dict}")
                drug_results.append(drug_dict)
                raw_strings.append(raw_string)

        logger.info(
            f"*** Deduplicated drug_results count: {len(drug_results)}")

        return {"results": drug_results, "raw": '|||'.join(raw_strings)}

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}"}


async def azure_get_query_data(
    med: str,
    patient_id: str = "",  # This parameter is not used in the function body shown
    selected_language: str = "",
    leaflet_names=None,
    gmappings=None
):
    """
    Retrieves the contextualized medication information and generates an answer.
    If no context is found for the medication, it returns None for the drug data.

    Parameters:
    - med : str
        The medication to search for in the database. This is a required parameter.
    - patient_id : str, optional
        The unique identifier of the patient. Defaults to an empty string if not provided.
    - selected_language : str, optional
        The language for the generated response. Defaults to an empty string if not provided.
    - leaflet_names : list, optional
        A list of leaflet names used to filter the search results. This parameter is required by signature but not used in current logic.

    Returns:
    - tuple
        A tuple containing:
        - A dictionary (from formatted string) with the contextualized medication response, or None if no context.
        - The raw response content generated by the answer generator, or a message indicating no context.

    Raises:
    ValueError
        If `leaflet_names` is not provided (though not used in current logic).
    """
    formatted_data = {}
    med_content = ""

    # Check if leaflet_names is provided
    if leaflet_names is None:
        logger.warning("leaflet_names is required but not provided.")

    # Step 1: Search mapping to get a list of filenames related to the query
    filtered_matches = await search_tables(med, leaflet_names, gmappings)
    logger.info(f"Filtered Matches for '{med}': {filtered_matches}")

    # Step 2: Extract and format the relevant context from the search response
    rag_ctx = await azure_get_context(filtered_matches)

    if rag_ctx and len(rag_ctx) >= 100:
        # Step 3: Configure the medication summary generation pipeline
        answer_generator = answer_generation_prompt | llm

        # Generate the response
        logger.info(f"Invoking LLM for '{med}' with context.")
        med_response = await answer_generator.ainvoke({
            'prescription': med,
            'context': rag_ctx,
            'language': selected_language
        })
        med_content = med_response.content
        # Ensure format_string can handle various LLM outputs
        formatted_data = format_string(med_content)

    return formatted_data, med_content


async def azure_get_context(matches):
    """
    Concatenates the context of the medication NDF using asyncio to load and process multiple blobs concurrently.

    Parameters:
    - matches : list
        A list of blob names (strings) representing the matches related to the medication. Each blob corresponds
        to a context that will be loaded asynchronously.

    Returns:
    - str
        A concatenated string containing the valid contents of all the successful NDF blobs, joined by newlines.
        If some blobs fail, they are ignored and not included in the final result.
    """

    # Create tasks for all matches
    tasks = [azure_load_HH_txt(blob) for blob in matches]

    # Gather all results concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out any errors and join the successful results
    valid_results = [result for result in results if isinstance(result, str)]

    return "\n".join(valid_results)


async def search_tables(query: str, leaflet_names: list, gmappings: list) -> List[str]:
    """
    Performs a fuzzy search on various tables (brands, groupings, and HH files) to retrieve matching filenames 
    based on the provided query.

    Parameters:
    - query : str
        The search query (e.g., medication name or ingredient) used for fuzzy matching against multiple datasets.

    Returns:
    - list
        A list of filenames (with `.txt` extension) that match the query or its derived terms from the search process.
    """
    threshold = 90
    search_terms: Set[str] = set()

    # --- Fuzzy search in gmapping table ---
    gmapping_filtered_matches = fuzzy_search(
        query.lower(), choices=gmappings, threshold=threshold)
    logger.debug(
        f"Grouping matches for '{query}': {gmapping_filtered_matches}")

    gmapping_tasks = [retrieve("gmapping", match)
                      for match in gmapping_filtered_matches]
    gmapping_results_list = await asyncio.gather(*gmapping_tasks)
    gmapping_results = dict(
        zip(gmapping_filtered_matches, gmapping_results_list))

    # Log what we found in gmapping
    logger.debug(f"Grouping results: {gmapping_results}")

    for match, records in gmapping_results.items():
        for record in records:
            gmapping = record["RowKey"].split(", ")
            search_terms.update(gmapping)
            logger.debug(f"Added groupings from {match}: {gmapping}")

    logger.debug(f"All search terms collected: {search_terms}")

    # --- Search HH files based on ingredients and groupings ---
    hh_matches: Set[str] = set()
    for term in search_terms:
        term_matches = fuzzy_search(
            term, choices=leaflet_names, threshold=threshold)
        hh_matches.update(term_matches)
        logger.debug(f"Matches for term '{term}': {term_matches}")

    # --- If no matches, search the original query ---
    if not hh_matches:
        direct_matches = fuzzy_search(
            query.lower(), choices=leaflet_names, threshold=threshold)
        hh_matches.update(direct_matches)
        logger.debug(f"Direct matches for '{query}': {direct_matches}")

    logger.info(f"Final HH File Headers for {query}: {hh_matches}")

    # --- Return file names with .txt extension ---
    filenames = [match + ".txt" for match in hh_matches]
    return filenames


async def retrieve(table_name, partition_value):
    """
    Retrieve entities from an Azure Table Storage table asynchronously based on PartitionKey.

    Parameters:
    - table_name (str): The name of the Azure Table Storage table.
    - partition_value (str): The PartitionKey value to filter records.

    Returns:
    - list: A list of entities (dictionaries) matching the PartitionKey.
              Returns an empty list if no matches are found or an error occurs.
    """
    table_client = get_table_client(table_name)
    filter_query = f"PartitionKey eq '{partition_value}'"

    try:
        async with table_client:
            entities = []
            async for entity in table_client.query_entities(query_filter=filter_query):
                entities.append(entity)
            return entities
    except Exception as e:
        logger.error(f"Error retrieving from {table_name}: {e}")
        return []


def fuzzy_search(query: str, choices: list, threshold: int) -> list:
    """
    Performs a fuzzy search on the provided choices to find matches that meet the given threshold.

    Parameters:
    - query : str
        The search query to be matched against the available choices.
    - choices : list
        A list of possible choices (strings) to search through.
    - threshold : int
        The minimum score required for a match to be considered valid. A higher threshold means stricter matching.

    Returns:
    - list
        A list of strings representing the matches that have a score greater than or equal to the threshold.

    """
    return [match[0] for match in process.extract(query.lower(), choices, limit=5) if match[1] >= threshold]


async def azure_load_HH_txt(blob: str) -> str:
    """
    Loads text content from an Azure blob file, attempting different encodings for successful decoding.

    Parameters:
    - blob : str
        The name of the blob file to be loaded from Azure.

    Returns:
    - str
        The content of the blob as a string, or None if the blob could not be loaded or decoded successfully.
    """
    try:
        # Get the blob name
        blob_name = blob
        logger.info(f"Attempting to load blob file: {blob_name}")

        # Get the blob client
        blob_client = container_client.get_blob_client(blob_name)

        # First try: Download the blob content directly
        try:
            blob_content = blob_client.download_blob().readall()
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'ascii', 'iso-8859-1']

            for encoding in encodings:
                try:
                    content = blob_content.decode(encoding)
                    logger.info(
                        f"Successfully decoded {blob_name} with {encoding} encoding")
                    return content
                except UnicodeDecodeError:
                    logger.warning(
                        f"Failed to decode {blob_name} with {encoding} encoding")
                    continue

            # If we get here, none of the encodings worked
            logger.error(
                f"Could not decode {blob_name} with any known encoding")
            return None

        except Exception as download_error:
            logger.error(
                f"Direct blob download failed for {blob_name}: {download_error}")

            # Fallback: Try using AzureBlobStorageFileLoader
            logger.info(
                f"Attempting fallback with AzureBlobStorageFileLoader for {blob_name}")
            loader = AzureBlobStorageFileLoader(
                conn_str=CONNECTION_STRING,
                container="ndf-drugs-txt",
                blob_name=blob_name
            )
            try:
                single_doc = loader.load()
                if single_doc and len(single_doc) > 0:
                    return single_doc[0].page_content
                else:
                    logger.error(
                        f"Loader returned empty content for {blob_name}")
                    return None
            except Exception as loader_error:
                logger.error(
                    f"Loader fallback failed for {blob_name}: {loader_error}")
                return None

    except Exception as e:
        logger.error(f"Critical error loading blob {blob[0]}: {str(e)}")
        logger.error(f"Full error details: {type(e).__name__}: {str(e)}")
        return None


def format_string(text: str, delimiter="\n"):
    """
    This is to format the string into a proper dictionary 

    Parameters:
    - text (str): String of contexts
    - delimiter (str): The specified delimiter to split the str

    Returns:
    - res[0] (dict): Returns a dictionary
    """
    res = []
    text = text.replace("------", "######").replace("**", "")
    target = text.split("######")
    for drug in target:
        drug_info = {}
        for segment in drug.split(delimiter):
            if segment.startswith('[Drug name]'):
                drug_info['drug_name'] = segment.split('[Drug name]:')[
                    1].strip()
                # drug_name = segment.split('[Drug name]: ')[1]
            elif segment.startswith('[Conditions]'):
                drug_info['Conditions'] = segment.split('[Conditions]:')[
                    1].strip()
            elif segment.startswith('[Administration]'):
                drug_info['Administration'] = segment.split(
                    '[Administration]:')[1].strip()
            elif segment.startswith('[Common side effects]'):
                drug_info['Common side effects'] = segment.split(
                    '[Common side effects]:')[1].strip()
            elif segment.startswith('[Storage]'):
                drug_info['Storage'] = segment.split('[Storage]:')[1].strip()

        # Append drug_info to res only if it contains some valid data
        if any(key in drug_info for key in ['drug_name', 'Conditions', 'Administration', 'Common side effects', 'Storage']):
            res.append(drug_info)

    logger.debug("+"*153)
    logger.debug(res)
    logger.debug("+"*153)
    return res[0] if len(res) > 0 else None


def list_to_str(image_to_text, chain):
    """
    Returns the formatted drugs information after OCR extract and processing

    Parameters:
    - drug_lst (list): The array that represents the image
    - chain: The chain to process each drug

    Returns:
    - final_resp (str): The processed context
    """
    results = ""
    logger.info(f"Generated text description from uploaded image:\n{
                image_to_text}")

    # Use a regex to safely split based on numbered bullet points
    # Match numbers followed by ". **Drug Name:**"
    pattern = r"\d+\.\s(?=\*\*Drug Name:\*\*)"
    items = re.split(pattern, image_to_text)

    # Remove the initial introduction or empty sections
    items = [item.strip() for item in items if "**Drug Name:**" in item]

    # Iterate over each item and pass as context to the chain
    for item in items:
        # Pass the current item as context
        response = chain.invoke({'context': item})
        logger.info(f"**** RESPONSE {response.content}")
        results += str(response.content) + "\n\n"
    return results


async def process_single_file(file: UploadFile, selected_language: str):
    """
    Returns the drugs list and pillbox schedule for each drug in each image or PDF.

    Parameters:
    - file (UploadFile): image or PDF file from an uploaded file
    - selected_language (str): language preference

    Returns:
    - drugs (list): The names of formatted drugs
    - json_output (dict): The pillbox schedule
    """
    response = {}
    drugs = []

    logger.info(f"FILE: {file.filename}")

    try:
        if file.content_type.startswith("image/"):
            if "VP" in file.filename:
                logger.info("Processing Image-PDF file...")

                all_elements = []

                # Determine file extension (important for tempfile)
                file_extension = "." + \
                    file.filename.split(".")[-1].lower()  # Get extension
                # Handle edge cases
                if file_extension not in (".jpg", ".jpeg", ".png", ".gif", ".bmp"):
                    file_extension = ".jpg"  # Default to JPG if unknown

                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_image:
                    # Write the image data to the temporary file
                    image_data = await file.read()
                    tmp_image.write(image_data)
                    temp_file_path = tmp_image.name  # Get the path

                # Verify the file was created and is a valid image:
                if not os.path.exists(temp_file_path):
                    raise FileNotFoundError(
                        "Temporary file was not created successfully.")

                try:
                    image_elements = pdf_processor.extract_image_text(
                        temp_file_path)
                    all_elements.extend(image_elements)
                except Exception as e:
                    logger.error(
                        f"Error processing image {temp_file_path}: {e}")

                if all_elements:
                    try:
                        response = pdf_processor.analyse_document(
                            all_elements, selected_language)
                    except Exception as e:
                        logger.error(f"Error analyzing document: {e}")
                else:
                    logger.warning(
                        "No text elements extracted from PDF images.")

            else:
                logger.info("Processing image file...")

                # Load and preprocess image
                loaded_image = Image.open(io.BytesIO(await file.read()))
                img = loaded_image.convert(
                    'RGB') if loaded_image.mode == 'RGBA' else loaded_image

                _, enhanced_image_bytes = image_rescaler(
                    img,
                    min_length=2048,
                    min_dpi=400,
                    quality=100
                )

                # Analyse with Azure Vision OCR
                ocr_results = vision_client.analyze(
                    enhanced_image_bytes,
                    visual_features=[VisualFeatures.READ],
                    gender_neutral_caption=True
                )

                parsed_results = azure_ocr_parser(ocr_results)
                ocr_output = parsed_results.get('concatenated_text', '')

                logger.debug(f"OCR output: {ocr_output}")

                ai_message = ocr_parser.invoke(ocr_output)

                # Process AI message with complete chain
                response = img_complete_chain.invoke({
                    "context2": ai_message.content,
                    "language": selected_language
                })

        elif file.content_type == "application/pdf":
            logger.info("Processing PDF file...")

            tmp_pdf_path = None
            image_paths = []

            # Write PDF to a temp file
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(await file.read())
                    tmp_pdf_path = tmp_pdf.name

                extracted_content = pdf_processor.extract_and_format_pdf_content(
                    pdf_path=tmp_pdf_path)
                response = pdf_processor.di_analyse_document(
                    extracted_content, selected_language)

            except Exception as e:
                logger.error(f"Error converting PDF to images: {e}")

            finally:
                # Clean up temp file
                if tmp_pdf_path and os.path.exists(tmp_pdf_path):
                    os.remove(tmp_pdf_path)

        else:
            logger.warning(f"Unsupported file type: {file.content_type}")
            return [], {}

        # Validate and extract drugs
        if 'json' in response and response['json']:
            drug_names = {entry.get('drug_name', '')
                          for entry in response['json'] if 'drug_name' in entry}
            drugs = sorted(filter(None, drug_names))

            logger.info(f"Extracted drugs: {drugs}")
            logger.info("Structured JSON Output:")
            logger.info(response["json"])
        else:
            logger.warning("No JSON data found in response.")
            response = {"json": {}}  # Ensure it always returns a json field

    except Exception as e:
        logger.exception(f"Unexpected error during image/PDF processing: {e}")
        response = {"json": {}}

    return drugs, response["json"]


async def find_similarity(d):
    dupl_indexes = set()
    for i in range(len(d) - 1):
        for j in range(i + 1, len(d)):
            if fuzz.token_sort_ratio(d[i].lower(), d[j].lower()) >= 90:
                dupl_indexes.add(j)
    dup_idx = list(dupl_indexes)
    dup_idx.sort(reverse=True)
    return dup_idx


async def process_all_files(images: List[UploadFile], patient_id: str = "", selected_language: str = "English"):
    """
    Returns the cumulated drugs list and pillbox schedule for every drug in all images

    Parameters:
    - image (list): images from an uploaded files
    - patient_id (str): Patient id

    Returns:
    - all_drug_name_occurrences (list): The compiled names of drugs formatted
    - all_schedule_entries (list): The compiled pillbox schedule for all images
    """
    all_drug_name_occurrences = []
    all_schedule_entries = []

    processed_results = await asyncio.gather(*(process_single_file(img, selected_language) for img in images))

    for file_drug_names, file_schedule_entries in processed_results:
        all_drug_name_occurrences.extend(file_drug_names)
        all_schedule_entries.extend(file_schedule_entries)

    indexes_to_drop = await find_similarity(all_drug_name_occurrences)

    final_drug_names_list = list(all_drug_name_occurrences)
    final_pillbox_schedules_list = list(all_schedule_entries)

    for i in indexes_to_drop:
        del final_drug_names_list[i]
        del final_pillbox_schedules_list[i]

    return final_drug_names_list, final_pillbox_schedules_list


@app.get("/select/all")
def get_all_options():
    test_path = os.path.join(os.path.dirname(__file__), "test_data.csv")
    with open(test_path, 'r') as f:
        data = f.readlines()[1:]

    ids = list(set([row.split(',')[0] for row in data]))
    return sorted(ids)


@app.get("/select/{id}")
def get_by_id(id):
    payload = ""
    with open(f"./static/{id.split('.')[0]}.json") as f:
        payload = json.load(f)
    return payload


@app.get("/api/v3/get-patient-info/{patient_id}")
async def get_patient_info(patient_id: str = "", language: Optional[str] = Query(None)):
    """Retrieve patient information, process medical data, and return a summary."""
    start_time = datetime.now()  # Track start time
    logger.info(f"Retrieving information for rx_id: {patient_id}")

    all_test_rx_names = [blob['name'] for blob in all_test_rx]
    rx_doc = [name for name in all_test_rx_names if patient_id in name][0]
    logger.info(f"Found blob document: {rx_doc}")

    # Get the blob name
    blob_name = rx_doc
    logger.info(f"Attempting to load blob file: {blob_name}")

    # Get the blob client
    blob_client = rx_client.get_blob_client(blob_name)

    # Determine selected language
    selected_language = LANGUAGE_MAPPING.get(language, "English")
    logger.info(f"Selected language: {selected_language}")

    # 1. Download the blob bytes
    blob_bytes = blob_client.download_blob().readall()

    tmp_pdf_path = None
    try:
        # 2. Write to a NamedTemporaryFile (just like your old file.read())
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(blob_bytes)
            tmp_pdf_path = tmp_pdf.name

        # 3. Call exactly the same extractor & analyzer
        extracted_content = pdf_processor.extract_and_format_pdf_content(
            pdf_path=tmp_pdf_path)
        response = pdf_processor.di_analyse_document(
            extracted_content, selected_language)

    except Exception as e:
        logger.error(f"Error processing blob PDF {blob_name}: {e}")
        response = None

    finally:
        # 4. Cleanup
        if tmp_pdf_path and os.path.exists(tmp_pdf_path):
            os.remove(tmp_pdf_path)

    # Validate and extract drugs
    if 'json' in response and response['json']:
        drug_names = {entry.get('drug_name', '')
                      for entry in response['json'] if 'drug_name' in entry}
        drugs = sorted(filter(None, drug_names))

        logger.info(f"Extracted drugs: {drugs}")
        logger.info("Structured JSON Output:")
        logger.info(response["json"])
    else:
        logger.warning("No JSON data found in response.")
        response = {"json": {}}  # Ensure it always returns a json field

    indexes_to_drop = await find_similarity(drugs)

    final_drug_names_list = list(drugs)
    final_pillbox_schedules_list = list(response["json"])

    for i in indexes_to_drop:
        del final_drug_names_list[i]
        del final_pillbox_schedules_list[i]

    logger.debug(f"*** Original Pillbox: {final_pillbox_schedules_list}")

    pillbox_list = normalise_pillbox(final_pillbox_schedules_list)

    logger.debug(f"*** Normalised Pillbox: {pillbox_list}")

    # Log retrieved medications
    drug_string = ", ".join(final_drug_names_list)
    logger.info(f"All Drugs: {drug_string}")
    logger.info(f"Pillbox: {pillbox_list}")

    logger.info("Creating medication summary")

    # Retrieve structured medication data
    res = await azure_get_data(prescriptions=final_drug_names_list, selected_language=selected_language)

    # Attach pillbox visual data
    res["info"] = pillbox_list

    # Log generated visual materials
    logger.debug("+" * 80)
    logger.debug(f"Pillbox Summary: {res['info']}")
    logger.debug("+" * 80)

    # Compute processing time
    end_time = datetime.now()
    total_seconds = (end_time - start_time).total_seconds()
    logger.info(
        f"Processing time: {total_seconds:.2f} seconds ({total_seconds / 60:.2f} minutes)")
    logger.info("Successfully created all visual materials")

    return res


def normalise_pillbox(pillbox):
    cleaned_pillbox = []

    for drug in pillbox:
        if not drug["schedule"]:  # Remove if schedule is empty
            continue

        # Check if the latest schedule has at least one non-zero value
        latest_schedule = drug["schedule"][-1]
        if sum(latest_schedule.values()) == 0:
            # Find the most recent non-zero schedule
            for sched in reversed(drug["schedule"]):
                if sum(sched.values()) > 0:
                    latest_schedule = sched
                    break
            else:
                continue  # Skip this entry if all schedules are zero

        # Keep only the relevant schedule entry
        drug["schedule"] = [latest_schedule]
        cleaned_pillbox.append(drug)

    return cleaned_pillbox


@app.post("/api/v3/from-image/{patient_id}")
async def from_image(
    images: List[UploadFile] = File(...),
    patient_id: str = "",
    language: Optional[str] = Query(None)
):
    start_time = datetime.now()  # Track start time
    """Process uploaded images to extract medication information and generate a visual pillbox summary."""

    # Determine selected language
    selected_language = LANGUAGE_MAPPING.get(language, "English")
    logger.info(f"Selected Language: {selected_language}")

    # Process images asynchronously
    drug_list, pillbox_list = await process_all_files(images, patient_id, selected_language)

    logger.debug(f"*** Original Pillbox: {pillbox_list}")

    pillbox_list = normalise_pillbox(pillbox_list)

    logger.debug(f"*** Normalised Pillbox: {pillbox_list}")

    # Log retrieved medications
    drug_string = ", ".join(drug_list)
    logger.info(f"All Drugs: {drug_string}")
    logger.info(f"Pillbox: {pillbox_list}")

    logger.info("Creating medication summary")

    # Retrieve structured medication data
    res = await azure_get_data(prescriptions=drug_list, selected_language=selected_language)

    # Attach pillbox visual data
    res["info"] = pillbox_list

    # Log generated visual materials
    logger.debug("+" * 80)
    logger.debug(f"Pillbox Summary: {res['info']}")
    logger.debug("+" * 80)

    # Compute processing time
    end_time = datetime.now()
    total_seconds = (end_time - start_time).total_seconds()
    logger.info(
        f"Processing time: {total_seconds:.2f} seconds ({total_seconds / 60:.2f} minutes)")
    logger.info("Successfully created all visual materials")

    return res


@app.get("/api/v3/get-raw")
def get_raw_all():

    test_path = os.path.join(os.path.dirname(__file__), "test_data.csv")
    with open(test_path, 'r') as f:
        all_lines = f.readlines()[1:]

    results = []

    # for line in all_lines:
    #     curLine = line.strip('\n').split(',')
    #     if len(curLine) > 0:
    #         results.append(RawEntry(
    #             patient_id=curLine[0],
    #             item_description=curLine[1],
    #             item_code=curLine[2],
    #             quantity=curLine[3],
    #             uom=curLine[4],
    #             dosage_instruction=curLine[5],
    #             frequency=curLine[6],
    #             dispensed_duration=curLine[7],
    #             dose=curLine[8],
    #         ))
    for line in all_lines:
        curLine = line.strip('\n').split(',')
        if len(curLine) > 0:
            results.append(RxEntry(
                rx_id=curLine[0],
                rx_url=curLine[1]
            ))
    logger.info({"data": results})
    return {"data": results}


@app.get("/test")
def test():
    # logger = Logger()
    # logger.log("Patient Testing", "Message for patient")
    return "hello"


if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=9000, reload=True)

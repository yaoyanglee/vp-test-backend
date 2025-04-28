import asyncio
import base64
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
import time
import typing
import uvicorn


from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from fastapi import FastAPI, HTTPException, Query, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fuzzywuzzy import fuzz
from io import BytesIO
from langchain_community.document_loaders import AzureBlobStorageFileLoader, DataFrameLoader, UnstructuredPDFLoader, PyPDFLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
# Import issues
from app.logger.Logger import Logger
from multiprocessing import Pool
from operator import itemgetter
from rapidfuzz import process, fuzz
from PIL import Image, ImageEnhance
# Import issues added absolute path instead
from app.structs.response.RawEntry import RawEntry
from app.structs.request.GetPatientDataRaw import GetPatientDataRaw
from app.routes.sse.sse_routes import sse_router
from app.prompt import CHAT_SYSTEM_PROMPT, STRUCTURED_GENERATOR_PROMPT, JSON_GENERATOR_PROMPT, IMG_EXTRACTION_PROMPT, IMG_OCR_PROMPT, IMG_JSON_GENERATOR_PROMPT
from pydantic import BaseModel, Field
from typing import Any, Iterator, List, Optional, Tuple, Union

import concurrent.futures


class ScheduleItem(BaseModel):
    morning: int = Field(
        description="Number of times for the drug to be taken in the morning")
    afternoon: int = Field(
        description="Number of times for the drug to be taken in the afternoon")
    evening: int = Field(
        description="Number of times for the drug to be taken in the evening")


class JSON_SCHEMA(BaseModel):
    drug_name: str = Field(description="Name of the drug")
    uom: str = Field(description="Unit of measurement for the drug")
    dosage: str = Field(description="Dosage to be taken for the drug")
    content_uom: str = Field(
        description="Dosage or the unit of the measurement")
    frequency: str = Field(description="Frequency to take the drug daily")
    instruction: str = Field(description="Meal preference")
    condition: str = Field(description="What is the medicine used to treat")
    schedule: List[ScheduleItem] = Field(
        description="Schedule with morning, afternoon, evening values")


# --- Load all configuration settings ---
config = configparser.ConfigParser()
# config.read(
#     r'./config.prop')
config_path = os.path.join(os.path.dirname(__file__), "config.prop")
config.read(config_path)

# --- Configure logger ---
# Create a named logger
logger = logging.getLogger(__name__)

# Configure the logger
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

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
CONTAINER_NAME = config["azure_storage"]["container_name"]
# Text file container
CONTAINER_NAME_TXT = config['azure_storage']["container_name_txt"]

# Azure Vision configuration
VISION_API_KEY = config['azure_vision']['azure_vision_api_key']
VISION_API_ENDPOINT = config['azure_vision']['azure_vision_endpoint']

try:
    logger.info("Initialising Blob Service Client...")
    blob_service_client = BlobServiceClient.from_connection_string(
        CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(
        CONTAINER_NAME_TXT)
    logger.info(f"Successfully connected to container: {CONTAINER_NAME_TXT}")

except Exception as e:
    logger.error(f"Failed to initialise Blob Service Client: {e}")
    raise
blobs = list(container_client.list_blobs())

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
    template=STRUCTURED_GENERATOR_PROMPT, input_variables=["context", "pdfcontext", "language"])
json_output_prompt = PromptTemplate(template=JSON_GENERATOR_PROMPT, input_variables=[
                                    "sum_context"], partial_variables={"format_instructions": parser.get_format_instructions()})

structured_chain_1 = structured_output_prompt | llm
structured_chain_2 = json_output_prompt | llm | parser

complete_chain = ({
    "sum_context": structured_chain_1,
    "context": itemgetter("context"),
    "pdfcontext": itemgetter("pdfcontext"),
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


# --- Helper Functions ---
def dewarp_text(image: np.ndarray) -> np.ndarray:
    """
    Attempts to correct curved text in images.
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
            thresh, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Find the largest contour (assumed to be the text area)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Get width and height of the detected rectangle
            width = int(rect[1][0])
            height = int(rect[1][1])

            # Get the perspective transform
            src_pts = box.astype("float32")
            dst_pts = np.array([
                [0, height-1],
                [0, 0],
                [width-1, 0],
                [width-1, height-1]
            ], dtype="float32")

            # Apply perspective transform
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            dewarped = cv2.warpPerspective(image, matrix, (width, height))

            return dewarped
    except Exception as e:
        logger.warning(f"Dewarping failed: {e}")
        return image

    return image


def image_rescaler(img: Image.Image,
                   min_length: int = 2048,
                   max_length: int = 4096,
                   min_dpi: int = 400,     # Optimized DPI setting
                   output_format: str = 'png',  # Changed to PNG for better quality
                   # Maximized quality
                   quality: int = 100) -> Tuple[np.ndarray, bytes]:
    """
    Adjusts image size based on resolution requirements and content analysis.
    Enhanced for optimal OCR performance.

    Note on parameters:
    - min_dpi = 400: Best balance between quality and file size. 500+ DPI rarely improves OCR
                     but significantly increases processing time and file size
    - quality = 100: For PNG, this doesn't affect quality but affects compression
                     For JPEG, 100 gives best quality but larger file size
    - output_format = 'png': Lossless format better for text recognition
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
    Convert ImagePoint objects to serializable dictionary format
    """
    return [{'x': point.x, 'y': point.y} for point in points]


def azure_ocr_parser(results):
    """
    Extract text from Azure Vision OCR result in different formats
    Returns both concatenated string and structured formats
    """
    # Initialize collections
    all_lines = []
    structured_text = {
        'lines': [],
        'words_with_confidence': []
    }

    try:
        if results.read is not None:
            for block in results.read.blocks:
                for line in block.lines:
                    # Add full line to collections
                    all_lines.append(line.text)
                    structured_text['lines'].append({
                        'text': line.text,
                        'bounding_box': convert_points_to_dict(line.bounding_polygon)
                    })

                    # Add individual words with confidence
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
        logger.error(f"Error processing OCR results: {str(e)}")
        return None


def transform_df(df):
    """Transform the Patients CSV Data into Required Format

    Parameters:
        df (pd.DataFrame): The pandas dataframe of the patients data

    Outputs:
        df (pd.DataFrame): The formatted dataframe of the patients data
    """
    df = df[df['uom'] != 'EA'].reset_index(drop=True)
    id2term = {'TA': 'Tablet', 'IJ': 'Injection'}
    df['uom'] = df['uom'].map(id2term)
    df['content'] = 'Drug: ' + df['item_description'] + ', UOM: ' + df['uom'] + ', Dosage instructions: ' + df[
        'dosage_instr']
    return df


# Global Patient Dataframe
# path = r'code\app\testing_data.csv'
testing_csv_path = os.path.join(os.path.dirname(__file__), "testing_data.csv")
full_df = (lambda x: transform_df(pd.read_csv(x)))(testing_csv_path)


# CSV Helper Functions
def fuzzy_dedup(df, similarity_threshold=98):
    """ Checks for similarity and removes duplicates

    Parameters:
        df (pd.DataFrame): The formatted dataframe of the patients data
        similarity_threshold (int): The expected similarity percentage to decide which duplicates to drop

    Outputs:
        df_copy (pd.DataFrame): The dataframe of the patients data without duplicates
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
        df (pd.DataFrame): The formatted df from transform_df
        patient_id (str): The ID for each patient (eg. B3144XXXXX)

    Outputs:
        docs (DataFrameLoader): The drug instructions from csv.
        prescriptions (list): The list of string of drug names
    """
    condition = df['ANYM_ID_NO'] == patient_id
    filtered_df = df[condition]
    rm_df = fuzzy_dedup(filtered_df)

    prescriptions = []
    for drug in rm_df["item_description"]:
        prescriptions.append(drug)

    rm_df = rm_df.drop(columns=["item_description", "dosage_instr"])
    data_loader = DataFrameLoader(rm_df, page_content_column="content")
    docs = data_loader.load_and_split()
    return docs, prescriptions


def extract_conditions(patient_id: str, prescriptions: list, selected_language: str):
    """
    Extracting the conditions from the medication summary to add to the csv data

    Parameters:
        patient_id (str): The ID for each patient (eg. B3144XXXXX)
        prescriptions (list): A list of prescriptions for the specified patient

    Outputs:
        formatted_response (dict): Returns the dictionary of the medication summary 
        medinfo (dict): Returns the dictionary of the conditions from the mdeication summary
    """
    # Change this line to call azure_get_data()
    formatted_response = azure_get_data(prescriptions, selected_language)
    medinfo = copy.deepcopy(formatted_response['results'])

    fields_to_drop = ["Administration", "Common side effects", "Storage"]
    for med in medinfo:
        for field in fields_to_drop:
            med.pop(field, None)

    return formatted_response, medinfo


def azure_get_data(prescriptions: list, selected_language: str):
    """
    Performs parallelisation calls to obtain the Medication Summary

    Parameters:
        prescriptions (list): A slit of prescriptions for the specified patient
        patient_id (str, optional): Patient id of the patient

    Outputs:
        Dictionary of the Medication Summary
    """
    # Change all to lower case
    prescriptions = [prescription.lower() for prescription in prescriptions]

    results = []
    logger.info(f"Prescription: {prescriptions}")
    blobs_names = [blob['name'] for blob in blobs]
    try:
        if prescriptions == []:
            raise HTTPException(
                status_code=400, detail="Prescription is empty")
        '''
        with Pool() as pool:
            results = pool.starmap(azure_get_query_data, [(
                prescription, "", selected_language, blobs_names) for prescription in prescriptions])
                
        # Filter out None values from results
        filtered_results = [
            drug for drug in results if drug and drug[0] and drug[1]]

        # Extract drug names and raw strings
        drug_results = [drug[0] for drug in filtered_results]
        raw_strings = [drug[1] for drug in filtered_results]

        return {"results": drug_results, "raw": '|||'.join(raw_strings)}
                
        '''
        drug_results = []
        raw_strings = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(azure_get_query_data, prescription,
                                "", selected_language, blobs_names)
                for prescription in prescriptions
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    # Collect the result as soon as it's completed
                    result = future.result()
                    if result and result[0] and result[1]:
                        drug_results.append(result[0])
                        raw_strings.append(result[1])
                except Exception as e:
                    print(f"Error processing future result: {e}")

        # Join the raw strings together with a delimiter
        return {"results": drug_results, "raw": '|||'.join(raw_strings)}

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}"}


def azure_get_query_data(med: str, patient_id: str = "", selected_language: str = "", blobs_names=None):
    """
    Retrieves the contextualized medication information and generates an answer.

    Parameters:
        med (str): The medication query.
        patient_id (str): Optional patient identifier (currently unused).
        selected_language (str): Language preference for the answer.

    Returns:
        tuple: Formatted string of the generated response and the raw response content.
    """
    if blobs_names is None:
        raise ValueError("blobs_names is required but not provided.")
    # SEARCH_BLOB
    threshold = 80
    matches = process.extract(
        med, blobs_names, scorer=fuzz.partial_token_set_ratio, limit=5)

    print("Matches: \n", matches)

    filtered_matches = [match for match in matches if match[1] >= threshold]
    # Step 2: Extract and format the relevant context from the search response.
    logger.info(f"Filtered Matches for {med}: {filtered_matches}")
    rag_ctx = azure_get_context(filtered_matches)

    # Step 3: Configure the answer generation pipeline using the extracted context and selected language.
    answer_generator = answer_generation_prompt | llm

    # Generate the response using the LLM with the provided prescription, context, and language.
    med_response = answer_generator.invoke({
        'prescription': med,
        'context': rag_ctx,
        'language': selected_language
    })

    # Step 4: Format and return the generated response.
    return format_string(med_response.content), med_response.content


def azure_get_context(matches):
    """
    Concatenating the context of the medication ndf

    Parameters:
        response (list): A list of documents

    Outputs:
        context (str): Returns a formatted context of the medication ndf in a string
    """
    context = ""

    # Use ThreadPoolExecutor to load PDFs for the drugs in the query concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Retrieval of txt files from the container with the text files from the azure storage account
        futures = [executor.submit(azure_load_ndf_txt, blob)
                   for blob in matches]

        # Collect the results as they complete
        for future in concurrent.futures.as_completed(futures):
            context += future.result() + "\n"
    return context


def azure_load_ndf_txt(blob):
    """
    Load text from Azure blob file.

    Parameters:
        blob (str): The blob file
    """
    try:
        print("blob: ", blob)
        # Get the blob name
        blob_name = blob[0]
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
                    logger.info(f"Successfully decoded {
                                blob_name} with {encoding} encoding")
                    return content
                except UnicodeDecodeError:
                    logger.warning(f"Failed to decode {
                                   blob_name} with {encoding} encoding")
                    continue

            # If we get here, none of the encodings worked
            logger.error(f"Could not decode {
                         blob_name} with any known encoding")
            return None

        except Exception as download_error:
            logger.error(f"Direct blob download failed for {
                         blob_name}: {download_error}")

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
                logger.error(f"Loader fallback failed for {
                             blob_name}: {loader_error}")
                return None

    except Exception as e:
        logger.error(f"Critical error loading blob {blob[0]}: {str(e)}")
        logger.error(f"Full error details: {type(e).__name__}: {str(e)}")
        return None


def format_string(text: str, delimiter="\n"):
    """
    This is to format the string into a proper dictionary 

    Parameters:
        text (str): String of contexts
        delimiter (str): The specified delimiter to split the str

    Outputs:
        res[0] (dict): Returns a dictionary
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

    logger.info("+"*153)
    logger.info(res)
    logger.info("+"*153)
    return res[0] if len(res) > 0 else None


def list_to_str(image_to_text, chain):
    """
    Returns the formatted drugs information after OCR extract and processing

    Parameters:
        drug_lst (list): The array that represents the image
        chain: The chain to process each drug

    Outputs:
        final_resp (str): The processed context
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


async def img_process_single_image(image: UploadFile, selected_language: str):
    """
    Returns the drugs list and pillbox schedule for each drug in each image

    Parameters:
        image (UploadFile): image from an uploaded file

    Outputs:
        drugs (list): The names of drugs formatted
        json_output (dict): The pillbox schedule
    """
    json_parser = JsonOutputParser(pydantic_object=JSON_SCHEMA)
    loadedImage = Image.open(io.BytesIO(await image.read()))

    img = loadedImage

    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    # Rescale image
    _, enhanced_image_bytes = image_rescaler(
        img,
        min_length=2048,  # Increased minimum size
        min_dpi=400,      # Higher DPI requirement
        quality=100        # High quality output
    )

    # Analyze with Azure Vision
    ocr_results = vision_client.analyze(
        enhanced_image_bytes,  # Convert the image to bytes
        visual_features=[VisualFeatures.READ],
        gender_neutral_caption=True
    )

    # Parse OCR results to be returned in the specified formats: full concatenated string, multi-texts with newline, structured
    parsed_results = azure_ocr_parser(ocr_results)
    ocr_output = parsed_results['concatenated_text']

    logger.info(f"OCR concatenated output:{ocr_output}")

    ai_message = ocr_parser.invoke(parsed_results['concatenated_text'])

    response = img_complete_chain.invoke(
        {"context2": ai_message.content, "language": selected_language})
    drug_names = {entry['drug_name'] for entry in response['json']}
    drugs = list(sorted(drug_names))

    logger.info(f"ALL DRUGS: {drugs}")

    return drugs, response["json"]


async def img_process_images(images: List[UploadFile], patient_id: str = "", selected_language: str = "English"):
    """
    Returns the cumulated drugs list and pillbox schedule for every drug in all images

    Parameters:
        image (list): images from an uploaded files
        patient_id (str): Patient id

    Outputs:
        drugs (list): The complied names of drugs formatted
        pillboxes (list): The complied pillbox schedule for all images
    """

    drugs = []
    pillboxes = []

    img_grps = await asyncio.gather(*(img_process_single_image(img, selected_language) for img in images))

    for img_grp in img_grps:
        drugs.extend(img_grp[0])
        pillboxes.extend(img_grp[1])

    def find_similarity(d):
        dupl_indexes = set()
        for i in range(len(d) - 1):
            for j in range(i + 1, len(d)):
                if fuzz.token_sort_ratio(d[i].lower(), d[j].lower()) >= 90:
                    dupl_indexes.add(j)
        dup_idx = list(dupl_indexes)
        dup_idx.sort(reverse=True)
        return dup_idx

    indexes_to_drop = find_similarity(drugs)
    drugs_copy = copy.deepcopy(drugs)

    for i in indexes_to_drop:
        del drugs_copy[i]
        del pillboxes[i]
    return drugs_copy, pillboxes


@app.get("/select/all")
def get_all_options():
    with open("phase_2\testing_data.csv", 'r') as f:
        data = f.readlines()[1:]

    patient_ids = list(set([row.split(',')[0] for row in data]))
    return sorted(patient_ids)


@app.get("/select/{id}")
def get_by_id(id):
    payload = ""
    with open(f"./static/{id.split('.')[0]}.json") as f:
        payload = json.load(f)
    return payload


@app.get("/api/v3/get-patient-info/{patient_id}")
def get_patient_info(patient_id: str = "", language: Optional[str] = Query(None)):
    medical_summary = {}
    logger = Logger()

    logger.log(patient_id, f"Getting patient information")

    selected_language = LANGUAGE_MAPPING.get(
        language, 'English')  # Default to 'English'
    logger.info(f"Selected Language: {selected_language}")

    try:
        docs, prescriptions = filter_df(full_df, patient_id)
        drug_string = ", ".join(prescriptions)
        logger.log(patient_id, f"Drugs retrieved: {drug_string}")

        logger.log(
            patient_id, f"Extracting conditions for patient: {patient_id}")

        logger.log(patient_id, f"Creating summary and pillbox content")
        content = "\n".join(str(d.page_content) for d in docs)
        medical_summary, pdfcontext = extract_conditions(
            patient_id, prescriptions, selected_language)

        logger.log(patient_id, f"Creating context for summary")

        context = complete_chain.invoke(
            input={'context': content, 'pdfcontext': pdfcontext, 'language': selected_language})
        logger.info(f"Translated Response: {context}")
        logger.info("+"*153)
        logger.info(f"Translated Response[JSON]:", context['json'])
        logger.info("+"*153)
        medical_summary["info"] = context['json']
        logger.info(f"Updated Medical Summary: {medical_summary}")
    except Exception as e:
        logger.log(
            patient_id, f"Exception occurred while invoking complete_chain: {e}")

    logger.log(patient_id, "Successfully created Pillbox and Summary!")
    return medical_summary


@app.post("/api/v3/from-image/{patient_id}")
def from_image(images: List[UploadFile] = File(...), patient_id: str = "", language: Optional[str] = Query(None)):
    selected_language = LANGUAGE_MAPPING.get(
        language, 'English')  # Default to 'English'
    logger.info(f"Selected Language: {selected_language}")

    drug_list, pillbox_list = asyncio.run(
        img_process_images(images, patient_id, selected_language))

    drug_string = ', '.join(drug_list)
    logger.info(f"Drugs retrieved: {drug_string}")
    logger.info("Creating medication summary")
    # change here to azure_data_NEW
    res = azure_get_data(prescriptions=drug_list,
                         selected_language=selected_language)
    res["info"] = pillbox_list
    logger.info("+"*153)
    logger.info(res["info"])
    logger.info("+"*153)
    logger.info("Successfully created all visual materials")
    return res


@app.get("/api/v3/get-raw")
def get_raw_all():

    with open("phase_2\testing_data.csv", 'r') as f:
        all_lines = f.readlines()[1:]

    results = []

    for line in all_lines:
        curLine = line.strip('\n').split(',')
        if len(curLine) > 0:
            results.append(RawEntry(
                patient_id=curLine[0],
                item_description=curLine[1],
                item_code=curLine[2],
                quantity=curLine[3],
                uom=curLine[4],
                dosage_instruction=curLine[5],
                frequency=curLine[6],
                dispensed_duration=curLine[7],
                dose=curLine[8],
            ))

    return {"data": results}


@app.get("/test")
def test():
    logger = Logger()
    logger.log("Patient Testing", "Message for patient")
    return "hello"


if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=9000, reload=True)

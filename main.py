from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import logging
import os
from PIL import Image, ImageDraw
from google.cloud import vision
from dotenv import load_dotenv
import numpy as np
from scipy import ndimage
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# CORS origins
origins = [
    'https://front-image-recognition-5vzpdcj6zq-uc.a.run.app',
    'http://front-image-recognition-5vzpdcj6zq-uc.a.run.app',
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    'https://localhost:3000',
    'https://127.0.0.1:3000'
]

# Adding CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Google Cloud Vision client
client = vision.ImageAnnotatorClient()

def find_largest_white_rectangle(image_path: str):
    try:
        # Load the image
        pil_image = Image.open(image_path).convert("RGB")
        np_image = np.array(pil_image)

        # Define the threshold for white areas
        threshold = 240
        white_areas = np.all(np_image > threshold, axis=-1)

        # Find bounding boxes for white areas
        labeled_array, num_features = ndimage.label(white_areas)

        max_area = 0
        largest_rectangle = None

        for i in range(1, num_features + 1):
            slice_x, slice_y = ndimage.find_objects(labeled_array == i)[0]
            (x, y, w, h) = (slice_x.start, slice_y.start, slice_x.stop - slice_x.start, slice_y.stop - slice_y.start)
            area = w * h
            if area > max_area:
                max_area = area
                largest_rectangle = (x, y, w, h)

        if largest_rectangle is None:
            raise ValueError("No white rectangles found in the background image.")

        logger.info(f"Largest white rectangle found at {largest_rectangle}")
        return largest_rectangle

    except Exception as e:
        logger.error(f"Error in find_largest_white_rectangle: {e}")
        raise HTTPException(status_code=500, detail="Error processing the image")

def apply_logo(image_path: str, background_image_path: str):
    try:
        # Load the background image and the upload image
        background_image = Image.open(background_image_path).convert("RGB")
        upload_image = Image.open(image_path).convert("RGB")

        # Detect the largest white rectangle in the background image
        largest_rectangle = find_largest_white_rectangle(background_image_path)

        x, y, w, h = largest_rectangle

        # Resize the upload image to fit the largest white rectangle
        resized_upload_image = upload_image.resize((h, w), Image.ANTIALIAS)

        # Paste the resized upload image onto the background image at the position of the largest white rectangle
        background_image.paste(resized_upload_image, (y, x))

        # Save the processed image
        processed_path = os.path.join('app', 'processed', os.path.basename(image_path))
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        background_image.save(processed_path)

        logger.info(f"Image processed and saved to {processed_path}")
        return processed_path

    except Exception as e:
        logger.error(f"Error in apply_logo: {e}")
        raise HTTPException(status_code=500, detail="Error applying the logo")

@app.post("/upload-campaign/")
async def upload_campaign_logo(file: UploadFile = File(...), background_name: str = Form(...)):
    try:
        os.makedirs('app/campaigns', exist_ok=True)  # Create 'campaigns' directory if it does not exist
        campaign_image_path = os.path.join('app', 'campaigns', file.filename)
        with open(campaign_image_path, "wb") as campaign_image:
            campaign_image.write(await file.read())

        # Apply campaign image to the background image
        background_image_path = os.path.join('app', 'images', background_name)  # Use the correct path to your background image
        if not os.path.exists(background_image_path):
            raise HTTPException(status_code=404, detail="Background image not found")

        processed_path = apply_logo(campaign_image_path, background_image_path)

        logger.info(f"File {file.filename} uploaded and processed with background {background_name}")
        return {"filename": file.filename, "processed_path": processed_path}

    except Exception as e:
        logger.error(f"Error in upload_campaign_logo: {e}")
        raise HTTPException(status_code=500, detail="Error uploading the campaign logo")

@app.get("/processed/{filename}")
async def get_processed_image(filename: str):
    try:
        file_path = os.path.join('app', 'processed', filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path)
    except Exception as e:
        logger.error(f"Error in get_processed_image: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving the processed image")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
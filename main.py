from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import logging
import os
from PIL import Image, ImageDraw, ImageOps
from google.cloud import vision
from dotenv import load_dotenv
from fpdf import FPDF
import numpy as np
from scipy import ndimage
import uuid
import csv
import re

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
    'https://127.0.0.1:3000',
    'https://testedevisibilidade.be180.com.br',
    'http://testedevisibilidade.be180.com.br'
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

def determine_suffix(filename: str) -> str:
    """Determina o sufixo correto para a imagem."""
    parts = filename.split('_')  # Exemplo: ["midia", "mub02", "noite", "3M.png"]
    base_suffix = parts[-1].replace('.png', '')  # Exemplo: "3M", "15M", "30M"

    # Verifica se o nome contém 'noite' e posiciona o sufixo corretamente
    if "noite" in filename.lower():
        return f"noite_{base_suffix}"
    return base_suffix


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

def apply_logo(image_path: str, background_image_path: str, common_id: str, suffix: str):
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

        # Save the processed image with the unique ID and suffix
        processed_path = os.path.join('app', 'processed', f"{common_id}_{suffix}.jpg")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        background_image.save(processed_path)

        logger.info(f"Image processed and saved to {processed_path}")
        return processed_path

    except Exception as e:
        logger.error(f"Error in apply_logo: {e}")
        raise HTTPException(status_code=500, detail="Error applying the logo")

def add_table(pdf):
    table_data = [
        ("ITEM", "PERGUNTA A SE FAZER", "DO", "DON'TS", "AÇÃO"),
        ("COR", "Existe contraste de cor na peça privilegiando os elementos principais?", "Se sim, aprovado", "Se não, reprovado", "Revisar criação"),
        ("", "As cores da marca estão presentes na peça?", "Se sim, aprovado", "Se não, reprovado", "Validar o motivo"),
        ("", "Existe degradê ou transparência em áreas que precisa de atenção?", "Se não, aprovado", "Se sim, reprovado", "Revisar criação"),
        ("TEXTO", "Os caracteres estão grandes e legíveis?", "Se sim, aprovado", "Se não, reprovado", "Revisar criação"),
        ("", "O texto legal foi incluído?", "Se sim, aprovado", "Se não, reprovado", "Validar o motivo"),
        ("HIERARQUIA", "Existe um líder na peça? É quem deveria ser?", "Se sim, aprovado", "Se não, reprovado", "Validar o motivo"),
        ("", "O logo está facilmente visível?", "Se sim, aprovado", "Se não, reprovado", "Revisar criação"),
        ("", "O layout está equilibrado em proporções considerando o formato?", "Se sim, aprovado", "Se não, reprovado", "Revisar criação")
    ]

    col_widths = [35, 85, 30, 30, 40]  # Largura das colunas
    line_height = 10  # Altura da linha

    for row in table_data:
        for i, item in enumerate(row):
            pdf.cell(col_widths[i], line_height, txt=item, border=1, align='C')
        pdf.ln(line_height)

def sort_images(image_paths: list) -> list:
    """
    Ordena as imagens para garantir que as diurnas (3M, 15M, 30M) venham antes
    e as noturnas (noite 3M, noite 15M, noite 30M) fiquem agrupadas depois.
    """
    def extract_key(filename: str):
        # Extrai se é 'noite' e a distância para ordenação
        is_night = 'noite' in filename.lower()

        # Usa regex para capturar o número da distância
        match = re.search(r'(\d+)m', filename.lower())
        distance = int(match.group(1)) if match else 0

        return (is_night, distance)

    # Ordena primeiro pelas diurnas e depois pelas noturnas
    return sorted(image_paths, key=extract_key)


def generate_pdf(image_paths: list, common_id: str):
    try:
        # Ordena as imagens antes de gerar o PDF
        image_paths = sort_images(image_paths)

        # Definindo o nome e o caminho do PDF
        pdf_name = f"{common_id}.pdf"
        pdf_path = os.path.join('app', 'pdfs', pdf_name)
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

        pdf = FPDF(orientation='P', unit='pt', format=(595, 2151))
        pdf.add_page()

        # Adiciona o fundo azul do menu
        pdf.set_fill_color(1, 72, 255)  # Cor #0148FF
        pdf.rect(0, 0, 595, 75, 'F')

        # Adiciona o logo
        pdf.set_xy(244, 23)
        pdf.image('app/images/novo_logo.png', w=106.42, h=30)

        # Título "Resultados de visibilidade"
        pdf.set_xy((595 - 268) / 2, 123)
        pdf.set_font("Arial", 'B', 24)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(268, 23, "Resultados de visibilidade", align='C')

        # Texto explicativo
        pdf.set_xy((595 - 460) / 2, 160)
        pdf.set_font("Arial", '', 16)
        pdf.multi_cell(
            460, 21,
            "Visualize abaixo, seu layout aplicado, já no formato de mídia selecionado em distâncias distintas.",
            align='C'
        )

        # Adiciona as imagens diurnas primeiro
        distances = ['3 m', '15 m', '30 m']
        top_positions = [244, 706, 1168]

        for i, distance in enumerate(distances):
            if i < len(image_paths) and os.path.exists(image_paths[i]):
                pdf.set_xy((595 - 45) / 2, top_positions[i] - 34)
                pdf.set_font("Arial", 'B', 20)
                pdf.cell(45, 19, distance, align='C')

                pdf.set_xy((595 - 460) / 2, top_positions[i])
                pdf.image(image_paths[i], w=460, h=369.85)

        # Adiciona uma nova página para as imagens noturnas, se houver
        if len(image_paths) > 3:
            pdf.add_page()
            pdf.set_xy((595 - 268) / 2, 123)
            pdf.set_font("Arial", 'B', 24)
            pdf.cell(268, 23, "Resultados de visibilidade - Noite", align='C')

            for i, distance in enumerate(distances):
                night_image_index = i + 3
                if night_image_index < len(image_paths) and os.path.exists(image_paths[night_image_index]):
                    pdf.set_xy((595 - 45) / 2, top_positions[i] - 34)
                    pdf.set_font("Arial", 'B', 20)
                    pdf.cell(45, 19, f"Noite - {distance}", align='C')

                    pdf.set_xy((595 - 460) / 2, top_positions[i])
                    pdf.image(image_paths[night_image_index], w=460, h=369.85)

        # Adiciona o checklist
        pdf.set_xy((595 - 460) / 2, 1630)
        pdf.set_font("Arial", 'B', 20)
        pdf.cell(460, 20, "Check-List de boas práticas para seu layout", align='C')

        pdf.set_xy((595 - 460) / 2, 1655)
        pdf.set_font("Arial", '', 16)
        checklist_text = (
            "Para facilitar o entendimento do resultado acima, disponibilizamos "
            "um check-list de boas práticas, que você consiga analisar o que deve ser "
            "observado na aprovação dos layouts de campanhas para OOH."
        )
        pdf.multi_cell(460, 21, checklist_text, align='C')

        # Imagem da tabela
        table_image_path = os.path.join('app', 'images', 'table_pdf.jpeg')
        pdf.set_xy((595 - 480) / 2, 1791)
        pdf.image(table_image_path, w=480, h=284.02)

        # Texto final
        pdf.set_xy((595 - 297) / 2, 2112)
        pdf.set_font("Arial", '', 12)
        pdf.set_text_color(1, 72, 255)
        pdf.cell(297, 16, "Em caso de dúvidas fale com o seu atendimento na BE180.", align='C')

        pdf.output(pdf_path)

        logger.info(f"PDF generated and saved to {pdf_path}")
        return pdf_path

    except Exception as e:
        logger.error(f"Error in generate_pdf: {e}")
        raise HTTPException(status_code=500, detail="Error generating PDF")


@app.post("/upload-campaign/")
async def upload_campaign_logo(file: UploadFile = File(...), background_names: str = Form(...)):
    try:
        # Gera um identificador único
        common_id = str(uuid.uuid4())

        # Cria o diretório para campanhas, se não existir
        os.makedirs('app/campaigns', exist_ok=True)
        campaign_image_path = os.path.join('app', 'campaigns', f"{common_id}_{file.filename}")

        # Salva o arquivo da campanha
        with open(campaign_image_path, "wb") as campaign_image:
            campaign_image.write(await file.read())

        background_names_list = background_names.split(",")  # Nomes dos backgrounds
        processed_paths = []

        # Processa cada imagem de background fornecida
        for background_name in background_names_list:
            background_image_path = os.path.join('app', 'images', background_name.strip())

            if not os.path.exists(background_image_path):
                raise HTTPException(status_code=404, detail=f"Background image {background_name} not found")

            # Determina o sufixo (ex.: "noite_3M" ou "15M")
            suffix = determine_suffix(background_name)
            processed_path = apply_logo(campaign_image_path, background_image_path, common_id, suffix)
            processed_paths.append(processed_path)

        # Gera um PDF com as imagens processadas
        pdf_path = generate_pdf(processed_paths, common_id)

        logger.info(f"File {file.filename} uploaded and processed with backgrounds {background_names}")
        return {"filename": file.filename, "processed_paths": processed_paths, "pdf_path": pdf_path}

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


@app.get("/pdf/{filename}")
async def get_pdf(filename: str):
    try:
        file_path = os.path.join('app', 'pdfs', filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path)
    except Exception as e:
        logger.error(f"Error in get_pdf: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving the PDF")

@app.get("/backgrounds/")
async def list_backgrounds():
    try:
        backgrounds_path = 'app/images'
        backgrounds = [f for f in os.listdir(backgrounds_path) if os.path.isfile(os.path.join(backgrounds_path, f))]
        return JSONResponse(content={"backgrounds": backgrounds})
    except Exception as e:
        logger.error(f"Error in list_backgrounds: {e}")
        raise HTTPException(status_code=500, detail="Error listing backgrounds")


@app.post("/register/")
async def register_user(email: str = Form(...), nome: str = Form(...), campanha: str = Form(...),
                        agencia: str = Form(...)):
    try:
        file_path = 'app/data/registrations.csv'

        # Certificar-se de que o diretório existe
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Verificar se o arquivo já existe para escrever o cabeçalho
        write_header = not os.path.exists(file_path)

        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(["Email", "Nome", "Campanha", "Agencia"])
            writer.writerow([email, nome, campanha, agencia])

        return {"message": "Registration successful"}
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail="Error registering user")

@app.get("/registrations/")
async def get_registrations():
    try:
        file_path = 'app/data/registrations.csv'
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="No registrations found")

        registrations = []
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                registrations.append(row)

        return registrations
    except Exception as e:
        logger.error(f"Error retrieving registrations: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving registrations")

@app.get("/images/{filename}")
async def get_image(filename: str):
    try:
        # Caminho para o diretório "images"
        file_path = os.path.join('app', 'images', filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path)
    except Exception as e:
        logger.error(f"Error in get_image: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving the image")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

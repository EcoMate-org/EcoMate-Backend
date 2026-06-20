from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from schemas.agent import ClassificationOutput
from services.classifier import agent_classifier
from pydantic import ValidationError

app = FastAPI(
    title="Item Classifying Agent",
    version="1.1"
)


@app.post(
    "/classify",
    response_model=ClassificationOutput
)
async def classify_image(
    image: UploadFile = File(...)
):
    MAX_FILE_SIZE = 5 * 1024 * 1024

    image_binary = await image.read()

    if len(image_binary) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds maximum size of 5MB."
        )

    try:
        result = await agent_classifier(
            image_binary,
            image.content_type
        )

        return result

    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=e.errors()
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
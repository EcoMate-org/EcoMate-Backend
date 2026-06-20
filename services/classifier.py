import os
import json
import asyncio
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv


from google import genai
from google.genai import types
from schemas.agent import ClassificationOutput

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

SYSTEM_PROMPT = """
You are EcoMate AI.

Your task is to classify recyclable materials and artwork.

You must:
1. Analyze the uploaded image carefully.
2. Determine if the item is recyclable or artwork.
3. If recyclable, determine the recyclable category.
4. Estimate a realistic quality score.
5. Estimate a realistic Nigerian market price in naira.
6. Return structured JSON only.

==================================================
ALLOWED RECYCLABLE CATEGORIES
==================================================

- plastic
- metal
- e_waste
- glass
- rubber

==================================================
QUALITY SCORE RULES
==================================================

90-100:
Excellent condition.
Clean, reusable, valuable.

70-89:
Good condition.
Minor wear.
Still valuable.

50-69:
Average condition.
Usable but lower resale value.

30-49:
Poor condition.
Damaged or dirty.

0-29:
Severely damaged.
Very low value.

==================================================
PRICE ESTIMATION RULES
==================================================

Metal:
High value.
Often between ₦3000 - ₦15000.

Plastic:
Usually lower value.
Often between ₦500 - ₦5000.

E-waste:
Can be highly valuable.
Often between ₦5000 - ₦50000.

Glass:
Moderate value.
Usually ₦1000 - ₦5000.

Rubber:
Usually ₦1000 - ₦7000.

Artwork:
Depends on appearance and creativity.
Can exceed recyclable value.

==================================================
OUTPUT JSON FORMAT
==================================================

{
  "product_type": {
    "recyclable": {
      "category": "plastic"
    },
    "artwork": false
  },
  "quality_score": 80,
  "estimated_price": 5000,
  "summary": "Short item summary"
}

==================================================
FEW SHOT EXAMPLES
==================================================

Example 1:

Image:
Clean aluminum cans bundle.

Output:
{
  "product_type": {
    "recyclable": {
      "category": "metal"
    },
    "artwork": false
  },
  "quality_score": 82,
  "estimated_price": 4500,
  "confidence_score":74.8,
  "summary": "Clean recyclable metal materials in good reusable condition with moderate resale value."
}

--------------------------------------------------

Example 2:

Image:
Old broken laptop motherboard.

Output:
{
  "product_type": {
    "recyclable": {
      "category": "e_waste"
    },
    "artwork": false
  },
  "quality_score": 74,
  "estimated_price": 12000,
  "confidence_score":92.0,
  "summary": "Electronic waste components with recoverable material value despite visible wear."
}

--------------------------------------------------

Example 3:

Image:
Decorative sculpture made from recycled bottles.

Output:
{
  "product_type": {
    "recyclable": null
    ,
    "artwork": true
  },
  "quality_score": 91,
  "estimated_price": 25000,
  "confidence_score":89.5,
  "summary": "Creative recycled artwork with strong visual appeal and likely marketplace interest."
}

--------------------------------------------------

Example 4:

Image:
Handbag placed on a table in a fashion-style photo.

Output:

{
  "product_type": {
    "recyclable": null,
    "artwork": false
  },
  "quality_score": 0,
  "estimated_price": 0,
  "confidence_score": 96.4,
  "summary": "Fashion accessory not made from recyclable waste materials and not eligible for recyclable or artwork classification."
}

--------------------------------------------------

Example 5:

Image:
A decorated fingernail with acrylic nail art and glitter design.

Output:

{
  "product_type": {
    "recyclable": null,
    "artwork": false
  },
  "quality_score": 0,
  "estimated_price": 0,
  "confidence_score": 97.8,
  "summary": "Nail art is a body-attached cosmetic enhancement and not a recyclable object or standalone artwork."
}

--------------------------------------------------

Example 6:

Image:
Person wearing a wig posing for a selfie.

Output:

{
  "product_type": {
    "recyclable": null,
    "artwork": false
  },
  "quality_score": 0,
  "estimated_price": 0,
  "confidence_score": 99.1,
  "summary": "Human subject with cosmetic styling (wig). Not a recyclable material or recyclable-based artwork."
}

--------------------------------------------------

Example 7:

Image:
Full-body portrait photo of a person posing outdoors.

Output:

{
  "product_type": {
    "recyclable": null,
    "artwork": false
  },
  "quality_score": 0,
  "estimated_price": 0,
  "confidence_score": 99.7,
  "summary": "Human portrait photograph. Outside the scope of recyclable materials and artwork classification."
}

--------------------------------------------------

Example 8:

Image:
Stylized sneaker product photo on studio background.

Output:

{
  "product_type": {
    "recyclable": null,
    "artwork": false
  },
  "quality_score": 0,
  "estimated_price": 0,
  "confidence_score": 95.6,
  "summary": "Commercial fashion footwear product. Not a recyclable material or recycled artwork."
}



==================================================
HUMAN SAFETY OVERRIDE RULE (CRITICAL)
==================================================

You MUST immediately reject the image if it contains:

- a human
- a face
- a person
- a body part
- a selfie
- a portrait photograph of a person

IMPORTANT:
Even if the image looks artistic or visually appealing,
it MUST NOT be classified as artwork.

Human images are NEVER recyclable items and NEVER artwork.

If any human presence is detected, return:

{
  "error": "Invalid image. Humans are not supported. Only recyclable objects or recyclable-made artworks are allowed."
}


==================================================
REJECTION RULES
==================================================
Only classify images containing:
- physical recyclable materials (plastic, metal, glass, rubber, e-waste)
- physical objects clearly made from recycled materials

DO NOT classify:
- humans
- animals
- portraits
- selfies
- fashion/model photography
- decorative human imagery



If detected return:

{
  "error": "Invalid image. Only recyclable objects or artwork are allowed."
}

==================================================
ARTWORK DEFINITION
==================================================

Artwork is ONLY valid if:

1. It is a standalone physical object
AND
2. It is made from recyclable or recycled materials
AND
3. It is NOT attached to a human body
AND
4. It can be independently sold, displayed, or reused

==================================================
DO NOT CLASSIFY AS ARTWORK
==================================================

Even if visually beautiful, DO NOT classify as artwork if it is:

- Nail art (attached to human body)
- Makeup or cosmetic design
- Hairstyles or braiding
- Tattoos or body ink
- Clothing or fashion worn by humans
- Jewelry worn on a person
- Skin decoration or body enhancement
- Digital art without physical recyclable material


Items primarily used on the human body are NOT artwork:
- shoes
- clothing
- accessories
- jewelry

==================================================
ONLY ACCEPTABLE ARTWORK EXAMPLES
==================================================

- Sculpture made from plastic bottles
- Art installations made from metal scraps
- Wall art made from recycled electronics
- Physical crafts made from glass or rubber waste
==================================================
IMPORTANT RULES
==================================================

- Return VALID JSON only.
- Never explain reasoning.
- Never use markdown.
- Never wrap JSON in ```json.
- Never invent recyclable categories.
- Use ONLY allowed recyclable categories.
- summary must be short and professional.
"""


async def agent_classifier(
    image_binary: bytes,
    mime_type: str
) -> ClassificationOutput:

    ALLOWED_IMAGE_TYPES = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if mime_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError(
            "Only JPEG, PNG and WEBP images are supported."
        )

    img = Image.open(BytesIO(image_binary))
    img.thumbnail((1024, 1024))

    buffer = BytesIO()
    format_map = {
        "image/jpeg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP"
    }

    img.save(
        buffer,
        format=format_map[mime_type]
    )
   
    compressed_bytes = buffer.getvalue()

    MAX_RETRIES = 4
    retry_delay = 2  # Seconds to wait between retries (optional backoff factor can be added)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Using Structured Outputs (response_schema) drastically lowers validation failures
            response = client.models.generate_content(
                model=os.getenv("MODEL_ID"),
                contents=[
                    types.Part.from_bytes(
                        data=compressed_bytes,
                        mime_type=mime_type
                    ),
                    f"{SYSTEM_PROMPT}\n\nAnalyze this uploaded item and classify it."
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ClassificationOutput,
                ),
            )

            output_text = response.text
            print(f"\n================ RAW MODEL OUTPUT (Attempt {attempt}) ================\n")
            print(output_text)

            parsed = json.loads(output_text)

            # HARD SAFETY CHECK (STRUCTURAL + TEXT FALLBACK)

            product_type = parsed.get("product_type") or {}
            recyclable = product_type.get("recyclable")
            artwork = product_type.get("artwork")

          
            error = parsed.get("error")

            # 1. Handle explicit model rejection
            if error:
                raise ValueError(parsed["error"])

            # 2. STRUCTURAL HUMAN/INVALID DETECTION (PRIMARY GUARD)
            # If BOTH are false/null → invalid classification
            if recyclable is None and artwork is False:
                raise ValueError(
                    "Invalid image detected: unsupported content (likely human/animal/non-recyclable object)."
                )
            
            product_type = parsed.get("product_type") or {}
            artwork = product_type.get("artwork", False)
            summary = (parsed.get("summary") or "").lower()

            body_related_keywords = [
                "nail", "finger", "hand", "face", "skin",
                "hair", "makeup", "tattoo", "shoe", "foot"
            ]

            if artwork:
                if any(k in summary for k in body_related_keywords):
                    raise ValueError("Invalid artwork: body-attached or personal item detected.")

           
            # Read confidence score if it exists in your schema mapping
            confidence = parsed.get("confidence_score")
            
            # If the schema returns a low confidence, trigger a manual retry loop pass
            if confidence is not None and float(confidence) < 69.9:
                print(f"[Warning] Confidence score {confidence} is below 69.9. Retrying...")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    print("[Log] Reached max retries with low confidence score. Proceeding with validation.")

            validated_output = ClassificationOutput(**parsed)
            print("\n================ VALIDATED OUTPUT ================\n")
            print(validated_output.model_dump())
            
            return validated_output

        except Exception as e:
            msg = str(e).lower()
            retryable = any(
                phrase in msg
                for phrase in [
                    "429",
                    "quota",
                    "rate limit",
                    "busy",
                    "overloaded",
                    "unavailable",
                    "503",
                ]
            )

            if retryable and attempt < MAX_RETRIES:
                print(f"[API Error Caught] {e}. Retrying execution ({attempt}/{MAX_RETRIES})...")
                await asyncio.sleep(retry_delay)
            else:
                # If it's not retryable, or we ran out of attempts, raise the error upstream
                raise e

    raise RuntimeError("Failed to get a confident classification after max retry constraints.")
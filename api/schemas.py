from pydantic import BaseModel, Field

class MushroomInput(BaseModel):
    cap_diameter: float = Field(..., alias="cap-diameter")
    cap_shape: str = Field(..., alias="cap-shape")
    cap_surface: str = Field(..., alias="cap-surface")
    cap_color: str = Field(..., alias="cap-color")
    does_bruise_or_bleed: str = Field(..., alias="does-bruise-or-bleed")
    gill_attachment: str = Field(..., alias="gill-attachment")
    gill_spacing: str = Field(..., alias="gill-spacing")
    gill_color: str = Field(..., alias="gill-color")
    stem_height: float = Field(..., alias="stem-height")
    stem_width: float = Field(..., alias="stem-width")
    stem_color: str = Field(..., alias="stem-color")
    has_ring: str = Field(..., alias="has-ring")
    ring_type: str = Field(..., alias="ring-type")
    habitat: str = Field(..., alias="habitat")
    season: str = Field(..., alias="season")

class PredictionOutput(BaseModel):
    prediction: int

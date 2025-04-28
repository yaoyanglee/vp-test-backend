from pydantic import BaseModel


class RawEntry(BaseModel):
    patient_id: str
    item_description: str
    item_code: str
    quantity: str
    uom: str
    dosage_instruction: str
    frequency: str
    dispensed_duration: str
    dose: str

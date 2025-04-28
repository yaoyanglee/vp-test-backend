from pydantic import BaseModel


class GetPatientDataRaw(BaseModel):
    patient_id: str

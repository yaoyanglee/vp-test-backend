from pydantic import BaseModel


class RxEntry(BaseModel):
    rx_id: str
    rx_url: str

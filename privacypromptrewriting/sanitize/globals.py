from langroid.utils.globals import GlobalState
from pyfpe_ff3 import FF3Cipher
from pydantic import BaseModel
from typing import Dict, List
from Crypto.Random import get_random_bytes

class SanitizationInfo(BaseModel):
    san: str
    orig: str
    category: str
    group_number: int

# Generate a random key
KEY = get_random_bytes(16)  # AES-128

class SanitizationState(GlobalState):
    info: List[SanitizationInfo] = []
    # categories: List[str] = ["Name", "Age", "Money", "CreditCard",
    #                          "Zipcode", "SSN", "Date",
    #                          "Email", "Phone Number", "Product ID", "Order ID"]
    categories: List[str] = ["Date", "Zipcode", "CreditCard"]
    N: Dict[str,float] = {'Money': 100000}
    rho: Dict[str,float] = {'Money': .05}
    money_units: int = 100
    epsilon:float = 1.0
    key:str = "EF4359D8D580AA4F7F036D6F04FC6A94"
    tweak:str = "D8E7920AFA330A73"
    sanitize_type: str = "fpe"
    aes_key = KEY

    def cipher_fn(self):
        return FF3Cipher(
            self.key,
            self.tweak,
            allow_small_domain=True,
            radix=10
        )

    def cipher_alphanum_fn(self):
        return FF3Cipher(
            self.key,
            self.tweak,
            allow_small_domain=True,
            radix=62
        )



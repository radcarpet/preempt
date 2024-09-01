"""
Agent to de-sanitize text, based on the global SanitizationState.san2orig mapping.
"""
from typing import Optional
import json
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.language_models.base import LLMConfig
from langroid.agent.chat_document import ChatDocument, ChatDocMetaData, Entity


from privacypromptrewriting.sanitize.globals import SanitizationState
from privacypromptrewriting.utils import *
from privacypromptrewriting.aes import decrypt

class DesanitizerAgentConfig(ChatAgentConfig):
    # any needed fields here
    # No need for LLM; the agent method will de-sanitize the text.
    llm: LLMConfig|None
    money_noise: bool = False
    sanitize_type: str = "fpe"


class DesanitizerAgent(ChatAgent):
    def __init__(self, config: DesanitizerAgentConfig):
        super().__init__(config)
        self.config: DesanitizerAgentConfig = config
        self.globals = SanitizationState.get_instance()

    def san2orig(self, san: str, grp: int) -> str:
        san2orig = {
            item.san: item.orig for item in self.globals.info
            if item.group_number == grp
        }
        return san2orig.get(san, "???")



    def agent_response(
        self,
        msg: Optional[str | ChatDocument] = None,
    ) -> Optional[ChatDocument]:
        """
        De-sanitize the text, using the global SanitizationState
        """

        message_str = msg.content if isinstance(msg, ChatDocument) else msg
        sanitized_items = self.globals.info

        # parse message_str which is a json dict with fields "answer" and "group"
        # and get the group number
        json_dict = json.loads(message_str)
        group_number = json_dict["group"]
        message_str = json_dict["answer"]

        # Look for sanitized strings, and only de-sanitize those.
        # Assumption: there are no "accidental" occurrences of sanitized strings
        # that did not come from the sanitization process.
        for item in sanitized_items:
            if not item.san in message_str:
                continue
            if self.globals.sanitize_type == "aes":
                replacement = decrypt(item.san, self.globals.aes_key)
            else:
                replacement = self._desanitize_category(
                    item.san,
                    item.category,
                    group_number
                )
            message_str = message_str.replace(item.san, replacement)

        return ChatDocument(
            content=message_str,
            metadata=ChatDocMetaData(
                source=Entity.AGENT,
                sender=Entity.AGENT,
                sender_name=self.config.name,
            ),
        )

    def _desanitize_category(self, value: str, category:str, grp: int) -> str:
        if self.globals.sanitize_type == "rand":
            return self.san2orig(value, grp)
        match category:
            #David: use self.config.money_noise to decide how to handle money
            case "Name" | "Money" | "Age" | "Date":
                if category == "Money" and not self.config.money_noise:
                    return format_align_digits(
                        self.globals.cipher_fn().decrypt(
                            value.replace("$","").replace(",","").replace(".","")
                        ),
                        value
                    )
                else:
                    return self.san2orig(value, grp)

            case "Zipcode":
                return self.globals.cipher_fn().decrypt(value)
            case "CreditCard":
                return format_align_digits(
                    self.globals.cipher_fn().decrypt(
                        value.replace("-","").replace(" ","")
                    ),
                    value
                )
            case "SSN":
                return format_align_digits(
                    self.globals.cipher_fn().decrypt(
                        value.replace("-","")
                    ),
                    value
                )
        
            case "Email":
                username, domain = split_email(value)
                old_username = self.globals.cipher_alphanum_fn().decrypt(username)
                return old_username + domain
        
            case "Phone Number":
                extracted_number = ''.join(char for char in value if char.isdigit())
                old_number = self.globals.cipher_fn().decrypt(extracted_number)
                return format_new_number(value, new_number)
        
            case "Product ID":
                product_id = ''.join(char for char in value if char.isalnum())
                decrypted_id = self.globals.cipher_alphanum_fn().decrypt(product_id)
                return separate_and_reintroduce(value, decrypted_id)
        
            case "Order ID":
                order_id = value
                return self.globals.cipher_alphanum_fn().decrypt(value)











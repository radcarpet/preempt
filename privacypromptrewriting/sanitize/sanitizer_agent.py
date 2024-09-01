from typing import List, Dict, Optional
import ast
import json
import langroid as lr

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.chat_document import ChatDocument
from langroid.utils.configuration import settings
from langroid.parsing.json import extract_top_level_json
from privacypromptrewriting.sanitize.globals import SanitizationState, SanitizationInfo
from privacypromptrewriting.utils import *
from privacypromptrewriting.sanitize.sanitize_tool import SanitizeTool
from privacypromptrewriting.sanitize.currency_utils import str2money, money2str
from pyfpe_ff3 import FF3Cipher
from privacypromptrewriting.utils import format_align_digits
from privacypromptrewriting.aes import encrypt
from privacypromptrewriting.rand import sanitize_cc, sanitize_zip, sanitize_date

import logging
logger = logging.getLogger(__name__)



class SanitizerAgentConfig(ChatAgentConfig):
    # any needed values here
    llm = lr.language_models.OpenAIGPTConfig(
        timeout=45,
        chat_model=lr.language_models.openai_gpt.OpenAIChatModel.GPT4_TURBO,
    )
    money_noise: bool = False




class SanitizerAgent(ChatAgent):
    def __init__(self, config: SanitizerAgentConfig):
        super().__init__(config)
        self.globals = SanitizationState.get_instance()
        self.config: SanitizerAgentConfig = config
        self.system_message = f"""
        You are a Privacy expert, experienced in detecting sensitive information in 
        text passages. You will be given a passage that may contain sensitive 
        information, and your job is to detect sensitive pieces of information and 
        highlight them using the `sanitize` tool/function.
        If there is no sensitive information, return an empty `sensitive_item` list, 
        and do not say anything else. 
        You must only focus on these sensitive categories: 
        {",".join(self.globals.categories)}
        """

        self.enable_message(SanitizeTool) # .create(self.globals.categories))
        self.sensitive_text: Optional[str] = None
        # look-up dict of sensitive -> sanitized mapping.
        self.orig2san: Optional[Dict[str,str]] = None
        # reverse look-up dict of sanitized -> sensitive mapping.
        self.san2orig: Optional[Dict[str,str]] = None


    def llm_response(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:

        assert message is not None, "No message specified"
        message_str = message.content if isinstance(message, ChatDocument) else message
        json_str = extract_top_level_json(message_str)[0]
        numbered_text = json.loads(json_str)
        self.sensitive_text = numbered_text["content"]
        self.group_number = numbered_text["number"]

        self.sensitive_text = message_str
        if isinstance(message, ChatDocument):
            message.content = self.sensitive_text
        else:
            message = self.sensitive_text
        message = f"""
        SENSITIVE TEXT:
        
        {message}
        """
        return super().llm_response(message)

    async def llm_response_async(
            self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:

        assert message is not None, "No message specified"
        message_str = message.content if isinstance(message, ChatDocument) else message
        json_str = extract_top_level_json(message_str)[0]
        numbered_text = json.loads(json_str)
        self.sensitive_text = numbered_text["content"]
        self.group_number = numbered_text["number"]

        if isinstance(message, ChatDocument):
            message.content = self.sensitive_text
        else:
            message = self.sensitive_text
        message = f"""
        SENSITIVE TEXT:
        
        {message}
        """
        return await super().llm_response_async(message)


    def _sanitize_category(self, value: str, category:str) -> str:
        match category:
            case "Name":
                return rng.generate(
                    descent=rng.Descent.ENGLISH,
                    sex=rng.Sex.MALE,
                    limit=1
                )[0]
            case "Age":
                age = M_epsilon(
                    int(value),10,99, self.globals.epislon
                )
                return age  #str(np.random.choice(100,1,age_probs)[0])
            case "Money":
                #David: use self.config.money_noise to decide whether to add noise
                if self.config.money_noise:
                    N = self.globals.N
                    rho = self.globals.rho
                    money_units = self.globals.money_units
                    v = str2money(value)
                    return money2str(
                        M_epsilon(
                            int(float(v)),
                            max(0,
                                int(float(v)*(1-rho['Money']))
                            ),
                            min(
                                int(float(v)*(1+rho['Money'])),
                                v*2,
                            ),
                            self.globals.epsilon,
                        ),       
                        value
                    )
                else:
                    return format_align_digits(
                      self.globals.cipher_fn().encrypt(
                          value.replace("$","").replace(",","").replace(".","")
                        ),
                        value
                    )

            case "Zipcode":
                if self.globals.sanitize_type == "rand":
                    return sanitize_zip(value)
                else:
                    return self.globals.cipher_fn().encrypt(value)
            case "CreditCard":
                if self.globals.sanitize_type == "rand":
                    return sanitize_cc(value)
                return format_align_digits(
                    self.globals.cipher_fn().encrypt(
                        value.replace("-","").replace(" ","")
                    ),
                    value
                )
            case "SSN":
                return format_align_digits(
                    self.globals.cipher_fn().encrypt(
                        value.replace("-","")
                    ),
                    value
                )
            case "Date":
                if self.globals.sanitize_type == "rand":
                    return sanitize_date(value)
                date = value
                numeric_date = convert_date_to_numeric(date)
                numeric_date = M_epsilon(numeric_date, -365, 730, self.globals.epsilon) % 365
                return convert_day_of_year_to_date(numeric_date)
        
            case "Email":
                username, domain = split_email(value)
                new_username = self.globals.cipher_alphanum_fn().encrypt(username)
                return new_username + domain
        
            case "Phone Number":
                extracted_number = ''.join(char for char in value if char.isdigit())
                new_number = self.globals.cipher_fn().encrypt(extracted_number)
                return format_new_number(value, new_number)
        
            case "Product ID":
                product_id = ''.join(char for char in value if char.isalnum())
                encrypted_id = self.globals.cipher_alphanum_fn().encrypt(product_id)
                return separate_and_reintroduce(value, encrypted_id)
        
            case "Order ID":
                order_id = value
                return self.globals.cipher_alphanum_fn().encrypt(value)
                        



    def sanitize(self, msg: SanitizeTool) -> str:
        if settings.debug:
            # show each sensitive item and its category
            logger.warning("Sensitive items:")
            for item in msg.sensitive_items:
                logger.warning(f"{item.value} ({item.category})")
        # dict of sensitive value -> category
        if self.globals.sanitize_type == "aes":
            self.orig2san = {
                item.value: encrypt(item.value, self.globals.aes_key)
                for item in msg.sensitive_items
            }
        elif self.globals.sanitize_type == "rand":
            self.orig2san = {
                item.value: self._sanitize_category(item.value, item.category)
                for item in msg.sensitive_items
            }
        else:
            self.orig2san = {
                item.value: self._sanitize_category(item.value, item.category)
                for item in msg.sensitive_items
            }
        self.san2orig = {
            v: k for k, v in self.orig2san.items()
        }
        # save the san, orig, category info in global SanitizationState
        # for the desanitizer agent to use, to de-sanitize the sanitized answer.
        for item in msg.sensitive_items:
            info = SanitizationState.get_value("info")
            SanitizationState.set_values(
                info=info + [
                  SanitizationInfo(
                      # to separate from other agents, this prevents collisions
                      name = self.config.name,
                      orig = item.value,
                      san = self.orig2san[item.value],
                      category = item.category,
                      group_number = self.group_number
                  )
                ]
            )

        sanitized_text = self.sensitive_text
        for value, new in self.orig2san.items():
            sanitized_text = sanitized_text.replace(value, new)
        return "DONE\n" + sanitized_text

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        # if msg has no tool, there must have been no sensitive info,
        # so just return DONE and the original text
        if isinstance(msg, ChatDocument) and msg.metadata.sender == lr.mytypes.Entity.LLM:
            return f"""
            You either forgot to use the `sanitize` tool/function, or there was 
            an error in the format of the `sanitize` tool/function. Try again.
            Here is the original text:
            
            SENSITIVE TEXT:

            {self.sensitive_text}                        
            """
        return None












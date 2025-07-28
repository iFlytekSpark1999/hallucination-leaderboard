import os
from typing import Literal

from openai import OpenAI

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

"""
Unique Notes:

iFlytek_Spark
"""

#TODO: Rename iFlytek_Spark

COMPANY = "iflytek_spark" # Previously iflytek spark
class iFlytek_SparkConfig(BasicLLMConfig):
    """Extended config for iflytek-spark-specific properties"""
    company: Literal["iflytek_spark"]
    model_name: Literal[
        "4.0ultra",  #fast
        "x1"    #slow
    ] # Only model names manually added to this list are supported.
    date_code: str = "" # do we need date code for iFlytek_Spark?
    execution_mode: Literal["api"] = "api" # Is iflytek only API based?
    endpoint: Literal["chat", "response"] = "chat" # The endpoint to use for the OpenAI API. Chat means chat.completions.create(), response means responses.create().
    thinking_tokens: bool = None
    enable_thinking: bool = None

class iFlytek_SparkSummary(BasicSummary):
    endpoint: Literal["chat", "response"] | None = None # No default. Needs to be set from from LLM config.
    endpoint: Literal["chat", "response"] | None = None # No default. Needs to be set from from LLM config.
    enable_thinking: bool | None = None

    class Config:
        extra = "ignore" # fields that are not in OpenAISummary nor BasicSummary are ignored.

class iFlytek_SparkLLM(AbstractLLM):
    """
    Class for models from iflytek
    """
    # In which way to run the model via web api. Empty dict means not supported for web api execution. 
    client_mode_group = {
        "4.0ultra": {
            "chat": 1
        },
        "x1": {
            "chat": 2
        }
    }

    # In which way to run the model on local GPU. Empty dict means not supported for local GPU execution
    local_mode_group = {}

    def __init__(self, config: iFlytek_SparkConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.enable_thinking = config.enable_thinking

    def summarize(self, prepared_text: str) -> str:
        summary = SummaryError.EMPTY_SUMMARY
        if self.client1 and  self.client2:
            match self.client_mode_group[self.model_name][self.endpoint]:
                case 1: # Default
                    completion = self.client1.chat.completions.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        messages=[
                            {"role": "user", "content": prepared_text}],
                        )
                    summary = completion.choices[0].message.content
                case 2: # Reasoning model
                    completion = self.client2.chat.completions.create(
                        model=self.model_fullname,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        extra_body = {"enable_thinking": self.enable_thinking},
                        messages=[
                            {"role": "user", "content": prepared_text}],
                        )
                    summary = completion.choices[0].message.content
        elif self.local_model: 
            pass
        else:
            raise Exception(
                ModelInstantiationError.MISSING_SETUP.format(
                    class_name=self.__class__.__name__
                )
            )
        return summary

    def setup(self):
        if self.execution_mode == "api":
            if self.model_name in self.client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, (
                    f"{COMPANY} API key not found in environment variable "
                    f"{COMPANY.upper()}_API_KEY"
                )
                self.client1 = OpenAI(
                    api_key=api_key, 
                    base_url="https://spark-api-open.xf-yun.com/v1",
                )
                
                self.client2 = OpenAI(
                    api_key=api_key, 
                    base_url="https://spark-api-open.xf-yun.com/v2",
                )
            else:
                raise Exception(
                    ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                        model_name=self.model_name,
                        company=self.company,
                        execution_mode=self.execution_mode
                    )
                )
        elif self.execution_mode == "local":
            pass

    def teardown(self):
        if self.client:
            self.close_client()
        elif self.local_model:
            # self.default_local_model_teardown()
            pass

    def close_client(self):
        pass
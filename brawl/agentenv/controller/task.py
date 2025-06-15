# task.py
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, TypedDict, Literal, List

import openai
from transformers import PreTrainedTokenizerBase, GenerationConfig

from brawl.agentenv.controller import BaseEnvClient


class ConversationMessage(TypedDict):
    from_: Literal["human", "gpt"]
    value: str
    loss: Optional[bool]


@dataclass
class ExperienceOutput:
    conversation: List[ConversationMessage]
    reward: float
    text: Optional[str]
    seq_ids: Optional[List[int]]
    attention_mask: Optional[List[int]]
    action_mask: Optional[List[int]]


class BaseTask:
    env_client_cls: Callable[[Any], BaseEnvClient] = None
    env_name: str = None

    def __init__(
        self,
        client_args: Mapping[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        *,
        vllm_base_url: str = "http://localhost:8002/v1",
        vllm_api_key: str = "dummy-api-key",
        vllm_model_name: str = "agent-llm",
    ) -> None:
        """
        Initializes the Task Object to use a vLLM OpenAI-compatible server
        """

        if self.env_client_cls is None or self.env_name is None:
            raise NotImplementedError("env_client_cls or env_name is not implemented")

        self.env_clients = [self.env_client_cls(**client_args)]
        self.len = len(self.env_clients[0])

        self.tokenizer = tokenizer
        self.vllm_model_name = vllm_model_name
        self.openai_client = openai.OpenAI(
            base_url=vllm_base_url,
            api_key=vllm_api_key,
        )


    def _format_conversation_for_openai(
        self, conversation: list[ConversationMessage]
    ) -> list[dict[str, str]]:
        """Convert internal conversation format → OpenAI format."""
        messages: list[dict[str, str]] = []
        for msg in conversation:
            role = "user" if msg["from_"] == "human" else "assistant"
            messages.append({"role": role, "content": msg["value"]})
        return messages

    def _reconstruct_tokenized_info(
        self, conversation: list[ConversationMessage]
    ) -> tuple[str, list[int], list[int]]:
        """
        Prepare the conversation in a single flattened text / id / mask triple.
        """
        full_text = ""
        full_input_ids: list[int] = []
        full_action_mask: list[int] = []

        for i, message in enumerate(conversation):
            if message["from_"] == "human":
                # prepend BOS once
                if i == 0 and self.tokenizer.bos_token:
                    bos = self.tokenizer.bos_token
                    bos_ids = self.tokenizer.encode(bos, add_special_tokens=False)
                    full_text += bos
                    full_input_ids += bos_ids
                    full_action_mask += [0] * len(bos_ids)

                current_text = f" [INST] {message['value']} [/INST]"
                current_ids = self.tokenizer.encode(
                    current_text, add_special_tokens=False
                )

                full_text += current_text
                full_input_ids += current_ids
                full_action_mask += [0] * len(current_ids)

            elif message["from_"] == "gpt":
                prefix = " " if full_text else ""
                current_text = f"{prefix}{message['value']}{self.tokenizer.eos_token}"
                current_ids = self.tokenizer.encode(
                    current_text, add_special_tokens=False
                )

                full_text += current_text
                full_input_ids += current_ids
                mask_val = 1 if message.get("loss") else 0
                full_action_mask += [mask_val] * len(current_ids)

        return full_text, full_input_ids, full_action_mask


    def _generate_experience_one(
        self,
        client: BaseEnvClient,
        idx: int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> ExperienceOutput:

        client.reset(idx)
        reward = 0.0
        done = False
        state = client.observe()

        conversation: list[ConversationMessage] = list(client.conversation_start)
        conversation.append({"from_": "human", "value": state, "loss": None})
        rounds = 0

        temperature = (
            generation_config.temperature
            if generation_config and generation_config.temperature is not None
            else 0.7
        )
        max_tokens = (
            generation_config.max_new_tokens
            if generation_config and generation_config.max_new_tokens is not None
            else 256
        )

        while not done:
            format_messages = self._format_conversation_for_openai(conversation)

            api_response = self.openai_client.chat.completions.create(
                model=self.vllm_model_name,
                messages=format_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            generated_text = api_response.choices[0].message.content or ""

            conversation.append(
                {"from_": "gpt", "value": generated_text, "loss": False}
            )

            step_output = client.step(generated_text)
            state, reward, done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )

            if not done:
                conversation.append({"from_": "human", "value": state, "loss": None})

            rounds += 1
            if max_rounds is not None and rounds >= max_rounds:
                break
            if len(conversation) > 50:
                print("Warning: max conversation turns reached")
                break

        return ExperienceOutput(
            conversation=conversation,
            reward=reward,
            text=None,
            seq_ids=None,
            attention_mask=None,
            action_mask=None,
        )

    def _generate_experience_batch(
        self,
        idxs: Sequence[int],
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput]:

        env_client = self.env_clients[0]
        return [
            self._generate_experience_one(
                client=env_client,
                idx=idx,
                generation_config=generation_config,
                max_rounds=max_rounds,
            )
            for idx in idxs
        ]


    def generate_experience(
        self,
        idxs: Sequence[int] | int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput] | ExperienceOutput:

        if not self.tokenizer:
            raise ValueError("Tokenizer must be provided to use vLLM")

        single = isinstance(idxs, int)
        idx_list: list[int] = [idxs] if single else list(idxs)

        results = self._generate_experience_batch(
            idxs=idx_list,
            generation_config=generation_config,
            max_rounds=max_rounds,
        )
        return results[0] if single else results


    def __len__(self) -> int:
        return self.len

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, TypedDict, Literal, List

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
    env_client_cls = None   # override in subclass
    env_name: str | None = None

    def __init__(
        self,
        client_args: Mapping[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        *,
        vllm_base_url: str = "http://localhost:8002/v1",
        vllm_api_key: str = "dummy-api-key",
        vllm_model_name: str = "agent-llm",
    ) -> None:

        if self.env_client_cls is None or self.env_name is None:
            raise NotImplementedError("env_client_cls or env_name not set")

        # ― single env client (extend if you need a pool)
        self.env_clients = [self.env_client_cls(**client_args)]
        self.len = len(self.env_clients[0])

        self.tokenizer = tokenizer
        self.vllm_model_name = vllm_model_name
        self.openai_client = openai.OpenAI(
            base_url=vllm_base_url,
            api_key=vllm_api_key,
        )

    # ---------------- internal helpers -----------------
    def _format_conversation_for_openai(
        self, conversation: list[ConversationMessage]
    ) -> list[dict[str, str]]:
        msgs = []
        for msg in conversation:
            role = "user" if msg["from_"] == "human" else "assistant"
            msgs.append({"role": role, "content": msg["value"]})
        return msgs

    def _reconstruct_tokenized_info(  # kept for future use
        self, conversation: list[ConversationMessage]
    ) -> tuple[str, list[int], list[int]]:
        full_text, full_ids, full_mask = "", [], []
        for i, message in enumerate(conversation):
            if message["from_"] == "human":
                if i == 0 and self.tokenizer.bos_token:
                    bos_ids = self.tokenizer.encode(
                        self.tokenizer.bos_token, add_special_tokens=False
                    )
                    full_text += self.tokenizer.bos_token
                    full_ids += bos_ids
                    full_mask += [0] * len(bos_ids)

                chunk = f" [INST] {message['value']} [/INST]"
                ids = self.tokenizer.encode(chunk, add_special_tokens=False)
                full_text += chunk
                full_ids += ids
                full_mask += [0] * len(ids)

            else:  # gpt
                chunk = f" {message['value']}{self.tokenizer.eos_token}"
                ids = self.tokenizer.encode(chunk, add_special_tokens=False)
                full_text += chunk
                full_ids += ids
                full_mask += [1 if message.get('loss') else 0] * len(ids)

        return full_text, full_ids, full_mask


    def generate_experience(
        self,
        idxs: Sequence[int] | int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput] | ExperienceOutput:

        single = isinstance(idxs, int)
        idx_list = [idxs] if single else list(idxs)
        out = [self._generate_experience_one(
            client=self.env_clients[0],
            idx=i,
            generation_config=generation_config,
            max_rounds=max_rounds,
        ) for i in idx_list]
        return out[0] if single else out


    def _generate_experience_one(
        self,
        client: BaseEnvClient,
        idx: int,
        generation_config: Optional[GenerationConfig],
        max_rounds: Optional[int],
    ) -> ExperienceOutput:

        client.reset(idx)
        reward, done, rounds = 0.0, False, 0
        state = client.observe()

        conv: list[ConversationMessage] = list(client.conversation_start)
        conv.append({"from_": "human", "value": state, "loss": None})

        temp = generation_config.temperature if generation_config else 0.7
        max_tokens = generation_config.max_new_tokens if generation_config else 256

        while not done:
            api_resp = self.openai_client.chat.completions.create(
                model=self.vllm_model_name,
                messages=self._format_conversation_for_openai(conv),
                temperature=temp,
                max_tokens=max_tokens,
                stream=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            generated = api_resp.choices[0].message.content or ""
            conv.append({"from_": "gpt", "value": generated, "loss": False})

            step = client.step(generated)
            state, reward, done = step.state, step.reward, step.done
            if not done:
                conv.append({"from_": "human", "value": state, "loss": None})

            rounds += 1
            if max_rounds and rounds >= max_rounds:
                break
            if len(conv) > 50:
                print("Warning: max conversation turns reached")
                break

        return ExperienceOutput(
            conversation=conv,
            reward=reward,
            text=None,
            seq_ids=None,
            attention_mask=None,
            action_mask=None,
        )

    def __len__(self) -> int:
        return self.len

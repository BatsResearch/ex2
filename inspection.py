import json
from dataclasses import dataclass
from pathlib import Path
from string import Template

import dotenv
import tyro
from openai import OpenAI
from tqdm import tqdm

import utils

dotenv.load_dotenv()


@dataclass
class ScriptArguments:
    input_filepath: str
    """Path to the file with descriptions"""
    dataset_name: str
    """Name of the dataset"""
    output_filepath: str
    """Where to save the API responses"""


args: ScriptArguments = tyro.cli(ScriptArguments)


def get_message_list(class_name, input_text):
    system_message = dict(
        role="system",
        content=f"""You are a knowledgeable teacher. Answer the questions in JSON format.""",
    )
    user_msg_template = Template(
        "You want to explain what a ${cn} is to your students. Does the following text snippet mention any specific details about ${cn} that increases your students' knowledge about ${cn}? Answer yes or no. Provide an explanation for your answer.\n\nText snippet: ${it}"
    )

    message_list = list()
    message_list.append(system_message)
    message_list.append(
        dict(
            role="user",
            content=user_msg_template.substitute(
                cn="tench", it="A photo of a tench, with dark green color."
            ),
        )
    )
    message_list.append(
        {
            "role": "assistant",
            "content": json.dumps(
                dict(
                    explanation="It teaches the students about the color of a tench.",
                    increases_knowledge="Yes",
                ),
                indent=0,
            ),
        }
    )
    message_list.append(
        dict(
            role="user",
            content=user_msg_template.substitute(cn=class_name, it=input_text),
        )
    )
    return message_list


class_names = utils.load_classnames(args.dataset_name)
with open(args.input_filepath, "r") as f:
    descriptions = json.load(f)


gen_kwargs = dict(
    model="gpt-3.5-turbo-1106",
    temperature=0.1,
    top_p=0.1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={"type": "json_object"},
    max_tokens=128,
)


client = OpenAI()

response_list = list()
for cls_name, cls_descs in tqdm(zip(class_names, descriptions)):
    response_list.append([])
    for desc in cls_descs:
        response = client.chat.completions.create(
            messages=get_message_list(class_name=cls_name, input_text=desc),
            **gen_kwargs,
        )
        response_json_str = response.model_dump_json()
        response_list[-1].append(response_json_str)

output_filepath = Path(args.output_filepath)
output_filepath.parent.mkdir(exist_ok=True, parents=True)
with open(output_filepath, "w") as f:
    json.dump(response_list, f)

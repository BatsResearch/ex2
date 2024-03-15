import json
from pathlib import Path

import hf_clip_wrapper
import open_clip_wrapper


def load_json(p):
    with open(p, "r") as f:
        obj = json.load(f)
    return obj


def get_vlm_class(vlm_name):
    if vlm_name.lower().startswith(open_clip_wrapper.AMD_OPEN_CLIP_PREFIX.lower()):
        return open_clip_wrapper.OpenCLIPWrapper
    return hf_clip_wrapper.HFCLIPWrapper


def load_classnames(dataset):
    def default_processor(cls):
        return cls.replace("_", " ")

    def fgvc_aircraft_processor(x):
        return x.replace("_", " ") + " aircraft"

    def stanford_dogs_processor(x):
        if "dog" in x:
            return x.replace("_", " ")
        else:
            return x.replace("_", " ") + " dog"

    clsname_path = Path(__file__).parent.joinpath(
        f"data_files/{dataset}_class_names.json"
    )
    assert clsname_path.exists()
    classnames = load_json(clsname_path)

    if dataset == "fgvc_aircraft":
        process_fn = fgvc_aircraft_processor
    elif dataset == "stanford_dogs":
        process_fn = stanford_dogs_processor
    else:
        process_fn = default_processor
    classnames = [process_fn(item) for item in classnames]
    return classnames


def load_query_templates():
    template_file = Path(__file__).parent.joinpath(f"data_files/mult_template_v1.json")
    assert template_file.exists()
    with template_file.open("r") as f:
        templates = json.load(f)
    return templates


def single_class_text_dataset(dataset_name):
    sample_formatter = lambda x: f"{x}\n\n"
    class_names = load_classnames(dataset_name)

    template_file = Path(__file__).parent.joinpath(f"data_files/mult_template_v1.json")
    assert template_file.exists()
    with template_file.open("r") as f:
        query_templates = json.load(f)

    query_list = list()
    query_to_metadata_map = dict()
    for class_id, class_name in enumerate(class_names):
        for q_template in query_templates:
            q_ = sample_formatter(q_template.format(class_name))
            query_list.append(q_)
            query_to_metadata_map[q_] = [class_id]

    eval_queries = list()
    for class_name in class_names:
        eval_queries.append(
            [sample_formatter(t.format(class_name)) for t in query_templates]
        )
    return query_list, query_to_metadata_map, eval_queries

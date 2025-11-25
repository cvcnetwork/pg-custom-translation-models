import os
import time
from pydantic import BaseModel
import yaml
import sys
import json
from typing import Annotated

from predictionguard import PredictionGuard
import pandas as pd
from comet import download_model, load_from_checkpoint
import deepl
from fastapi import FastAPI, Header, HTTPException
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import uuid
import traceback
from openai import OpenAI
import munch
import huggingface_hub
import requests


#--------------------------#
#         Config           #
#--------------------------#

ymlcfg = yaml.safe_load(open(os.path.join(sys.path[0],  'config.yml')))
initial_cfg = munch.munchify(ymlcfg)

app = FastAPI()

# Hugging Face login
huggingface_hub.login(token=initial_cfg.huggingface.token)
TOKENIZERS_PARALLELISM = initial_cfg.huggingface.tokenizers_parallelism
os.environ['TOKENIZERS_PARALLELISM'] = str(TOKENIZERS_PARALLELISM)

# Get a list of all supported languages
def get_supported_languages(cfg: dict):
    supported_languages = []
    for m in cfg.engines.keys():
        if m == "predictionguard" or m == "custom":
            for m_inner in cfg.engines[m].models:
                for l in cfg.engines[m].models[m_inner].languages:
                    supported_languages.append(l)
        else:
            for l in cfg.engines[m].languages:
                supported_languages.append(l)
    return supported_languages

# Get a list of supported models
def get_supported_models(cfg: dict):
    supported_models = []
    for m in cfg.engines.keys():
        if m == "predictionguard" or m == "custom":
            for m_inner in cfg.engines[m].models:
                supported_models.append(m + "__" + m_inner)
        else:
            supported_models.append(m)
    return supported_models


#-------------------------#
# LLM Translation Prompt  #
#-------------------------#

trans_prompt="""Translate the following {source_language} text to {target_language}. Only respond with the translation and no other text. Don't add, remove, or modify any information when translating.

{source_language} text: {input}

{target_language} translation:"""


#-------------------------#
# ISO Code Language Data  #
#-------------------------#

# Download the data
headers = {'User-Agent': 'Mozilla/5.0'}
url = 'https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3.tab'
r = requests.get(url, headers=headers)
with open('iso-639-3.tab', 'wb') as fh:
    fh.write(r.content)

# Read it into memory
iso = pd.read_csv('iso-639-3.tab', sep='\t')


#----------------------#
# COMET Quality Score  #
#----------------------#

# Download the COMET model
model_path = download_model(initial_cfg.comet.model)
comet_model = load_from_checkpoint(model_path)

# Define the input and output models for COMET scoring
class QAInput(BaseModel):
    source: str
    translation: str

class QAOutput(BaseModel):
    score: float

# Function to get quality scores
def get_quality_score(input: QAInput):
    data = [{
        "src": input.source,
        "mt": input.translation,
    }]
    model_output = comet_model.predict(data, batch_size=8, gpus=0, num_workers=1)
    return QAOutput(score=model_output.system_score)


#----------------------#
# MT APIs/ Models      #
#----------------------#

if "deepl" in initial_cfg.engines.keys():

    # Create a map of language codes to deepl languages
    deepl_languages = {
        "ara": "AR",
        "bul": "BG",
        "cmn": "ZH",
        "ces": "CS",
        "dan": "DA",
        "nld": "NL",
        "eng": "EN-GB",
        "est": "ET",
        "fin": "FI",
        "fra": "FR",
        "deu": "DE",
        "hun": "HU",
        "ind": "ID",
        "ita": "IT",
        "jpn": "JA",
        "kor": "KO",
        "lav": "LV",
        "lit": "LT",
        "ell": "EL",
        "nor": "NB",
        "pol": "PL",
        "por": "PT-BR",
        "ron": "RO",
        "rus": "RU",
        "slk": "SK",
        "slv": "SL",
        "spa": "ES",
        "swe": "SV",
        "tur": "TR",
        "ukr": "UK"
    }

def deepl_translation(text, target_language, cfg):

    # Process target language code
    target_language = deepl_languages[target_language]

    # Initialize the deepl translator
    deepl_translator = deepl.Translator(auth_key=cfg.engines.deepl.api_key)

    # Get the translation
    response = deepl_translator.translate_text(text, target_lang=target_language).text

    # Process the response
    if response is not None and response.strip():
        qa_input = QAInput(source=text, translation=response)
        score = get_quality_score(qa_input).score
        return {
            "translation": response, 
            "score": score, 
            "model": "deepl", 
            "status": "success"
        }
    else:
        return {
            "translation": "", 
            "score": -100, 
            "model": "deepl", 
            "status": "error: could not get translation"
        }


def pg_openai_translation(text, source_language, target_language, model, cfg):

    # Initialize the client
    if "gpt" in model:
        client = OpenAI(api_key=cfg.engines.openai.api_key)
    else:
        DEFAULT_URL = "https://api.predictionguard.com"
        predictionguard_engine = cfg.engines.predictionguard
        if 'url' not in predictionguard_engine.keys():
            url = DEFAULT_URL
        else:
            url = predictionguard_engine.url
        client = PredictionGuard(api_key=cfg.engines.predictionguard.api_key, url=url)

    # Call the API
    result = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user", 
            "content": trans_prompt.format(
                input=text, 
                source_language=source_language, 
                target_language=target_language
            )
        }],
        temperature=0.1
    )

    # Process the response
    response_message = result['choices'][0]['message']['content'].strip().split('\n')[0]
    if response_message:
        qa_input = QAInput(source=text, translation=response_message)
        score = get_quality_score(qa_input).score
        return {
            "translation": response_message, 
            "score": score, 
            "model": "openai", 
            "status": "success"}
    else:
        return {
            "translation": "", 
            "score": -100, "model": 
            "openai", "status": 
            "error: could not get translation"
        }
    

def custom_translation(text, source_language, target_language, model, cfg):

    # TODO: Make source language in the JSON body optional

    # Call the API
    headers = {'x-api-key': cfg.custom.models['model'].api_key}
    url = cfg.custom.models[model].url
    response = requests.post(
        url, 
        json={
            'model': model,
            'text': text, 
            'source_language': source_language, 
            'target_language': target_language
        }, 
        headers=headers)
    response = response.json()

    # Process the response
    if 'translation' in response.keys() and len(response['translation']) > 0:
        qa_input = QAInput(source=text, translation=response['translation'])
        score = get_quality_score(qa_input).score
        return {
            "translation": response['translation'], 
            "score": score, 
            "model": "custom",
            "status": "success"}
    else:
        return {
            "translation": "", 
            "score": -100, 
            "model": "custom", 
            "status": "error: could not get translation"
        }
        

#-----------------------------------------#
# Concurrent Translation functionality    #
#-----------------------------------------#

def translate_and_score(text, source_language_iso639, target_language_iso639, cfg):

    translation_results = []
    best_translation = None
    best_score = -1
    best_model = ""

    created_timestamp = int(time.time())
    unique_id = "translation-" + str(uuid.uuid4()).replace("-", "")

    # filter supported models based on language codes
    supported_models = get_supported_models(cfg)
    supported_models_filtered = []
    for model in supported_models:
        if "predictionguard" in model or "custom" in model:
            engine_type = model.split('__')[0]
            pg_langs = cfg.engines[engine_type]['models'][model.split('__')[-1]].languages
            if target_language_iso639 in pg_langs and source_language_iso639 in pg_langs:
                supported_models_filtered.append(model)
        else:
            other_langs = cfg.engines[model].languages
            if target_language_iso639 in other_langs and source_language_iso639 in other_langs:
                supported_models_filtered.append(model)

    def process_translation(model, cfg):
        try:
            if model == "deepl":
                result = deepl_translation(text, target_language_iso639, cfg)
            elif "predictionguard" in model:
                result = pg_openai_translation(
                    text, 
                    source_language_iso639, 
                    target_language_iso639, 
                    model.split('__')[-1],
                    cfg
                )
            elif model == "openai":
                result = pg_openai_translation(
                    text, 
                    source_language_iso639, 
                    target_language_iso639, 
                    cfg.engines.openai.model,
                    cfg
                )
            elif "custom" in model:
                result = custom_translation(
                    text, 
                    source_language_iso639, 
                    target_language_iso639, 
                    model.split('__')[-1],
                    cfg
                )
            else:
                raise ValueError(f"Unsupported model: {model}")
            return result
        except Exception as e:
            print(f"Error translating with {model}: {e}")
            traceback.print_exc()
            return {
                "score": 0,
                "translation": "",
                "model": model,
                "status": f"error: {str(e)}"
            }

    with ThreadPoolExecutor(max_workers=len(supported_models_filtered)) as executor:
        futures = [executor.submit(process_translation, model, cfg) for model in supported_models]
        for future in futures:
            result = future.result()
            translation_results.append(result)
            if result["status"] == "success" and result["score"] > best_score:
                best_translation = result["translation"]
                best_score = result["score"]
                best_model = result["model"]

    output = {
        "translations": translation_results,
        "best_translation": best_translation if best_translation else "We don't support the requested language pair",
        "best_score": best_score,
        "best_translation_model": best_model,
        "created": created_timestamp,
        "id": unique_id,
        "object": "translation"
    }

    return output


#---------------------#
# FastAPI app         #
#---------------------#

@app.get("/")
def read_root():
    return {"status": "healthy"}


class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str


def is_valid_language(language_code, cfg):
    return language_code in get_supported_languages(cfg)


@app.post("/translate")
def update_item(
        req: TranslateRequest,
        workspace_config: Annotated[str | None, Header()] = None
):
    if workspace_config != None:
        cfg = munch.munchify(json.loads(str(workspace_config)))
    else:
        cfg = initial_cfg
    if not is_valid_language(req.source_lang, cfg) or not is_valid_language(req.target_lang, cfg):
        raise HTTPException(status_code=400, detail="Invalid language code(s)")

    # Now you can proceed with the translations
    return translate_and_score(req.text, req.source_lang, req.target_lang, cfg)

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")

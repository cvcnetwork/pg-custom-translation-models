import os
import time
import uuid
from pydantic import BaseModel
import yaml
import sys
import asyncio

from predictionguard import PredictionGuard
import pandas as pd
from comet import download_model, load_from_checkpoint
import deepl
import uvicorn
import traceback
from openai import OpenAI
import munch
import huggingface_hub
huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 1000  # Set timeout to 1000 seconds
import httpx
import logging
from google.cloud import translate_v2 as translate
from fastapi import FastAPI, Header, HTTPException, Depends, Security, status
from fastapi.security import APIKeyHeader
from typing import Optional


app = FastAPI()

logging.basicConfig(level=logging.DEBUG)

#--------------------------#
#         Config           #
#--------------------------#


ymlcfg = yaml.safe_load(open(os.path.join(sys.path[0], 'config.yml')))
cfg = munch.munchify(ymlcfg)

app = FastAPI()

API_KEY = cfg.api.key
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

# Hugging Face login
if 'HUGGINGFACE_TOKEN' in os.environ:
    cfg.huggingface.token = os.environ['HUGGINGFACE_TOKEN']

huggingface_hub.login(token=cfg.huggingface.token)
TOKENIZERS_PARALLELISM = cfg.huggingface.tokenizers_parallelism
os.environ['TOKENIZERS_PARALLELISM'] = str(TOKENIZERS_PARALLELISM)


# Model URLs and API keys for NLLB and mBART
MODEL_URLS = {
    "nllb": "https://model-2qjl172w.api.baseten.co/development/predict",
    "mbart": "https://model-4q9vx963.api.baseten.co/development/predict",
}

API_KEYS = {
    "nllb": cfg.custom.models.nllb.api_key,
    "mbart": cfg.custom.models.mbart.api_key,
}

# Get a list of all supported languages
supported_languages = set()
for m in cfg.engines.keys():
    if m == "predictionguard":
        for m_inner in cfg.engines[m].models:
            supported_languages.update(cfg.engines[m].models[m_inner].languages)
    else:
        supported_languages.update(cfg.engines[m].languages)

supported_languages.update(cfg.custom.models.nllb.languages)
supported_languages.update(cfg.custom.models.mbart.languages)

# Get a list of supported models
supported_models = ['nllb', 'mbart', 'gpt-4', 'Hermes-2-Pro-Llama-3-8B', 'Hermes-2-Pro-Mistral-7B', 'Neural-Chat-7B', 'deepl', 'google_translate']

#-------------------------#
# LLM Translation Prompt  #
#-------------------------#

trans_prompt = """Translate the following {source_language} text to {target_language}. Only respond with the translation and no other text. Don't add, remove, or modify any information when translating.

{source_language} text: {input}

{target_language} translation:"""

#----------------------#
# COMET Quality Score  #
#----------------------#

# Download the COMET model
model_path = download_model(cfg.comet.model)
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
    model_output = comet_model.predict(data, batch_size=8, gpus=0)
    return QAOutput(score=model_output.system_score)

#----------------------#
# MT APIs/ Models      #
#----------------------#

async def make_request(url: str, request: dict, api_key: str):
    headers = {"Authorization": f"Api-Key {api_key}"}
    async with httpx.AsyncClient(timeout=500.0) as client:
        try:
            response = await client.post(url, json=request, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f"Request error: {str(exc)}")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=f"HTTP error: {exc.response.text}")

NLLB_LANGUAGE_MAP = {
    'ace_Arab': 'ace',
    'acm_Arab': 'acm',
    'acq_Arab': 'acq',
    'aeb_Arab': 'aeb',
    'afr_Latn': 'afr',
    'ajp_Arab': 'ajp',
    'aka_Latn': 'aka',
    'amh_Ethi': 'amh',
    'apc_Arab': 'apc',
    'arb_Arab': 'arb',
    'ars_Arab': 'ars',
    'ary_Arab': 'ary',
    'arz_Arab': 'arz',
    'asm_Beng': 'asm',
    'ast_Latn': 'ast',
    'awa_Deva': 'awa',
    'ayr_Latn': 'ayr',
    'azb_Arab': 'azb',
    'azj_Latn': 'azj',
    'bak_Cyrl': 'bak',
    'bam_Latn': 'bam',
    'ban_Latn': 'ban',
    'bel_Cyrl': 'bel',
    'bem_Latn': 'bem',
    'ben_Beng': 'ben',
    'bho_Deva': 'bho',
    'bjn_Arab': 'bjn',
    'bod_Tibt': 'bod',
    'bos_Latn': 'bos',
    'bug_Latn': 'bug',
    'bul_Cyrl': 'bul',
    'cat_Latn': 'cat',
    'ceb_Latn': 'ceb',
    'ces_Latn': 'ces',
    'cjk_Latn': 'cjk',
    'ckb_Arab': 'ckb',
    'crh_Latn': 'crh',
    'cym_Latn': 'cym',
    'dan_Latn': 'dan',
    'deu_Latn': 'deu',
    'dik_Latn': 'dik',
    'dyu_Latn': 'dyu',
    'dzo_Tibt': 'dzo',
    'ell_Grek': 'ell',
    'eng_Latn': 'eng',
    'epo_Latn': 'epo',
    'est_Latn': 'est',
    'eus_Latn': 'eus',
    'ewe_Latn': 'ewe',
    'fao_Latn': 'fao',
    'fij_Latn': 'fij',
    'fin_Latn': 'fin',
    'fon_Latn': 'fon',
    'fra_Latn': 'fra',
    'fur_Latn': 'fur',
    'fuv_Latn': 'fuv',
    'gaz_Latn': 'gaz',
    'gla_Latn': 'gla',
    'gle_Latn': 'gle',
    'glg_Latn': 'glg',
    'grn_Latn': 'grn',
    'guj_Gujr': 'guj',
    'hat_Latn': 'hat',
    'hau_Latn': 'hau',
    'heb_Hebr': 'heb',
    'hin_Deva': 'hin',
    'hne_Deva': 'hne',
    'hrv_Latn': 'hrv',
    'hun_Latn': 'hun',
    'hye_Armn': 'hye',
    'ibo_Latn': 'ibo',
    'ilo_Latn': 'ilo',
    'ind_Latn': 'ind',
    'isl_Latn': 'isl',
    'ita_Latn': 'ita',
    'jav_Latn': 'jav',
    'jpn_Jpan': 'jpn',
    'kab_Latn': 'kab',
    'kac_Latn': 'kac',
    'kam_Latn': 'kam',
    'kan_Knda': 'kan',
    'kas_Arab': 'kas',
    'kat_Geor': 'kat',
    'kaz_Cyrl': 'kaz',
    'kbp_Latn': 'kbp',
    'kea_Latn': 'kea',
    'khk_Cyrl': 'khk',
    'khm_Khmr': 'khm',
    'kik_Latn': 'kik',
    'kin_Latn': 'kin',
    'kir_Cyrl': 'kir',
    'kmb_Latn': 'kmb',
    'kmr_Latn': 'kmr',
    'knc_Arab': 'knc',
    'kon_Latn': 'kon',
    'kor_Hang': 'kor',
    'lao_Laoo': 'lao',
    'lij_Latn': 'lij',
    'lim_Latn': 'lim',
    'lin_Latn': 'lin',
    'lit_Latn': 'lit',
    'lmo_Latn': 'lmo',
    'ltg_Latn': 'ltg',
    'ltz_Latn': 'ltz',
    'lua_Latn': 'lua',
    'lug_Latn': 'lug',
    'luo_Latn': 'luo',
    'lus_Latn': 'lus',
    'lvs_Latn': 'lvs',
    'mag_Deva': 'mag',
    'mai_Deva': 'mai',
    'mal_Mlym': 'mal',
    'mar_Deva': 'mar',
    'min_Arab': 'min',
    'mkd_Cyrl': 'mkd',
    'plt_Latn': 'plt',
    'mlt_Latn': 'mlt',
    'mni_Beng': 'mni',
    'mos_Latn': 'mos',
    'mri_Latn': 'mri',
    'mya_Mymr': 'mya',
    'nld_Latn': 'nld',
    'nno_Latn': 'nno',
    'nob_Latn': 'nob',
    'npi_Deva': 'npi',
    'nso_Latn': 'nso',
    'nus_Latn': 'nus',
    'nya_Latn': 'nya',
    'oci_Latn': 'oci',
    'ory_Orya': 'ory',
    'pag_Latn': 'pag',
    'pan_Guru': 'pan',
    'pap_Latn': 'pap',
    'pes_Arab': 'pes',
    'pol_Latn': 'pol',
    'por_Latn': 'por',
    'prs_Arab': 'prs',
    'pbt_Arab': 'pbt',
    'quy_Latn': 'quy',
    'ron_Latn': 'ron',
    'run_Latn': 'run',
    'rus_Cyrl': 'rus',
    'sag_Latn': 'sag',
    'san_Deva': 'san',
    'sat_Olck': 'sat',
    'scn_Latn': 'scn',
    'shn_Mymr': 'shn',
    'sin_Sinh': 'sin',
    'slk_Latn': 'slk',
    'slv_Latn': 'slv',
    'smo_Latn': 'smo',
    'sna_Latn': 'sna',
    'snd_Arab': 'snd',
    'som_Latn': 'som',
    'sot_Latn': 'sot',
    'spa_Latn': 'spa',
    'als_Latn': 'als',
    'srd_Latn': 'srd',
    'srp_Cyrl': 'srp',
    'ssw_Latn': 'ssw',
    'sun_Latn': 'sun',
    'swe_Latn': 'swe',
    'swh_Latn': 'swh',
    'szl_Latn': 'szl',
    'tam_Taml': 'tam',
    'tat_Cyrl': 'tat',
    'tel_Telu': 'tel',
    'tgk_Cyrl': 'tgk',
    'tgl_Latn': 'tgl',
    'tha_Thai': 'tha',
    'tir_Ethi': 'tir',
    'taq_Latn': 'taq',
    'tpi_Latn': 'tpi',
    'tsn_Latn': 'tsn',
    'tso_Latn': 'tso',
    'tuk_Latn': 'tuk',
    'tum_Latn': 'tum',
    'tur_Latn': 'tur',
    'twi_Latn': 'twi',
    'tzm_Tfng': 'tzm',
    'uig_Arab': 'uig',
    'ukr_Cyrl': 'ukr',
    'umb_Latn': 'umb',
    'urd_Arab': 'urd',
    'uzn_Latn': 'uzn',
    'vec_Latn': 'vec',
    'vie_Latn': 'vie',
    'war_Latn': 'war',
    'wol_Latn': 'wol',
    'xho_Latn': 'xho',
    'ydd_Hebr': 'ydd',
    'yor_Latn': 'yor',
    'yue_Hant': 'yue',
    'zho_Hans': 'zho',
    'zsm_Latn': 'zsm',
    'zul_Latn': 'zul',
}
def convert_to_nllb_code(language_code):
    return NLLB_LANGUAGE_MAP.get(language_code, language_code)

MBART_LANGUAGE_MAP = {
    'arb_Arab': 'ar', 'ces_Latn': 'cs', 'deu_Latn': 'de', 'eng_Latn': 'en', 'spa_Latn': 'es', 'est_Latn': 'et', 
    'fin_Latn': 'fi', 'fra_Latn': 'fr', 'guj_Gujr': 'gu', 'hin_Deva': 'hi', 'ita_Latn': 'it', 'jpn_Jpan': 'ja', 
    'kaz_Cyrl': 'kk', 'kor_Hang': 'ko', 'lit_Latn': 'lt', 'lvs_Latn': 'lv', 'mya_Mymr': 'my', 'npi_Deva': 'ne', 
    'nld_Latn': 'nl', 'rus_Cyrl': 'ru', 'sin_Sinh': 'si', 'tur_Latn': 'tr', 'vie_Latn': 'vi', 'zho_Hans': 'zh'
}

def convert_to_mbart_source_code(nllb_code):
    return MBART_LANGUAGE_MAP.get(nllb_code, nllb_code[:2])  # Default to first two characters if not found

async def nllb_mbart_translation(text, source_language, target_language, model):
    try:
        url = MODEL_URLS[model]
        api_key = API_KEYS[model]
        
        if model == "mbart":
            source_language = convert_to_mbart_source_code(source_language)
            # target_language remains in NLLB format for mBART
        
        request_data = {
            "text": text,
            "source_language": source_language,
            "target_language": target_language
        }
        response = await make_request(url, request_data, api_key)
        
        if 'translation' in response:
            translation_text = response['translation']
            qa_input = QAInput(source=text, translation=translation_text)
            score = await asyncio.to_thread(get_quality_score, qa_input)
            return {
                "translation": translation_text,
                "score": score.score,
                "model": model,
                "status": "success"
            }
        else:
            raise ValueError(f"Unexpected response format from {model} API")

    except Exception as e:
        logging.error(f"{model} translation error: {str(e)}")
        return {
            "translation": "",
            "score": -100,
            "model": model,
            "status": f"error: {str(e)}"
        }

import logging
from deepl import Translator, DeepLException

DEEPL_LANGUAGE_MAP = {
    'arb_Arab': 'AR', 'ara': 'AR',  # Arabic
    'bul_Cyrl': 'BG', 'bul': 'BG',  # Bulgarian
    'zho_Hans': 'ZH', 'zho': 'ZH',  # Chinese (Simplified)
    'ces_Latn': 'CS', 'ces': 'CS',  # Czech
    'dan_Latn': 'DA', 'dan': 'DA',  # Danish
    'nld_Latn': 'NL', 'nld': 'NL',  # Dutch
    'eng_Latn': 'EN-GB', 'eng': 'EN-GB',  # English
    'est_Latn': 'ET', 'est': 'ET',  # Estonian
    'fin_Latn': 'FI', 'fin': 'FI',  # Finnish
    'fra_Latn': 'FR', 'fra': 'FR',  # French
    'deu_Latn': 'DE', 'deu': 'DE',  # German
    'hun_Latn': 'HU', 'hun': 'HU',  # Hungarian
    'ind_Latn': 'ID', 'ind': 'ID',  # Indonesian
    'ita_Latn': 'IT', 'ita': 'IT',  # Italian
    'jpn_Jpan': 'JA', 'jpn': 'JA',  # Japanese
    'kor_Hang': 'KO', 'kor': 'KO',  # Korean
    'lvs_Latn': 'LV', 'lav': 'LV',  # Latvian
    'lit_Latn': 'LT', 'lit': 'LT',  # Lithuanian
    'ell_Grek': 'EL', 'ell': 'EL',  # Greek
    'nob_Latn': 'NB', 'nob': 'NB',  # Norwegian Bokmål
    'pol_Latn': 'PL', 'pol': 'PL',  # Polish
    'por_Latn': 'PT-BR', 'por': 'PT-BR',  # Portuguese
    'ron_Latn': 'RO', 'ron': 'RO',  # Romanian
    'rus_Cyrl': 'RU', 'rus': 'RU',  # Russian
    'slk_Latn': 'SK', 'slk': 'SK',  # Slovak
    'slv_Latn': 'SL', 'slv': 'SL',  # Slovenian
    'spa_Latn': 'ES', 'spa': 'ES',  # Spanish
    'swe_Latn': 'SV', 'swe': 'SV',  # Swedish
    'tur_Latn': 'TR', 'tur': 'TR',  # Turkish
    'ukr_Cyrl': 'UK', 'ukr': 'UK',  # Ukrainian
}
def convert_to_deepl_code(language_code):
    return DEEPL_LANGUAGE_MAP.get(language_code)

async def deepl_translation(text, target_language):
    try:
        # Convert NLLB or ISO code to DeepL code
        deepl_code = convert_to_deepl_code(target_language)
        if not deepl_code:
            logging.warning(f"Unsupported DeepL target language: {target_language}")
            return {
                "translation": "",
                "score": -100,
                "model": "deepl",
                "status": f"error: Unsupported target language: {target_language}"
            }

        # Initialize the deepl translator
        deepl_translator = Translator(auth_key=cfg.engines.deepl.api_key)

        # Get the translation
        response = await asyncio.to_thread(
            deepl_translator.translate_text,
            text,
            target_lang=deepl_code
        )

        # Process the response
        if response is not None and response.text.strip():
            qa_input = QAInput(source=text, translation=response.text)
            score = await asyncio.to_thread(get_quality_score, qa_input)
            return {
                "translation": response.text,
                "score": score.score,
                "model": "deepl",
                "status": "success"
            }
        else:
            raise ValueError("Empty response from DeepL API")

    except DeepLException as e:
        logging.error(f"DeepL API error: {str(e)}")
        return {
            "translation": "",
            "score": -100,
            "model": "deepl",
            "status": f"error: DeepL API error: {str(e)}"
        }
    except Exception as e:
        logging.error(f"DeepL translation error: {str(e)}")
        return {
            "translation": "",
            "score": -100,
            "model": "deepl",
            "status": f"error: {str(e)}"
        }
# Modify the pg_openai_translation function
async def pg_openai_translation(text, source_language, target_language, model):
    try:
        # Initialize the client
        if "gpt" in model:
            client = OpenAI(api_key=cfg.engines.openai.api_key)
        else:
            client = PredictionGuard(api_key=cfg.engines.predictionguard.api_key)

        # Call the API (wrap in asyncio.to_thread as these clients might not be async)
        result = await asyncio.to_thread(
            client.chat.completions.create,
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
        if hasattr(result, 'choices') and result.choices:
            response_message = result.choices[0].message.content.strip().split('\n')[0]
        else:
            # Handle PredictionGuard response or unexpected structure
            response_message = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        if response_message:
            qa_input = QAInput(source=text, translation=response_message)
            score = await asyncio.to_thread(get_quality_score, qa_input)
            return {
                "translation": response_message, 
                "score": score.score, 
                "model": model,
                "status": "success"
            }
        else:
            raise ValueError("Empty response from API")
    except Exception as e:
        logging.error(f"Error in {model} translation: {str(e)}")
        return {
            "translation": "", 
            "score": -100, 
            "model": model, 
            "status": f"error: {str(e)}"
        }

async def google_translate(text, source_language, target_language):
    try:
        url = f"https://translation.googleapis.com/language/translate/v2?key={cfg.engines.google_translate.api_key}"
        
        # Convert NLLB or ISO code to Google Translate code
        source_code = convert_to_google_code(source_language)
        target_code = convert_to_google_code(target_language)
        
        if not target_code:
            logging.warning(f"Unsupported Google Translate target language: {target_language}")
            return {
                "translation": "",
                "score": -100,
                "model": "google_translate",
                "status": f"error: Unsupported target language: {target_language}"
            }

        # Prepare the request payload
        payload = {
            "q": text,
            "target": target_code,
            "source": source_code,
            "format": "text"
        }

        # Make the API call
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            result = response.json()

        if 'data' in result and 'translations' in result['data'] and result['data']['translations']:
            translation_text = result['data']['translations'][0]['translatedText']
            qa_input = QAInput(source=text, translation=translation_text)
            score = await asyncio.to_thread(get_quality_score, qa_input)
            return {
                "translation": translation_text,
                "score": score.score,
                "model": "google_translate",
                "status": "success"
            }
        else:
            raise ValueError("Unexpected response format from Google Translate API")

    except Exception as e:
        logging.error(f"Google Translate error: {str(e)}")
        return {
            "translation": "",
            "score": -100,
            "model": "google_translate",
            "status": f"error: {str(e)}"
        }

def convert_to_google_code(language_code):
    # Split the code and take the first part (e.g., 'fra' from 'fra_Latn')
    code = language_code.split('_')[0]
    
    # If the code is 3 letters long, take the first two letters
    if len(code) == 3:
        return code[:2]
    
    # For other cases (2-letter codes or unexpected formats), return as is
    return code

# Update the translate_and_score function
async def translate_and_score(text, source_language_nllb, target_language_nllb):
    translation_results = []
    best_translation = None
    best_score = -1
    best_model = ""

    created_timestamp = int(time.time())
    unique_id = "translation-" + str(uuid.uuid4()).replace("-", "")

    # Prepare tasks for all models
    tasks = [
        nllb_mbart_translation(text, source_language_nllb, target_language_nllb, "nllb"),
        nllb_mbart_translation(text, source_language_nllb, target_language_nllb, "mbart"),
        pg_openai_translation(text, source_language_nllb, target_language_nllb, "gpt-4"),
        pg_openai_translation(text, source_language_nllb, target_language_nllb, "Hermes-2-Pro-Llama-3-8B"),
        pg_openai_translation(text, source_language_nllb, target_language_nllb, "Hermes-2-Pro-Mistral-7B"),
        pg_openai_translation(text, source_language_nllb, target_language_nllb, "Neural-Chat-7B"),
        deepl_translation(text, target_language_nllb),
        google_translate(text, source_language_nllb, target_language_nllb)  # Add this line
    ]

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    model_names = ['nllb', 'mbart', 'gpt-4', 'Hermes-2-Pro-Llama-3-8B', 'Hermes-2-Pro-Mistral-7B', 'Neural-Chat-7B', 'deepl', 'google_translate']

    # Process results
    for i, (model, result) in enumerate(zip(model_names, results)):
        logging.info(f"Processing result for model: {model}")
        if isinstance(result, Exception):
            logging.error(f"Translation error for {model}: {str(result)}")
            translation_results.append({
                "translation": "",
                "score": -100,
                "model": model,
                "status": f"error: {str(result)}"
            })
        else:
            logging.info(f"Successful result for {model}: {result}")
            translation_results.append(result)
            if result["status"] == "success" and result["score"] > best_score:
                best_translation = result["translation"]
                best_score = result["score"]
                best_model = result["model"]
                logging.info(f"New best translation from {model} with score {best_score}")

    logging.info(f"Number of translation results: {len(translation_results)}")

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

class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

def is_valid_language(language_code):
    return language_code in supported_languages

@app.get("/")
async def read_root(api_key: str = Depends(get_api_key)):
    return {"status": "healthy"}

@app.post("/translate")
async def translate(req: TranslateRequest, api_key: str = Depends(get_api_key)):
    if not is_valid_language(req.source_lang) or not is_valid_language(req.target_lang):
        raise HTTPException(status_code=400, detail="Invalid language code(s)")

    return await translate_and_score(req.text, req.source_lang, req.target_lang)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)


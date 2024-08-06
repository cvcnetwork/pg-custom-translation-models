# Translation API built on Prediction Guard

Example translation app built on top of Prediction Guard, extendable with DeepL, OpenAI, etc.

To run:
1. Copy `config_template.yml` to `config.yml`
2. Fill in or modify `config.yml` as needed/ appropriate
3. Run `python main.py`

(alternatively run in Docker)

## Adding a customer translation engine

This translation service can integrate with custom translation engines, assuming they fulfill the expected API contract. When adding a custom engine, the following kind of entry is needed in the `config.yml`:

```
  custom:
    models:
      model_name:
        url: https://your-custom-url
        api_key: your-custom-api-key
        languages:
          - eng
          - fra
          - deu
          - cmn
```

The `config.yml` entry should have:
- `url`: The endpoint location to call the engine
- `api_key`: The API key for the engine endpoint, which will be added as `x-api-key` in the API call headers
- `languages`: List of supported ISO639-3 language codes

The custom translation engine endpoint should expect a JSON body that looks like:

```json
{
    "text": "The sky is blue",
    "model": "nllb",
    "source_lang": "eng",
    "target_lang": "fra"
}
```

Where:
- `text` (required): The text that will be translated
- `model` (required): The name of the model used (as some endpoints will integrate multiple models)
- `target_lang` (required): The ISO639-3 (three letter) code specifying the target language
- `source_lang` (optional): The ISO639-3 (three letter) code specifying the source language

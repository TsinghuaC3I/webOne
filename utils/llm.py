
import time
import logging
import requests
from utils.utils_localhost import retry_with_exponential_backoff, convert_msg_format_openai_to_localhost
import json


def call_gemini_api(messages):
    retry_times = 0
    while True:
        try:
            logging.info('Calling Gemini API...')

            payload = json.dumps({
                "contents": messages,
                }
            )
            headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + 'sk-Aef1uNqa4Eds3zxzlQ5uXs87K8KvywBvdV4vLhM3agZHqa33',
            }
            url = "http://35.164.11.19:3887/v1beta/models/gemini-2.5-pro:generateContent"

            response = requests.request("POST", url, headers=headers, data=payload)

            # print(response.text)
            openai_response = json.loads(response.content.decode('utf-8'))
            # openai_response = openai_client.chat.completions.create(
            #     model=args.api_model, messages=messages, max_tokens=1000, seed=args.seed
            # )
            prompt_tokens = openai_response["usageMetadata"]['promptTokenCount']
            completion_tokens = openai_response["usageMetadata"]['candidatesTokenCount']

            logging.info(f'Prompt Tokens: {prompt_tokens}; Completion Tokens: {completion_tokens}')

            gpt_call_error = False
            return prompt_tokens, completion_tokens, gpt_call_error, openai_response

        except Exception as e:
            logging.info(f'Error occurred, retrying. Error type: {type(e).__name__}')
            print('error in calling Gemini API')
            print(e)
            # print(openai_response)
            time.sleep(15)
            if type(e).__name__ == 'RateLimitError':
                time.sleep(10)

            elif type(e).__name__ == 'APIError':
                time.sleep(15)

            # elif type(e).__name__ == 'InvalidRequestError':
            #     gpt_call_error = True
            #     return None, None, gpt_call_error, None
            #
            # else:
            #     gpt_call_error = True
            #     return None, None, gpt_call_error, None

        retry_times += 1
        if retry_times == 100:
            logging.info('Retrying too many times')
            return None, None, True, None


@retry_with_exponential_backoff(initial_delay=1, exponential_base=1.05, jitter=False, max_retries=10)
def post_with_retry(url, cur_example):
    response = requests.post(url, json=cur_example)
    if response.status_code == 200:
        result = response.json()
    else:
        print(f"Wrong HTTP response code {response.status_code}")
        raise ValueError(f"Wrong HTTP response code {response.status_code}")
    return result

def call_localhost_api(args, localhost_api, messages, task_dir):
    # First, transfer the format of messages
    converted_msg = convert_msg_format_openai_to_localhost(messages, task_dir, last_n=args.max_attached_imgs)

    retry_times = 0
    while True:
        try:
            logging.info('Calling Localhost API...')
            result = post_with_retry(localhost_api, converted_msg)
            openai_response = result["text"]
            prompt_tokens = result["prompt_tokens"]
            completion_tokens = result["completion_tokens"]
            logging.info(f'Prompt Tokens: {prompt_tokens}; Completion Tokens: {completion_tokens}')

            call_error = False
            return prompt_tokens, completion_tokens, call_error, openai_response

        except Exception as e:
            logging.info(f'Error occurred, retrying. Error type: {type(e).__name__}')

            if type(e).__name__ == 'ValueError':
                time.sleep(10)

            else:
                call_error = True
                return None, None, call_error, None

        retry_times += 1
        if retry_times == 10:
            logging.info('Retrying too many times')
            return None, None, True, None


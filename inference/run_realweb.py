import platform
import argparse
import time
import json
import re
import os
import shutil
import logging
from time import sleep

import requests

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from utils.utils import clip_message_and_obs_gemini, print_message_gemini
from utils.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_TEXT_ONLY, SYSTEM_PROMPT_ACTREE_IMAGE
from openai import OpenAI
from utils.utils import get_web_element_rect, encode_image, extract_information, print_message, \
    get_webarena_accessibility_tree, get_pdf_retrieval_ans_from_assistant, clip_message_and_obs, \
    clip_message_and_obs_text_only
import copy
from utils.utils_localhost import retry_with_exponential_backoff, convert_msg_format_openai_to_localhost
import json


def read_json(ffile):
    with open(ffile, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def setup_logger(folder_path):
    log_file_path = os.path.join(folder_path, 'agent.log')

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def driver_config(args):
    options = webdriver.ChromeOptions()

    if args.save_accessibility_tree:
        args.force_device_scale = True

    if args.force_device_scale:
        options.add_argument("--force-device-scale-factor=1")
    if args.headless:
        options.add_argument("--headless")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
    options.add_experimental_option(
        "prefs", {
            "download.default_directory": args.download_dir,
            "plugins.always_open_pdf_externally": True
        }
    )

    options.add_argument("--no-sandbox")

    return options


def format_msg_acctree_and_screenshot_gemini(it, init_msg, pdf_obs, warn_obs, ac_tree, web_img_b64):
    if it == 1:
        init_msg += f"I've provided the Accessibility Tree and Screenshot of the No.1 Page:\n{ac_tree}\nScreenshot:\n"
        init_msg_format = {
            "role": "user",
            "parts": [
                {"text": init_msg},
            ]
        }
        init_msg_format["parts"].append(
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": web_img_b64
                },
            }
        )
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                "role": "user",
                "parts": [
                    {
                        "text": f"The No.{it} Page:\nObservation:{warn_obs} please analyze the attached Accessibility Tree & screenshot and give the Thought and Action.\n{ac_tree}\nScreenshot:\n"
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": web_img_b64
                        },
                    }
                ]
            }
        else:
            curr_msg = {
                "role": "user",
                "parts": [
                    {
                        "text": f"The No.{it} Page:\nObservation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The accessibility tree & screenshot of the current page (The No.{it} Page) is also attached, give the Thought and Action.\n{ac_tree}\nScreenshot:\n"},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": web_img_b64
                        },
                    }
                ]
            }
        return curr_msg


def format_msg_acctree_and_screenshot(it, init_msg, pdf_obs, warn_obs, ac_tree, web_img_b64):
    if it == 1:
        init_msg += f"I've provided the Accessibility Tree and Screenshot.\n{ac_tree}\nScreenshot:\n"
        init_msg_format = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': init_msg},
            ]
        }
        init_msg_format['content'].append({"type": "image_url",
                                           "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}})
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text',
                     'text': f"Observation:{warn_obs} please analyze the attached Accessibility Tree & screenshot and give the Thought and Action.\n{ac_tree}\nScreenshot:\n"},
                    {
                        'type': 'image_url',
                        'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}
                    }
                ]
            }
        else:
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text',
                     'text': f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The accessibility tree & screenshot of the current page is also attached, give the Thought and Action.\n{ac_tree}\nScreenshot:\n"},
                    {
                        'type': 'image_url',
                        'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}
                    }
                ]
            }
        return curr_msg


def call_gemini_api(args, openai_client, messages):
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

            print(response.text)
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

            if type(e).__name__ == 'RateLimitError':
                time.sleep(10)

            elif type(e).__name__ == 'APIError':
                time.sleep(15)

            elif type(e).__name__ == 'InvalidRequestError':
                gpt_call_error = True
                return None, None, gpt_call_error, None

            # else:
            #     gpt_call_error = True
            #     return None, None, gpt_call_error, None

        retry_times += 1
        if retry_times == 10:
            logging.info('Retrying too many times')
            return None, None, True, None


def call_gpt4v_api(args, openai_client, messages):
    retry_times = 0
    while True:
        try:
            logging.info('Calling OpenAI API...')
            openai_response = openai_client.chat.completions.create(
                model=args.api_model, messages=messages, max_tokens=1000, seed=args.seed
            )
            prompt_tokens = openai_response.usage.prompt_tokens
            completion_tokens = openai_response.usage.completion_tokens

            logging.info(f'Prompt Tokens: {prompt_tokens}; Completion Tokens: {completion_tokens}')

            gpt_call_error = False
            return prompt_tokens, completion_tokens, gpt_call_error, openai_response

        except Exception as e:
            logging.info(f'Error occurred, retrying. Error type: {type(e).__name__}')

            if type(e).__name__ == 'RateLimitError':
                time.sleep(1)

            elif type(e).__name__ == 'APIError':
                time.sleep(1)

            elif type(e).__name__ == 'InvalidRequestError':
                gpt_call_error = True
                return None, None, gpt_call_error, None

            else:
                gpt_call_error = True
                return None, None, gpt_call_error, None

        retry_times += 1
        if retry_times == 10:
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


def exec_action_click(info, web_ele, driver_task):
    # update, open too much tabs in some webs
    original_tabs = driver_task.window_handles
    driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
    web_ele.click()
    time.sleep(3)
    new_tabs = driver_task.window_handles
    if len(new_tabs) > len(original_tabs):
        new_tab = [tab for tab in new_tabs if tab not in original_tabs][0]
        driver_task.switch_to.window(new_tab)
        new_tab_url = driver_task.current_url
        driver_task.close()
        driver_task.switch_to.window(original_tabs[0])
        driver_task.get(new_tab_url)
        time.sleep(2)


def exec_action_type(info, web_ele, driver_task):
    warn_obs = ""
    type_content = info['content']

    ele_tag_name = web_ele.tag_name.lower()
    ele_type = web_ele.get_attribute("type")
    # outer_html = web_ele.get_attribute("outerHTML")
    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (
            ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
        warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."

    # 待修改
    try:
        # Not always work to delete
        web_ele.clear()
        # Another way to delete
        if platform.system() == 'Darwin':
            web_ele.send_keys(Keys.COMMAND + "a")
        else:
            web_ele.send_keys(Keys.CONTROL + "a")
        web_ele.send_keys(" ")
        web_ele.send_keys(Keys.BACKSPACE)
    except:
        pass

    actions = ActionChains(driver_task)
    actions.click(web_ele).perform()
    actions.pause(1)

    try:
        if 'www.cvs.com' not in driver_task.current_url:
            driver_task.execute_script(
                """window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};""")
    except:
        pass

    actions.send_keys(type_content)
    actions.pause(2)

    actions.send_keys(Keys.ENTER)
    actions.perform()
    time.sleep(10)
    return warn_obs


def calculate_intersection(args, bound_box):
    x1, y1, w1, h1 = (0, 0, args.window_width, args.window_height)
    x2, y2, w2, h2 = bound_box

    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)

    width = max(0, right - left)
    height = max(0, bottom - top)

    return [left, top, width, height]


def exec_action_scroll(info, web_eles, driver_task, args, obs_info):
    scroll_ele_number = info['number']
    scroll_content = info['content']
    if scroll_ele_number == "WINDOW":
        if scroll_content == 'down':
            driver_task.execute_script(f"window.scrollBy(0, {args.window_height * 2 // 3});")
        else:
            driver_task.execute_script(f"window.scrollBy(0, {-args.window_height * 2 // 3});")
    else:

        scroll_ele_number = int(scroll_ele_number)
        element_box = obs_info[scroll_ele_number]['union_bound']
        element_box = calculate_intersection(args, element_box)
        element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
        web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);",
                                             element_box_center[0], element_box_center[1])
        actions = ActionChains(driver_task)
        driver_task.execute_script("arguments[0].focus();", web_ele)
        if scroll_content == 'down':
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_DOWN).key_up(Keys.ALT).perform()
        else:
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_UP).key_up(Keys.ALT).perform()
    time.sleep(3)


# openai.api_key = "sk-vpj3a5KGeKl145pIDa1d3aB9Af994c7bAdAa53B0E2447807"  # os.environ["OPENAI_API_KEY"]
# openai.api_base = "https://api3.apifans.com/v1"
from PIL import Image
import os


def compress_image(input_path, output_path, target_size=(1024, 768)):
    """
    将图像压缩到指定尺寸

    参数:
    input_path: 输入图像路径
    output_path: 输出图像路径
    target_size: 目标尺寸，默认为(1024, 768)
    """
    try:
        # 打开原始图像
        with Image.open(input_path) as img:
            # 获取原始图像尺寸
            original_size = img.size
            # print(f"原始图像尺寸: {original_size}")

            # 调整图像尺寸
            resized_img = img.resize(target_size, Image.LANCZOS)

            # 保存调整后的图像
            resized_img.save(output_path)
            # print(f"图像已成功压缩到: {target_size}")
            # print(f"保存路径: {output_path}")

            return True

    except Exception as e:
        print(f"处理图像时出错: {e}")
        return False


import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='./test_auto_eval/alltest.json')
    parser.add_argument('--max_iter', type=int, default=15)
    parser.add_argument("--api_localhost", default="", type=str, help="localhost API")
    parser.add_argument("--api_file_assistant", action='store_true', help="Assistant API to analyse files")
    parser.add_argument("--api_key", default="sk-vpj3a5KGeKl145pIDa1d3aB9Af994c7bAdAa53B0E2447807", type=str,
                        help="YOUR_OPENAI_API_KEY")
    parser.add_argument("--api_model", default="/root/models2/Qwen3-VL/Qwen3-VL-8B-Instruct", type=str, help="api model name")
    parser.add_argument("--output_dir", type=str, default='./results/without_guidebook')
    parser.add_argument("--evaluation_name", type=str, default="qwen3-vl-8b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_attached_imgs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--download_dir", type=str, default="downloads")
    parser.add_argument("--text_only", action='store_true')
    # for web browser
    parser.add_argument("--headless", default=True, help='The window of selenium')
    parser.add_argument("--save_accessibility_tree", default=True)
    parser.add_argument("--force_device_scale", action='store_true')
    parser.add_argument("--window_width", type=int, default=1024)
    parser.add_argument("--window_height", type=int, default=1024)  # for headless mode, there is no address bar
    parser.add_argument("--fix_box_color", action='store_true')

    args = parser.parse_args()

    # OpenAI client
    if not args.api_localhost:
        openai_api_key = "EMPTY"
        openai_api_base = "http://101.6.66.14:15006/v1/completions"

        # client = OpenAI(
        #     api_key=openai_api_key,
        #     base_url=openai_api_base,
        # )
        client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

        # client = OpenAI(api_key=args.api_key, base_url="https://api3.apifans.com/v1")
    else:
        client_localhost = args.api_localhost

        if args.api_file_assistant:
            client_file_assis = OpenAI(api_key=args.api_key)

    options = driver_config(args)
    print(111)
    # Save Result file

    result_dir = args.output_dir
    result_dir = os.path.join(result_dir, args.evaluation_name)
    os.makedirs(result_dir, exist_ok=True)

    large_window = {}

    # Load tasks
    tasks = read_json(args.test_file)

    # random.seed(0)
    random.seed(1)
    random.shuffle(tasks)
    tasks = tasks[0:100]
    random.shuffle(tasks)

    print('number of tasks', len(tasks))
    for task_id in range(0, len(tasks), 1):
        task = tasks[task_id]
        id = f"task{task['new_id']}"

        cur_website = task['web']

        # if id not in retry_task:
        #     continue
        task_dir = os.path.join(result_dir, 'task{}'.format(task["id"]))
        if os.path.exists(os.path.join(task_dir, 'interact_messages.json')):
            data = json.load(open(os.path.join(task_dir, 'interact_messages.json')))[-1]['content']
            if 'ANSWER' in data:
                print('1 This task has been processed', task["id"])
                continue

        # if os.path.exists(os.path.join(task_dir, 'eval_res.json')):
        #     data = json.load(open(os.path.join(task_dir, 'eval_res.json')))
        #     if data[-1]["success"] == 1:
        #         print('2 This task has been processed', task["id"])
        #         continue

        os.makedirs(task_dir, exist_ok=True)
        setup_logger(task_dir)
        logging.info(f'########## TASK{task["id"]} ##########')
        print(f'########## TASK{task["id"]} ##########')
        try:
            driver_task = webdriver.Chrome(options=options)
        except Exception as e:
            print(e)
            continue

        # About window size, 765 tokens
        # You can resize to height = 512 by yourself (255 tokens, Maybe bad performance)

        window_height = args.window_height
        window_width = args.window_width

        if cur_website in large_window:
            window_height = 1536
            window_width = args.window_width
        driver_task.set_window_size(window_width, window_height)  # larger height may contain more web information
        driver_task.get(task['web'])  # https://www.recreation.gov/
        # if 'amazon' in task['web']:
        #     import pickle
        #     with open('amazon_cookies.pkl', 'rb') as file:
        #         cookies = pickle.load(file)
        #         for cookie in cookies:
        #             driver_task.add_cookie(cookie)
        #     driver_task.refresh()

        try:
            driver_task.find_element(By.TAG_NAME, 'body').click()
        except:
            pass
        # sometimes enter SPACE, the page will sroll down
        driver_task.execute_script(
            """window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")
        time.sleep(5)

        # We only deal with PDF file
        for filename in os.listdir(args.download_dir):
            file_path = os.path.join(args.download_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        download_files = []  # sorted(os.listdir(args.download_dir))

        fail_obs = ""  # When error execute the action
        pdf_obs = ""  # When download PDF file
        warn_obs = ""  # Type warning
        pattern = r'Thought:|Action:|Observation:'

        messages = [{'role': 'user', 'content': [{"text": SYSTEM_PROMPT_ACTREE_IMAGE, "type": "text"}]}]
        obs_prompt = "Observation: please analyze the attached accessibility tree & screenshot and give the Thought and Action. "

        init_msg = f"""Now given a task: {task['ques']}  Please interact with https://www.example.com and get the answer. (if the first page is a checkout, you just need to GoBack !!! once it appears. Close all irrelevant pop-ups that appear during this process.)\n"""
        if 'cvs' in cur_website:
            init_msg = f"""Now given a task: {task['ques']}  Please interact with https://www.example.com and get the answer. (You will see the notice page when you open a new page in the cvs, just Click the 'find out more about CVS'!!! once it appears. Don't restart!! Close all irrelevant pop-ups that appear during this process.)\n"""
        if 'sports.yahoo' in cur_website:
            init_msg = f"""Now given a task: {task['ques']}  Please interact with https://www.example.com and get the answer. (if the first page is a checkout, you just need to GoBack !!! once it appears.  Close all irrelevant pop-ups that appear during this process.)\n"""
        init_msg = init_msg.replace('https://www.example.com', task['web']).replace('THEYEAR','2025').replace('LastYear', '2024').replace('THEMONTH','NOV.').replace('NEXTMONTH', 'DEC.')
        task['ques'] = task['ques'].replace('THEYEAR', '2025').replace('LastYear', '2024').replace('THEMONTH', 'NOV.').replace('NEXTMONTH', 'DEC.')

        init_msg = init_msg + obs_prompt
        print(f"Task: {task['ques']}")
        it = 0
        accumulate_prompt_token = 0
        accumulate_completion_token = 0
        # corresponding_box_mapping = {}
        while it < args.max_iter:
            logging.info(f'Iter: {it}')
            print(f"Step: {it}")
            it += 1

            try:
                accessibility_tree_path = os.path.join(task_dir, 'accessibility_tree{}'.format(it))
                ac_tree, obs_info = get_webarena_accessibility_tree(driver_task, accessibility_tree_path)

            except Exception as e:
                logging.error('Driver error when obtaining accessibility tree.')
                logging.error(e)
                break

            img_path = os.path.join(task_dir, 'screenshot{}.png'.format(it))
            driver_task.save_screenshot(img_path)
            compress_image(img_path, img_path)

            # html_path = os.path.join(task_dir, 'page{}.html'.format(it))
            # with open(html_path, 'w', encoding='utf-8') as f_html:
            #     f_html.write(driver_task.page_source)

            # encode image
            b64_img = encode_image(img_path)

            if not fail_obs:
                curr_msg = format_msg_acctree_and_screenshot(it, init_msg, pdf_obs, warn_obs, ac_tree, b64_img)
                messages.append(curr_msg)
            else:
                curr_msg = format_msg_acctree_and_screenshot(it, init_msg, "", fail_obs, ac_tree, b64_img)
                messages.append(curr_msg)

            # Clip messages, too many attached images may cause confusion
            if args.api_localhost:
                prompt_tokens, completion_tokens, call_error, model_response = call_localhost_api(args,
                                                                                                  client_localhost,
                                                                                                  messages, task_dir)
            else:
                messages_ = clip_message_and_obs(copy.deepcopy(messages), args.max_attached_imgs)
                # prompt_tokens, completion_tokens, call_error, openai_response = call_gpt4v_api(args, client, messages_)
                prompt_tokens, completion_tokens, call_error, openai_response = call_gpt4v_api(args, client, messages_)

            if call_error:
                sleep(1)
                continue
                # break
            else:
                accumulate_prompt_token += prompt_tokens
                accumulate_completion_token += completion_tokens
                logging.info(
                    f'Accumulate Prompt Tokens: {accumulate_prompt_token}; Accumulate Completion Tokens: {accumulate_completion_token}')
                logging.info('API call complete...')

            if args.api_localhost:
                response_msg = model_response
                messages.append({'role': 'assistant', 'content': model_response})
            elif True:
                response_msg = openai_response.choices[0].message.content
                messages.append({'role': 'assistant', 'content': response_msg})
            else:
                response_msg = openai_response['candidates'][0]['content']['parts']
                messages.append({'role': 'model', 'parts': response_msg})

            # extract action info
            try:
                if 'Thought:' not in response_msg:
                    response_msg[0]['text'] = 'Thought:' + response_msg
                assert 'Thought:' in response_msg and 'Action:' in response_msg
            except AssertionError as e:
                logging.error(e)
                fail_obs = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
                print(fail_obs)
                continue

            bot_thought = re.split(pattern, response_msg)[1].strip()
            print(f'Thought: {bot_thought}')
            chosen_action = re.split(pattern, response_msg)[2].strip()

            # print(chosen_action)
            action_key, info = extract_information(chosen_action)
            print(f'Action: {chosen_action}')
            fail_obs = ""
            pdf_obs = ""
            warn_obs = ""
            # execute actionq

            try:
                window_handle_task = driver_task.current_window_handle
                driver_task.switch_to.window(window_handle_task)

                if action_key == 'click':

                    click_ele_number = int(info[0])
                    element_box = obs_info[click_ele_number]['union_bound']
                    element_box = calculate_intersection(args, element_box)
                    element_box_center = (element_box[0] + element_box[2] // 2,
                                          element_box[1] + element_box[3] // 2)
                    web_ele = driver_task.execute_script(
                        "return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0],
                        element_box_center[1])
                    web_ele_attr_info = f"Web Element Info; <role: '{obs_info[click_ele_number]['role']['value']}'; name: '{obs_info[click_ele_number]['name']['value']}'>"
                    print(web_ele_attr_info)

                    ele_tag_name = web_ele.tag_name.lower()
                    ele_type = web_ele.get_attribute("type")
                    # print ('ELEMENT TO OPERATE ON', web_ele, ele_tag_name, ele_type)
                    exec_action_click(info, web_ele, driver_task)

                    if args.api_file_assistant:
                        # deal with PDF file
                        current_files = sorted(os.listdir(args.download_dir))
                        if current_files != download_files:
                            # wait for download finish
                            time.sleep(10)
                            current_files = sorted(os.listdir(args.download_dir))

                            current_download_file = [pdf_file for pdf_file in current_files if
                                                     pdf_file not in download_files and pdf_file.endswith('.pdf')]
                            if current_download_file:
                                pdf_file = current_download_file[0]
                                pdf_obs = get_pdf_retrieval_ans_from_assistant(client_file_assis,
                                                                               os.path.join(args.download_dir,
                                                                                            pdf_file), task['ques'])
                                shutil.copy(os.path.join(args.download_dir, pdf_file), task_dir)
                                pdf_obs = "You downloaded a PDF file, I ask the Assistant API to answer the task based on the PDF file and get the following response: " + pdf_obs
                            download_files = current_files

                    if ele_tag_name == 'button' and ele_type == 'submit':
                        time.sleep(10)
                    if 'target' in cur_website:
                        time.sleep(10)

                elif action_key == 'wait':
                    time.sleep(5)
                    if 'target' in cur_website:
                        time.sleep(10)

                elif action_key == 'type':
                    type_ele_number = int(info['number'])
                    element_box = obs_info[type_ele_number]['union_bound']
                    element_box = calculate_intersection(args, element_box)
                    element_box_center = (element_box[0] + element_box[2] // 2,
                                          element_box[1] + element_box[3] // 2)
                    web_ele = driver_task.execute_script(
                        "return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0],
                        element_box_center[1])
                    web_ele_attr_info = f"Web Element Info; <role: '{obs_info[type_ele_number]['role']['value']}'; name: '{obs_info[type_ele_number]['name']['value']}'>"
                    print(web_ele_attr_info)
                    warn_obs = exec_action_type(info, web_ele, driver_task)
                    if 'wolfram' in task['web']:
                        time.sleep(5)
                    if 'target' in cur_website:
                        time.sleep(10)

                elif action_key == 'scroll':
                    exec_action_scroll(info, None, driver_task, args, obs_info)

                elif action_key == 'goback':
                    driver_task.back()
                    time.sleep(2)

                elif action_key == 'restart':
                    # break
                    driver_task.get(cur_website)
                    time.sleep(2)

                elif action_key == 'answer':
                    logging.info(info['content'])
                    logging.info('finish!!')
                    break

                else:
                    raise NotImplementedError
                fail_obs = ""
            except Exception as e:
                logging.error('driver error info:')

                print(e)
                logging.error(e)
                if 'element click intercepted' not in str(e):
                    fail_obs = "The action you have chosen cannot be executed. Please double-check if you have selected the correct element or used correct action format. Then provide the revised Thought and Action."
                else:
                    fail_obs = ""
                time.sleep(2)

        print_message(messages, task_dir, is_v=True)
        driver_task.quit()
        if not args.api_localhost:
            logging.info(
                f'Total cost: {accumulate_prompt_token / 1000 * 0.01 + accumulate_completion_token / 1000 * 0.03}')


if __name__ == '__main__':
    main()
    print('End of process')

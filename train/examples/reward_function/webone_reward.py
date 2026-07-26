# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
from typing import Any

import json
import time
import requests
import base64
from openai import OpenAI

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import string

SYSTEM_PROMPT_FINAL = """As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'NOT SUCCESS'."""
USER_PROMPT_FINAL = """TASK: <task>
Result Response: <answer>
<num> screenshots at the end: """

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""), base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))

# 初始化全局工具
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """预处理文本：小写化、去除标点"""
    if isinstance(text, list):
        text = ' '.join(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def get_tokens(text):
    """获取单词列表"""
    return text.split()

def get_stems(tokens):
    """获取词干形式"""
    return [stemmer.stem(token) for token in tokens]

def get_lemmas(tokens):
    """获取词元形式"""
    return [lemmatizer.lemmatize(token) for token in tokens]


def exact_match_ratio(source, target):
    """精确匹配比例"""
    source_tokens = set(get_tokens(preprocess_text(source)))
    target_tokens = set(get_tokens(preprocess_text(target)))

    if not source_tokens:
        return 0.0

    intersection = source_tokens & target_tokens
    return len(intersection) / len(source_tokens)


def stem_match_ratio(source, target):
    """词干匹配比例"""
    source_tokens = get_tokens(preprocess_text(source))
    target_tokens = get_tokens(preprocess_text(target))

    source_stems = set(get_stems(source_tokens))
    target_stems = set(get_stems(target_tokens))

    if not source_stems:
        return 0.0

    intersection = source_stems & target_stems
    return len(intersection) / len(source_stems)


def lemma_match_ratio(source, target):
    """词元匹配比例"""
    source_tokens = get_tokens(preprocess_text(source))
    target_tokens = get_tokens(preprocess_text(target))

    source_lemmas = set(get_lemmas(source_tokens))
    target_lemmas = set(get_lemmas(target_tokens))

    if not source_lemmas:
        return 0.0

    intersection = source_lemmas & target_lemmas
    return len(intersection) / len(source_lemmas)

def extract_patterntext(result, label):
    match = re.search(r'<%s>(.*?)</%s>' % (label, label), result, re.DOTALL)

    if match:
        extracted_text = match.group(1).strip()

        new_txt = ''
        for line in extracted_text.split("\n"):
            if not line.startswith('#'):
                new_txt += line.strip()
        extracted_text = new_txt
    else:
        extracted_text = None
    return extracted_text



def extract_information(text):
    patterns = {
        "click": r"Click \[?(\d+)\]?",
        "clear": r"Clear \[?(\d+)\]?",
        "type":  r"Type \[?(\d+)\]?[; ]+\[?(.[^\]]*)\]?",
        # "delete_and_type": r"Delete_and_Type \[?(\d+)\]?[; ]+\[?(.[^\]]*)\]?",
        "scroll": r"Scroll \[?(\d+|WINDOW)\]?[; ]+\[?(up|down)\]?",
        "wait": r"^Wait",
        "goback": r"^GoBack",
        "restart": r"^Restart",
        "answer": r"ANSWER[;: ]+\[?(.[^\]]*)\]?"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            if key in ["click", "wait", "goback", "restart", "clear"]:
                # no content
                return key, match.groups()
            else:
                return key, {"number": match.group(1), "content": match.group(2)} if key in ["type", "scroll"] else {"content": match.group(1)}
    return None, None


def format_reward(response: str) -> float:
    # pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    # format_match = re.fullmatch(pattern, response)

    reference = extract_patterntext(response, "reference")
    think = extract_patterntext(response, "think")
    action_desc = extract_patterntext(response, "action_desc")
    action = extract_patterntext(response, "action")

    if think is None or reference is None or action_desc is None or action is None:
        return 0.0

    return 1.0

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def wrap_current_messages(obs, screenshot_path):

    history_message = []

    b64_img = encode_image(screenshot_path)

    cur_observe = f"The Current Page:\nplease analyze the attached Accessibility Tree & screenshot and complete the current subtask.\n{obs}\nScreenshot:\n"

    history_message.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": cur_observe
            },
            {

                "type": "image_url",
                'image_url': {"url": f"data:image/png;base64,{b64_img}"}

            }
        ],
    })

    return history_message

def call_gpt4o_api(messages, api_model="gpt-4o"):

    error_nums = 0
    while True:
        try:
            print('Calling gpt4v API to get the auto evaluation......')
            openai_response = openai_client.chat.completions.create(
                model=api_model, messages=messages, max_tokens=1000, seed=42, temperature=0
            )
            # print('Prompt Tokens:', openai_response.usage.prompt_tokens, ';',
            #       'Completion Tokens:', openai_response.usage.completion_tokens)
            # print('Cost:', openai_response.usage.prompt_tokens / 1000 * 0.01
            #       + openai_response.usage.completion_tokens / 1000 * 0.03)

            print('API call complete...')
            break
        except Exception as e:
            error_nums += 1
            if error_nums > 3:
                raise ValueError(str(e))
            print(e)
            if type(e).__name__ == 'RateLimitError':
                time.sleep(10)
            elif type(e).__name__ == 'APIError':
                time.sleep(15)
            elif type(e).__name__ == 'InvalidRequestError':
                exit(0)
            elif e.code == 'content_filter':
                print('Content filter!!!!!!!')
                return [{'content': 'content_filter', 'success': 0}]
            else:
                time.sleep(10)
    gpt_4v_res = openai_response.choices[0].message.content

    return gpt_4v_res

def call_gemini_api(messages):
    retry_times = 0
    while True:
        try:

            payload = json.dumps({
                "contents": messages,
                }
            )
            headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + os.environ.get("WEBONE_GEMINI_API_KEY", ""),
            }
            url = os.environ.get("WEBONE_GEMINI_URL", "http://localhost:8000/v1beta/models/gemini-2.5-pro:generateContent")

            response = requests.request("POST", url, headers=headers, data=payload)

            print(response.text)
            openai_response = json.loads(response.content.decode('utf-8'))
            # openai_response = openai_client.chat.completions.create(
            #     model=args.api_model, messages=messages, max_tokens=1000, seed=args.seed
            # )
            prompt_tokens = openai_response["usageMetadata"]['promptTokenCount']
            completion_tokens = openai_response["usageMetadata"]['candidatesTokenCount']


            gpt_call_error = False
            return prompt_tokens, completion_tokens, gpt_call_error, openai_response

        except Exception as e:
            print('error in calling Gemini API')
            print(e)
            time.sleep(15)
            if type(e).__name__ == 'RateLimitError':
                time.sleep(10)

            elif type(e).__name__ == 'APIError':
                time.sleep(15)


        retry_times += 1
        if retry_times == 100:
            print('Retrying too many times')
            return None, None, True, None


def call_actor(cur_task, obs, screenshot_path):

    hisitory_message = wrap_current_messages(obs, screenshot_path)

    text_prompt = f'''Imagine you are a robot browsing the web, just like humans. Now you need to complete a provided instruction. In current step, you have received an Observation that includes a screenshot and an accessibility tree of a webpage.
Carefully analyze the observation to identify the Numerical Label (in the Accessibility Tree) corresponding to the Web Element that requires interaction, then follow the guidelines and choose one of the following actions to complete the specific subtask:
1. Click a Web Element.
2. Delete existing content in a textbox and then type content (use the clear action). 
3. Scroll up or down. Multiple scrolls are allowed to browse the webpage. Pay attention!! The default scroll is the whole window. If the scroll widget is located in a certain area of the webpage, then you have to specify a Web Element in that area. I would hover the mouse there and then scroll.
4. Wait. Typically used to wait for unfinished webpage processes, with a duration of 5 seconds.
5. Go back, returning to the previous webpage (Deliberately).
6. Restart, directly jump to the beginning page. When you can't complete the task, try starting over from the beginning (Deliberately).
7. Answer. This action should only be chosen when the global task have been solved and should stop. (If the current subtask states that the global task has been completed, please summarize and answer to the global task as required.)

Correspondingly, Action should STRICTLY follow the format:
- Click [Numerical_Label]
- Type [Numerical_Label]; [Content]
- Clear [Numerical_Label]
- Scroll [Numerical_Label or WINDOW]; [up or down]
- Wait
- GoBack
- Restart
- ANSWER; [content]

Key Guidelines You MUST follow:
* Action guidelines *
1) Type [Numerical_Label]; [Content]: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0, i.e., Type [Numerical_Label]; [Content]; [0]. (When you need to input multiple fields consecutively, you may need to set the argument to 0 until the last input)
2) To input text, Don't click on textbox first, directly select Type action. After typing, the system automatically hits `ENTER` key. Sometimes you should click the search button to apply search filters. Try to use simple language when searching.  
3) You must Distinguish between textbox and search button, don't type content into the button! If no textbox is found, you may need to click the search button first before the textbox is displayed. 
4) Execute only one action per iteration.
5) When a complex global Task involves multiple questions or steps, select "ANSWER" only at the very end, after addressing all of these questions (steps). Flexibly combine your own abilities with the information in the web page. Double check the formatting requirements in the global task when ANSWER. 
* Web Browsing Guidelines *
1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages. Pay attention to Key Web Elements like search textbox and menu.
2) Vsit video websites like YouTube is allowed BUT you can't play videos. Clicking to download PDF is allowed and will be analyzed by the Assistant API.
3) Focus on the date in the sub-task, you must look for results that match the date. It may be necessary to find the correct year, month and day at calendar.
4) Pay attention to the filter and sort functions on the page, which, combined with scroll, can help you solve conditions like 'highest', 'cheapest', 'lowest', 'earliest', etc. Try your best to find the answer that best fits the task.

The Global task has been divided into several subtasks. Now, you are required to complete the following subtask:
# The Current SUBTASK:
{cur_task}

Your reply should strictly follow the format:
<think>Your_brief_thoughts_(briefly_summarize_the_info_that_will_help_ANSWER)</think>
<action>One_Action_format_you_choose</action>
'''

    message = [
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": text_prompt
            }]
        }
    ]

    num_try = 0
    while num_try <= 3:
        num_try += 1
        raw_output = call_gpt4o_api(hisitory_message + message)
        try:
            # raw_output = openai_response['candidates'][0]['content']['parts'][0]['text']
            action = extract_patterntext(raw_output, "action")
            if action is not None:
                return raw_output
        except:
             print(f"Exception in call the actor")


    return ""



def llm_as_judge_final_state(global_task, prediction_result, image_path_list):

    ans_info = prediction_result
    if isinstance(ans_info, str):
        ans_info = ans_info.replace("Action:ANSWER", "Action: ANSWER")

    pattern_ans = r"ANSWER[;:\n ]+\[?(.[^\]]*)\]?"
    matches_ans = re.search(pattern_ans, ans_info)
    answer_content = matches_ans.group(1).strip()

    whole_content_img = []
    for png_file in image_path_list[-5:]:
        b64_img = encode_image(png_file)
        whole_content_img.append(
            {
                'type': 'image_url',
                'image_url': {"url": f"data:image/png;base64,{b64_img}"}
            }
        )

    user_prompt_tmp = USER_PROMPT_FINAL.replace('<task>', global_task)
    user_prompt_tmp = user_prompt_tmp.replace('<answer>', answer_content)
    user_prompt_tmp = user_prompt_tmp.replace('<num>', str(len(image_path_list)))
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT_FINAL},
        {
            'role': 'user',
            'content': [
                           {'type': 'text', 'text': user_prompt_tmp}
                       ]
                       + whole_content_img
                       + [{'type': 'text', 'text': "Your verdict:\n"}]
        }
    ]

    num_try = 0
    while num_try <= 3:
        num_try += 1
        raw_output = call_gpt4o_api(messages)
        print(f"pred output: {prediction_result};")
        try:
            # raw_output = openai_response['candidates'][0]['content']['parts'][0]['text']
            auto_eval_res = 0 if 'NOT SUCCESS' in raw_output else 1
            if 'SUCCESS' not in raw_output:
                auto_eval_res = 0  # None
            print(f"The final verdict: {raw_output}")
            return auto_eval_res

        except:
            print(f"Exception in final llm verdict")

    return 0.0


def llm_as_judge_process(history_message, global_task, checklist, obs, think, next_task, screenshot_path):

    cur_turn = wrap_current_messages(obs, screenshot_path)

    text_prompt = f"""You are an expert evaluator of web agent. Your task is to assess how helpful a given agent’s THOUGHT and ACTION is in making progress toward the user’s goal, based on the trajectory (given through previous messages), current state of the webpage, and the provided checklist.
# Action space: [Click, Type, Clear, Scroll, Wait, GoBack, Restart, ANSWER]

# Task Description
Evaluate how helpful the given thought and action is for achieving the goal.
Use the following scale:
**Scoring Criteria (1 to 5):**
- **5 (Very Helpful)**: The action directly and effectively moves toward fulfilling a key part
of the goal (e.g. a step in the checklist, or recovering from the failure).
- **4 (Helpful)**: The action contributes meaningfully to progress, though it may require
follow-up actions (e.g. a partial step in the checklist, or recovering from the failure).
- **3 (Somewhat Helpful)**: The action is partially relevant or a preparatory step, but doesn’t
make immediate progress.
- **2 (Slightly Helpful)**: The action is weakly related to the goal or might only indirectly
help.
- **1 (Not Helpful)**: The action is unrelated, redundant, or distracts from the goal.

NOTE: If you are in an erroneous or failed step/state, you should determine whether the current "think and action" description can help you recover from the failure and return to the normal path (where the normal path refers to the checklist). If the answer is true, then a score of 4 or 5 should be given.

# Given Information
## User Instruction (Global goal):
{global_task}

## Checklist (Detailed guidebooks to complete the global goal):
{checklist}

## Agent Response:

THOUGHT:
{think}

ACTION Description:
{next_task}

# Output Format
Please return your response in the following format:
<reason>your_explanation_for_the_score</reason>
<score>your_score_[1-5]</score>
"""

    message = [
        {
            "role": "user",
            "parts": [{
                "text": text_prompt
            }]
        }
    ]
    num_try = 0
    while num_try <= 3:
        num_try += 1
        prompt_tokens, completion_tokens, call_error, openai_response = call_gemini_api(history_message + cur_turn + message)

        try:
            raw_output = openai_response['candidates'][0]['content']['parts'][0]['text']
            print(f"llm as judge: {raw_output}")
            score = extract_patterntext(raw_output, "score")
            reason = extract_patterntext(raw_output, "reason")
            if '5' in score or '4' in score:
                return 1.0
            elif '3' in score or '2' in score or '1' in score:
                return 0.0
            elif num_try >= 3:
                return 0.0
        except:
            print(f"Exception in extracting patterntext in llm as judge in the process")

    return 0.0


def match_reference(pred_reference, gt_reference):
    if pred_reference is None:
        return 0.0
    if gt_reference is None:
        return 1.0

    pred_reference = pred_reference.lower()
    gt_reference = gt_reference.lower()

    if 'none' in gt_reference:
        if "none" in pred_reference.lower():
            return 1.0
        else:
            return 0.0

    if 'recovery' in gt_reference:
        if 'recovery' not in pred_reference:
            return 0.5
        else:
            return 1.0

    if gt_reference in pred_reference:
        return 1.0
    if gt_reference.replace('#', ' ') in pred_reference:
        return 1.0
    if gt_reference.replace(' ', '#') in pred_reference:
        return 1.0

    return 0.0


def match_groundtruth(response, ground_truth):
    # think = extract_patterntext(response, "think")
    #
    # action = extract_patterntext(response, "action")
    # if action is None:
    #     return 0.0
    action = response

    keyy, args = extract_information(action)
    target_keyy, target_args = extract_information(ground_truth)

    if keyy is None:
        return 0.0

    if keyy in ["click", "clear"]:
        if keyy == target_keyy and args[0] == target_args[0]:
            return 1.0
        else:
            return 0.0
    if keyy in ["wait", "goback", "restart", "answer"]:
        if keyy == target_keyy:
            return 1.0
        else:
            return 0.0
    if keyy == "type":
        if keyy != target_keyy:
            return 0.0
        target_number = target_args['number'] if 'number' in target_args else -1
        target_content = target_args['content']
        pred_number = args['number'] if 'number' in args else -1
        pred_content = args['content']
        if target_number == pred_number:
            exact_ratio = exact_match_ratio(pred_content, target_content)
            stem_ratio = stem_match_ratio(pred_content, target_content)
            lemma_ratio = lemma_match_ratio(pred_content, target_content)
            if exact_ratio > 0.5 or stem_ratio > 0.5 or lemma_ratio > 0.5:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0
    if keyy == "scroll":
        if keyy != target_keyy:
            return 0.0
        target_number = target_args['number'] if 'number' in target_args else -1
        target_content = target_args['content']
        pred_number = args['number'] if 'number' in args else -1
        pred_content = args['content']
        if pred_content.lower() != target_content.lower():
            return 0.0
        else:
            return 1.0

    return 0.0

def accuracy_reward(response: str, ground_truth: str) -> float:

    ground_truth = json.loads(ground_truth)
    ground_truth_output = ground_truth['gt_action']
    meta_info = ground_truth['meta_info']

    cur_observation = meta_info['obs']
    cur_image = meta_info['image']
    global_task = meta_info['task']
    history_message = meta_info['history_messages']
    history_image_list = meta_info['history_images']
    checklist = meta_info['checklist']
    golden_result = meta_info['gt_result']

    gt_action = extract_patterntext(ground_truth_output, "action")
    gt_subtask = extract_patterntext(ground_truth_output, "subtask").lower()

    pred_reference = extract_patterntext(response, "reference")
    pred_think = extract_patterntext(response, "think")
    pred_action_desc = extract_patterntext(response, "action_desc")
    pred_action = extract_patterntext(response, "action")

    if pred_action is None:
        return 0.0

    # if the final state
    if "ANSWER" in pred_action and gt_action == '': # gt_action == '' 表示这是最后一步
        score = llm_as_judge_final_state(global_task, pred_action, history_image_list)
        score = max(score, 0.5) # 虽然没答对，但是知道这一步需要终止，还是给0.3分
        return score
    if gt_action == '' and "ANSWER" not in pred_action:
        return 0.0 #当前步需要终止；但是没有终止
    if "ANSWER" in pred_action and gt_action != '': # gt_action != '' 表示这不是最后一步
        return 0.0 #0.5 * llm_as_judge_final_state(global_task, pred_action, history_image_list) # 当前步还未到终止步；即使答对，给0.5

    # 比较 actor生成的动作 和 ground_truth 动作
    action_reward = match_groundtruth(pred_action, gt_action)
    reference_reward = match_reference(pred_reference, gt_subtask)
    print(f"pred_reference: {pred_reference}; gt_subtask: {gt_subtask}; {reference_reward}")
    action_reward = action_reward * 0.8 + reference_reward * 0.2
    return action_reward

    # if action_reward == 1.0:
    #     return 1.0
    #
    # # if action reward is NOT positive
    # # llm_score = llm_as_judge_process(history_message, global_task, checklist, cur_observation, pred_think, pred_next, cur_image)
    # # return llm_score
    #
    # return 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        # print(reward_input["response"])
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores

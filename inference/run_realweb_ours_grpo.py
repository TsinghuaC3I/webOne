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
from utils.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_TEXT_ONLY, SYSTEM_PROMPT_ACTREE_IMAGE_GUIDEBOOK
from openai import OpenAI
from utils.utils import get_web_element_rect, encode_image, extract_information, print_message, \
    get_webarena_accessibility_tree, get_pdf_retrieval_ans_from_assistant, clip_message_and_obs, \
    clip_message_and_obs_text_only
import copy
from utils.utils_localhost import retry_with_exponential_backoff, convert_msg_format_openai_to_localhost
import json
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor

System_head = "### Task Description: Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, you will receive an Observation that includes a screenshot and an accessibility tree of a webpage.\nPlease, learn useful information (e.g., plan, actions, UI elements, etc.) from the provided guidebook above, while ignoring any details irrelevant to the current task.\n\n### Context Description: I will now provide you with the task context and detailed requirements. After receiving all the information, please follow the instructions to complete the task:\n\n1. I will provide you with a potentially useful guidebook. From this, you may learn the task workflow, actions for each step, how to recover from errors, and the criteria for determining termination states.\n\n2. I will provide you with a step-by-step stream of historical operation data.\n\n3. I will provide you with the outputs from the last step's observation, planning model, and action model.\n\n4. I will provide you with the current step's observation and specific instruction. Please follow the instructions to complete the task.\n\n### Current Task:\n"

HEAD = """Carefully analyze the provided guidebook, history trajectory, last and current observation to choose one of the following actions to complete the specific step:
    1. Click a Web Element.
    2. Delete existing content in a textbox and then type content (use the clear action). 
    3. Scroll up or down. Multiple scrolls are allowed to browse the webpage. Pay attention!! The default scroll is the whole window. If the scroll widget is located in a certain area of the webpage, then you have to specify a Web Element in that area. I would hover the mouse there and then scroll.
    4. Wait. Typically used to wait for unfinished webpage processes, with a duration of 5 seconds.
    5. Go back, returning to the previous webpage (Deliberately).
    6. Restart, directly jump to the beginning page. When you can't complete the task, try starting over from the beginning (Deliberately).
    7. Answer. This action should only be chosen when the global task have been solved and should stop. (If the current subtask states that the global task has been completed, please summarize and answer to the global task as required.)


* Task Tips *
1) You should CAREFULLY consider the provided guidebook, historical trajectory, last-turn observation and result, and the current turn's observation, then generate output STRICTLY following the given `Output Format`.
2) You should detect meaningful changes between pre- and post-screenshots. Unexpected states (errors, timeouts, missing elements) can be considered as the previous action being a \"failure\". STRICTLY avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait action is also NOT allowed.
3) You should refer to the provided guidebook and determine which subtask (in the guidebook) the current step relates to.
4) The action types in action_desc and action should be within [Click, Clear, Type, Scroll, Wait, GoBack, Restart, ANSWER]. Avoid generating navigation or location actions, as they do not align with any specified step above; directly click on the desired link instead.
5) IMPORTANT: Sometimes you need to perform reasoning and thinking. You should include your reasoning about the information provided in the guidebook, historical trajectory, and last and current state in the \"thinking\" section. This should be completed before outputting the current subtask, action_desc, and action.
6) Do not redo the same actions from previous steps if they have already been completed. Do not output any comments.
7) Please refer to the guidebooks. Check if the current UI state matches the main goal's success criteria. Select the action \"ANSWER;\" when you believe you have completed the global task, and summarize and respond to this task in the format: <action_desc>ANSWER; any information related to the answer</action_desc>\n<action>ANSWER; your answer</action>

* Action Tips *
1) Type [Numerical_Label]; [Content]: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0, i.e., Type [Numerical_Label]; [Content]; [0]. (When you need to input multiple fields consecutively, you may need to set the argument to 0 until the last input)
2) To input text, Don't click on textbox first, directly select Type action. After typing, the system automatically hits `ENTER` key. Sometimes you should click the search button to apply search filters. Try to use simple language when searching.  
3) You must Distinguish between textbox and search button, don't type content into the button! If no textbox is found, you may need to click the search button first before the textbox is displayed. 
4) Execute only one action per iteration.
5) When a complex global Task involves multiple questions or steps, select \"ANSWER\" only at the very end, after addressing all of these questions (steps). Flexibly combine your own abilities with the information in the web page. Double check the formatting requirements in the global task when ANSWER. 

* Web Browsing Tips *
1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages. Pay attention to Key Web Elements like search textbox and menu.
2) Vsit video websites like YouTube is allowed BUT you can't play videos. Clicking to download PDF is allowed and will be analyzed by the Assistant API.
3) Focus on the date in the sub-task, you must look for results that match the date. It may be necessary to find the correct year, month and day at calendar.
4) Pay attention to the filter and sort functions on the page, which, combined with scroll, can help you solve conditions like 'highest', 'cheapest', 'lowest', 'earliest', etc. Try your best to find the answer that best fits the task.

Now, the current Observation(Screenshot, Accessibility tree), global task, the last step are provided:

"""

OUTPUT_INS = "* Output Instructions: *\n\n1. Think: You are required to carefully analyze the provided guidebook, history trajectory, and the last and current turn states, which helps current step actions or the final answers.\n\n2. Reference: You are required to refer to the provided guidebook and determine which subtask (in the guidebook) the current step relates to, and use this as the current reference. You should output in the format `<reference>Subtask#[id]: content_of_the_subtask_or_your_think</reference>`. If the current state is in the process of recovering from an error, you do not need to find reference from the guidebook; simply output `recovery`.\n\n3. Action_desc: Based on the provided guidebook, screenshot, observations, and historical information:\n    3.1 If in an intermediate step, you are permitted to output only the current ONE STEP (in a sentence) to solve the task based on the information below.\n    3.2 If you have completed the global task, please provide any information relevant to your answer, or summarize your thoughts and actions to complete the task.\n\n4. Action: Action should STRICTLY follow the format:\n    4.1 Click [Numerical_Label]\n    4.2 Type [Numerical_Label]; [Content]\n    4.3 Clear [Numerical_Label]\n    4.4 Scroll [Numerical_Label or WINDOW]; [up or down]\n    4.5 Wait\n    4.6 GoBack\n    4.7 Restart\n    4.8 ANSWER; [content]\n\n\nIMPORTANT!!! * Output Format: *\n```\n<think>your_think_about_the_guidebook_history_trajectory_last_and_current_state</think>\n<reference>reference_in_the_guidebook(subtask#[id])</reference>\n<action_desc>one_sentence_describe_your_action_towards_the_global_task</action_desc>\n<action>your_action</action>\n```\n"

TARGET_SOFTWARE_TIPS = """
* Task Tips *
1) You should CAREFULLY consider the provided guidebook, historical trajectory, last-turn observation and result, and the current turn's observation, then generate output STRICTLY following the given `Output Format`.
2) You should detect meaningful changes between pre- and post-screenshots. Unexpected states (errors, timeouts, missing elements) can be considered as the previous action being a "failure". STRICTLY avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait action is also NOT allowed.
3) You should refer to the provided guidebook and determine which subtask (in the guidebook) the current step relates to.
4) The action types in action_desc and action should be within [Click, Clear, Type, Scroll, Wait, GoBack, Restart, ANSWER]. Avoid generating navigation or location actions, as they do not align with any specified step above; directly click on the desired link instead.
5) IMPORTANT: Sometimes you need to perform reasoning and thinking. You should include your reasoning about the information provided in the guidebook, historical trajectory, and last and current state in the "thinking" section. This should be completed before outputting the current subtask, action_desc, and action.
6) Do not redo the same actions from previous steps if they have already been completed. Do not output any comments.
7) Please refer to the guidebooks. Check if the current UI state matches the main goal's success criteria. Select the action "ANSWER;" when you believe you have completed the global task, and summarize and respond to this task in the format: <action_desc>ANSWER; any information related to the answer</action_desc>\n<action>ANSWER; your answer</action>

* Action Tips *
1) Type [Numerical_Label]; [Content]: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0, i.e., Type [Numerical_Label]; [Content]; [0]. (When you need to input multiple fields consecutively, you may need to set the argument to 0 until the last input)
2) To input text, Don't click on textbox first, directly select Type action. After typing, the system automatically hits `ENTER` key. Sometimes you should click the search button to apply search filters. Try to use simple language when searching.  
3) You must Distinguish between textbox and search button, don't type content into the button! If no textbox is found, you may need to click the search button first before the textbox is displayed. 
4) Execute only one action per iteration.
5) When a complex global Task involves multiple questions or steps, select "ANSWER" only at the very end, after addressing all of these questions (steps). Flexibly combine your own abilities with the information in the web page. Double check the formatting requirements in the global task when ANSWER. 

* Web Browsing Tips *
1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages. Pay attention to Key Web Elements like search textbox and menu.
2) Vsit video websites like YouTube is allowed BUT you can't play videos. Clicking to download PDF is allowed and will be analyzed by the Assistant API.
3) Focus on the date in the sub-task, you must look for results that match the date. It may be necessary to find the correct year, month and day at calendar.
4) Pay attention to the filter and sort functions on the page, which, combined with scroll, can help you solve conditions like 'highest', 'cheapest', 'lowest', 'earliest', etc. Try your best to find the answer that best fits the task.

Now, the current Observation(Screenshot, Accessibility tree), global task, the last step are provided:
"""

TARGET_OUTPUT_FORMAT = """
* Output Instructions: *

1. Think: You are required to carefully analyze the provided guidebook, history trajectory, and the last and current turn states, which helps current step actions or the final answers.

2. Reference: You are required to refer to the provided guidebook and determine which subtask (in the guidebook) the current step relates to, and use this as the current reference. You should output in the format `<reference>Subtask#[id]: content_of_the_subtask_or_your_think</reference>`. If the current state is in the process of recovering from an error, you do not need to find reference from the guidebook; simply output `recovery`.

3. Action_desc: Based on the provided guidebook, screenshot, observations, and historical information:
    3.1 If in an intermediate step, you are permitted to output only the current ONE STEP (in a sentence) to solve the task based on the information below.
    3.2 If you have completed the global task, please provide any information relevant to your answer, or summarize your thoughts and actions to complete the task.

4. Action: Action should STRICTLY follow the format:
    4.1 Click [Numerical_Label]
    4.2 Type [Numerical_Label]; [Content]
    4.3 Clear [Numerical_Label]
    4.4 Scroll [Numerical_Label or WINDOW]; [up or down]
    4.5 Wait
    4.6 GoBack
    4.7 Restart
    4.8 ANSWER; [content]


IMPORTANT!!! * Output Format: *
```
<think>your_think_about_the_guidebook_history_trajectory_last_and_current_state</think>
<reference>reference_in_the_guidebook(subtask#[id])</reference>
<action_desc>one_sentence_describe_your_action_towards_the_global_task</action_desc>
<action>your_action</action>
```
"""

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


def get_history_output(model_output_list):

    text = "You are provided with a step-by-step stream of history operation data.\n\n"

    for ii, model_output in enumerate(model_output_list):
        cur_think = extract_patterntext(model_output, 'think')
        cur_action = extract_patterntext(model_output, 'action')
        cur_next = extract_patterntext(model_output, 'action_desc')

        text += f"### Step {ii + 1}:\n- The No.{ii + 1} Page:Observation: A screenshot and some texts (Omitted in context.), and the outputs of the planning and action model are as follows:\n"
        text += f"- Planning Model Output:\n<think>{cur_think}</think><<next>{cur_next}</next>\n- Action Model Output:\n<action>{cur_action}</action>\n\n"

    return text.strip()


def get_last_prompt(last_obs, model_output, iter):
    cur_think = extract_patterntext(model_output, 'think')
    cur_action = extract_patterntext(model_output, 'action')
    cur_next = extract_patterntext(model_output, 'action_desc')
    text = f"\n- The No.{iter} Page:{last_obs} \n\n\nThe screenshot for this step is provided above, and the outputs of the planning and action model are as follows:\n- Planning Model Output:\n<think>{cur_think}</think><next>{cur_next}</next>\n- Action Model Output:\n<action>{cur_action}</action>\n"

    return text.strip()


#"Carefully analyze the provided guidebook, history trajectory, last and current observation to choose one of the following actions to complete the specific step:\n    1. Click a Web Element.\n    2. Delete existing content in a textbox and then type content (use the clear action). \n    3. Scroll up or down. Multiple scrolls are allowed to browse the webpage. Pay attention!! The default scroll is the whole window. If the scroll widget is located in a certain area of the webpage, then you have to specify a Web Element in that area. I would hover the mouse there and then scroll.\n    4. Wait. Typically used to wait for unfinished webpage processes, with a duration of 5 seconds.\n    5. Go back, returning to the previous webpage (Deliberately).\n    6. Restart, directly jump to the beginning page. When you can't complete the task, try starting over from the beginning (Deliberately).\n    7. Answer. This action should only be chosen when the global task have been solved and should stop. (If the current subtask states that the global task has been completed, please summarize and answer to the global task as required.)\n    \n\n* Task Tips *\n1) You should CAREFULLY consider the provided guidebook, historical trajectory, last-turn observation and result, and the current turn's observation, then generate output STRICTLY following the given `Output Format`.\n2) You should detect meaningful changes between pre- and post-screenshots. Unexpected states (errors, timeouts, missing elements) can be considered as the previous action being a \"failure\". STRICTLY avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait action is also NOT allowed.\n3) You should refer to the provided guidebook and determine which subtask (in the guidebook) the current step relates to.\n4) The action types in action_desc and action should be within [Click, Clear, Type, Scroll, Wait, GoBack, Restart, ANSWER]. Avoid generating navigation or location actions, as they do not align with any specified step above; directly click on the desired link instead.\n5) IMPORTANT: Sometimes you need to perform reasoning and thinking. You should include your reasoning about the information provided in the guidebook, historical trajectory, and last and current state in the \"thinking\" section. This should be completed before outputting the current subtask, action_desc, and action.\n6) Do not redo the same actions from previous steps if they have already been completed. Do not output any comments.\n7) Please refer to the guidebooks. Check if the current UI state matches the main goal's success criteria. Select the action \"ANSWER;\" when you believe you have completed the global task, and summarize and respond to this task in the format: <action_desc>ANSWER; any information related to the answer</action_desc>\n<action>ANSWER; your answer</action>\n\n* Action Tips *\n1) Type [Numerical_Label]; [Content]: Use this to type the content into the field with id. By default, the \"Enter\" key is pressed after typing unless press_enter_after is set to 0, i.e., Type [Numerical_Label]; [Content]; [0]. (When you need to input multiple fields consecutively, you may need to set the argument to 0 until the last input)\n2) To input text, Don't click on textbox first, directly select Type action. After typing, the system automatically hits `ENTER` key. Sometimes you should click the search button to apply search filters. Try to use simple language when searching.  \n3) You must Distinguish between textbox and search button, don't type content into the button! If no textbox is found, you may need to click the search button first before the textbox is displayed. \n4) Execute only one action per iteration.\n5) When a complex global Task involves multiple questions or steps, select \"ANSWER\" only at the very end, after addressing all of these questions (steps). Flexibly combine your own abilities with the information in the web page. Double check the formatting requirements in the global task when ANSWER. \n\n* Web Browsing Tips *\n1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages. Pay attention to Key Web Elements like search textbox and menu.\n2) Vsit video websites like YouTube is allowed BUT you can't play videos. Clicking to download PDF is allowed and will be analyzed by the Assistant API.\n3) Focus on the date in the sub-task, you must look for results that match the date. It may be necessary to find the correct year, month and day at calendar.\n4) Pay attention to the filter and sort functions on the page, which, combined with scroll, can help you solve conditions like 'highest', 'cheapest', 'lowest', 'earliest', etc. Try your best to find the answer that best fits the task.\n\nNow, the current Observation(Screenshot, Accessibility tree), global task, the last step are provided:\nThe No.1 Page:[1] RootWebArea 'Cambridge Dictionary | English Dictionary, Translations & Thesaurus' \n\t[2] button 'Close autocomplete'\n\t\t[4] link 'Dictionary' \n\t\t[5] link 'Translate' \n\t\t[6] link 'Grammar' \n\t\t[7] link 'Thesaurus' \n\t\t[8] link '+Plus' \n\t\t[9] link 'Games' \n\t\t[10] link 'Shop \\uf35d' \n\t\t[11] link '\\uf09a' \n\t\t[12] link '\\uf16d' \n\t\t[14] StaticText 'Log in'\n\t\t[16] StaticText 'Sign up'\n\t[19] main 'Close header popups'\n\t\t[21] StaticText 'Make your words meaningful'\n\t\t[23] button '\\uf00d'\n\t\t[24] button 'Choose a dictionary'\n\t\t\t[25] StaticText 'English'\n\t\t[26] button 'Search'\n\t\t[29] button 'Set dictionary search to Spanish–English'\n\t\t[31] heading 'Explore the Cambridge Dictionary'\n\t\t[32] heading 'English dictionaries'\n\t\t[33] link 'English' \n\t\t\t[38] link 'Grammar' \n\t\t[39] image 'Plus promo' \n\t\t[43] link 'Go to + Plus' \n\t\t[46] generic 'These are topics related to the article that might interest you'\n\t\t[48] link 'Online Language Tutors and Mentors'\n\t\t\t[53] link 'More information about your privacy, opens in a new tab' \n\t\t\t\t[54] StaticText 'Privacy and Cookies Policy'\n\t\t\t[55] button 'Do Not Sell My Personal Information, Opens the preference center dialog'\n\t\t\t[56] button 'Accept Cookies'\n\t\t\t[57] button 'Close'\n\n\nThe screenshot for this step is provided above.\n\nPlease, First learn useful information (e.g., plan, actions, UI elements, etc.) from the provided guidebook above, while ignoring any details irrelevant to the current task. \nThen, based on the current (The No.1 Page) and last (The No.0 Page) observations and their screenshots, Output the status and the next suggestions based on the below information. \n\nGlobal Task (the final objective you need to complete step by step.):\nNow given a task: Navigate to the Cambridge Dictionary's blog section and summarize the main points from a recent article about language learning.  Please interact with https://dictionary.cambridge.org/ and get the answer. (Accept or close all cookies. Close all irrelevant pop-ups that appear during this process.)\n\n\nLast Step (the last step you executed in the previous step, awaiting judgment):\nNone (Currently, it is the first step with no previous step.)\n\n* Output Instructions: *\n\n1. Think: You are required to carefully analyze the provided guidebook, history trajectory, and the last and current turn states, which helps current step actions or the final answers.\n\n2. Reference: You are required to refer to the provided guidebook and determine which subtask (in the guidebook) the current step relates to, and use this as the current reference. You should output in the format `<reference>Subtask#[id]: content_of_the_subtask_or_your_think</reference>`. If the current state is in the process of recovering from an error, you do not need to find reference from the guidebook; simply output `recovery`.\n\n3. Action_desc: Based on the provided guidebook, screenshot, observations, and historical information:\n    3.1 If in an intermediate step, you are permitted to output only the current ONE STEP (in a sentence) to solve the task based on the information below.\n    3.2 If you have completed the global task, please provide any information relevant to your answer, or summarize your thoughts and actions to complete the task.\n\n4. Action: Action should STRICTLY follow the format:\n    4.1 Click [Numerical_Label]\n    4.2 Type [Numerical_Label]; [Content]\n    4.3 Clear [Numerical_Label]\n    4.4 Scroll [Numerical_Label or WINDOW]; [up or down]\n    4.5 Wait\n    4.6 GoBack\n    4.7 Restart\n    4.8 ANSWER; [content]\n\n\nIMPORTANT!!! * Output Format: *\n```\n<think>your_think_about_the_guidebook_history_trajectory_last_and_current_state</think>\n<reference>reference_in_the_guidebook(subtask#[id])</reference>\n<action_desc>one_sentence_describe_your_action_towards_the_global_task</action_desc>\n<action>your_action</action>\n```\n"
def get_cur_prompt(cur_obs, last_step, global_task, iter):
    text = HEAD
    text += f"The No.{iter} Page: {cur_obs}\n\n"
    text += f"""The screenshot for this step is provided above.\n\nPlease, First learn useful information (e.g., plan, actions, UI elements, etc.) from the provided guidebook above, while ignoring any details irrelevant to the current task.
Then, based on the current (The No.{iter} Page) and last (The No.{iter - 1} Page) observations and their screenshots, Output the status and the next suggestions based on the below information.

Global Task (the final objective you need to complete step by step.):
{global_task}

Last Step (the last step you executed in the previous step, awaiting judgment):
{last_step}


"""
    text += OUTPUT_INS

    return text.strip()

def prepare_messages(head_prompt, tutorial_messages, cur_turn_prompt, cur_turn_image, last_turn_prompt=None, last_turn_image=None, history_outputs=None):

    messages = [{
       "role": "system",
        "content": [
            {
                "type": "text",
                "text": head_prompt,
            }
        ]
    }]

    messages += tutorial_messages
    messages += [{
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "OK, I have received the guidebook. Please input the historical operation data, the last and the current observation, and the instruction."
            }
        ]
    }]

    if history_outputs is not None:
        messages += [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": history_outputs,
            }]
        }]

        messages += [{
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "OK, I have received the previous operation data. Now I am waiting for you to input the latest observation and instruction."
                }
            ]
        }]

    if last_turn_prompt is not None:
        messages += [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": last_turn_image
                    },
                    {
                        "type": "text",
                        "text": last_turn_prompt,
                    }
                ]
            }]
        messages += [{
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "OK, I have received the last step observation and the output of planning and action model. Next, please provide the observation and instruction for the current step."
                    }
                ]
            }]

    messages += [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": cur_turn_image
            },
            {
                "type": "text",
                "text": cur_turn_prompt,
            }
        ]
    }]

    return messages






def read_json(ffile):
    with open(ffile, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

def get_guidebook_messages(interact_messages):

    title = interact_messages[1]['title']
    ques = interact_messages[0]['ques']
    images = []
    text = f"\nYou are provided with a potentially helpful guidebook, titled: {title}."

    for iii in range(2, len(interact_messages)):
        message = interact_messages[iii]
        step_id = message['step_id']
        screenshot_path = message['screenshot_path'] if 'screenshot_path' in message else None
        # text = message['text']
        # alternatives = message['alternatives']
        if screenshot_path is not None and screenshot_path != "none":
            text_input = f"""# Subtask {iii - 1}: {message["text"]}
Alternative options potentially available for other similar tasks: {message["alternatives"]}
Screenshot: <image>
"""
            screenshot_path = screenshot_path.replace('./video', './guidebook').replace("./guidebooks1", './guidebooks').replace('./guidebooks3', './guidebooks').replace('Screenshot', 'screenshot')
            if 'Screenshot' in screenshot_path:
                screenshot_path = screenshot_path
                print(screenshot_path)

            images.append(screenshot_path)
        else:
            text_input = f"""# Subtask {iii - 1}: {message["text"]}
Alternative options potentially available for other similar tasks: {message["alternatives"]}
(The screenshot for this step has been omitted.)
"""


        text += "\n"
        text += text_input

    if len(images) > 5:
        print(images)

    guidebook_text = text

    messages = []

    if "<image>" in guidebook_text:
        text = guidebook_text.split("<image>")
        if len(text) == 2:
            content = guidebook_text.replace("<image>", "")
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "./data/" + images[0],
                    },
                    {
                        "type": "text",
                        "text": content,
                    }
                ]
            })
        else:
            temp = []
            ii = 0
            for segment in text[:-1]:
                temp.append({"type": "text", "text": segment})
                temp.append({"type": "image", "image": "./data/" + images[ii]})
                ii += 1
            messages.append({
                "role": "user",
                "content": temp,
            })
    else:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text":    guidebook_text}],
        })

    return messages


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


def format_msg_acctree_and_screenshot(it, pdf_obs, warn_obs, ac_tree):
    if it == 1:
        return ac_tree
    else:
        if not pdf_obs:
            return f"{ac_tree}\nWarn Observation:{warn_obs}"
        else:
             return  f"{ac_tree}\nPDFObservation: {pdf_obs}"

def get_guidebook_messages1(interact_messages):
    messages = []
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": "There is no relevant guidebook. Please complete this task based on the provided memory and your knowledge and skills, without using an external guidebook."}],
    })

    return messages

def call_qwen_vl(model, processor, messages):

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages, image_patch_size=16)
    #
    # # since qwen-vl-utils has resize the images/videos, \
    # # we should pass do_resize=False to avoid duplicate operation in processor!
    inputs = processor(text=text, images=images, videos=videos, do_resize=False, return_tensors="pt")
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=1024,
                                   do_sample=True,
                                   temperature=1,
                                   top_p=1
                                   )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(output_text)

    return output_text


@retry_with_exponential_backoff(initial_delay=1, exponential_base=1.05, jitter=False, max_retries=10)
def post_with_retry(url, cur_example):
    response = requests.post(url, json=cur_example)
    if response.status_code == 200:
        result = response.json()
    else:
        print(f"Wrong HTTP response code {response.status_code}")
        raise ValueError(f"Wrong HTTP response code {response.status_code}")
    return result


def clean_accessibility_tree(accessibility_tree):
    treenodes = accessibility_tree.split('\n')
    results = ''

    for tnode in treenodes:
        tnode = tnode.split('url:')[0]
        results += tnode + '\n'

    return results

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
    parser.add_argument("--api_key", default="sk-cFCgbqnsCtpI2hpalt0wtBbH7HnB35yIrquC2h8rFNo3sgB9", type=str,
                        help="YOUR_OPENAI_API_KEY")
    parser.add_argument("--api_model", default="gpt-5", type=str, help="api model name")
    parser.add_argument("--output_dir", type=str, default='./results/guidebook')
    parser.add_argument("--evaluation_name", type=str, default="grpo_baseline_0301", help="window width")
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

    model = Qwen3VLForConditionalGeneration.from_pretrained("/root/models2/webone/baseline_step_200/huggingface", dtype="auto",
                                                            device_map="auto")

    processor = AutoProcessor.from_pretrained("/root/models2/webone/baseline_step_200/huggingface")  # /root/models2/Qwen3-VL/Qwen3-VL-8B-Instruct

    # OpenAI client
    if not args.api_localhost:
        client = OpenAI(api_key=args.api_key, base_url="http://35.164.11.19:3887/v1")

    else:
        client_localhost = args.api_localhost

        if args.api_file_assistant:
            client_file_assis = OpenAI(api_key=args.api_key)

    options = driver_config(args)

    # Save Result file

    result_dir = args.output_dir
    result_dir = os.path.join(result_dir, args.evaluation_name)
    os.makedirs(result_dir, exist_ok=True)

    # retry_task = {task: 0 for task in read_json('./retry_mind2web_list.json')}

    large_window = {}

    # Load tasks
    tasks = read_json(args.test_file)

    # random.seed(0)
    random.seed(1)
    random.shuffle(tasks)
    tasks = tasks[0:280]
    # random.shuffle(tasks)

    print('number of tasks', len(tasks))
    for task_id in range(0, len(tasks), 1):
        task = tasks[task_id]
        id = f"{task['new_id']}"
        print(f"{task_id}: {id}")

        cur_website = task['web']
        if task['website'].lower() == 'apple':
            continue
        # if id not in retry_task:
        #     continue
        task_dir = os.path.join(result_dir, 'task{}'.format(id))
        if os.path.exists(os.path.join(task_dir, 'interact_messages.json')):
            print('1 This task has been processed', id)
            continue
            data = json.load(open(os.path.join(task_dir, 'interact_messages.json')))[-1]['content']
            if 'ANSWER' in data:
                print('1 This task has been processed', id)
                continue
        if os.path.exists(os.path.join(task_dir, 'screenshot15.png')):
            print('This task has been processed', task["id"])
            continue
        # if os.path.exists(os.path.join(task_dir, 'eval_res.json')):
        #     success = read_json(os.path.join(task_dir, 'eval_res.json'))[-1]['success']
        #     if success == 1:
        #         continue

        try:
            tutorial_message = get_guidebook_messages(
                read_json(os.path.join('./guidebooks', task['website'], task['tutorial_name'], 'guidebook2.json')))
        except Exception as e:
            print(e)
            continue


        os.makedirs(task_dir, exist_ok=True)
        setup_logger(task_dir)
        logging.info(f'########## TASK{task["new_id"]} ##########')
        print(f'########## TASK{task["new_id"]} ##########')
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
        try:
            driver_task.get(task['web'])  # https://www.recreation.gov/
        except Exception as e:
            continue

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

        obs_prompt = "Observation: please analyze the attached accessibility tree & screenshot and give the Thought and Action. "

        init_msg = f"""Now given a task: {task['ques']}  Please interact with https://www.example.com and get the answer. (Accept all cookie policy and Close all irrelevant pop-ups that appear during this process.)\n"""
        if 'sports.yahoo' in cur_website:
            init_msg = f"""Now given a task: {task['ques']}  Please interact with https://www.example.com and get the answer. (if the first page is a checkout, you just need to GoBack !!! once it appears. Close all irrelevant pop-ups that appear during this process.)\n"""

        init_msg = init_msg.replace('https://www.example.com', task['web']).replace('THEYEAR', '2025').replace(
            'LastYear', '2025').replace('THEMONTH', 'MAR.').replace('NEXTMONTH', 'APR.')
        task['ques'] = task['ques'].replace('THEYEAR', '2026').replace('LastYear', '2025').replace('THEMONTH',
                                                                                                   'MAR.').replace(
            'NEXTMONTH', 'APR.')

        cur_system_prompt = System_head + init_msg
        last_step, last_screenshot, last_obs = "", "", ""
        model_outputs = []
        print(f"Task: {task['ques']}")
        it = 0
        while it < args.max_iter:
            logging.info(f'Iter: {it}')
            print(f"Step: {it}")
            it += 1

            try:
                accessibility_tree_path = os.path.join(task_dir, 'accessibility_tree{}'.format(it))
                ac_tree, obs_info = get_webarena_accessibility_tree(driver_task, accessibility_tree_path)
                ac_tree = clean_accessibility_tree(ac_tree)
            except Exception as e:
                logging.error('Driver error when obtaining accessibility tree.')
                logging.error(e)
                break

            img_path = os.path.join(task_dir, 'screenshot{}.png'.format(it))
            try:
                driver_task.save_screenshot(img_path)
            except Exception as e:
                continue
            compress_image(img_path, img_path)


            if not fail_obs:
                curr_obs = format_msg_acctree_and_screenshot(it, pdf_obs, warn_obs, ac_tree)
            else:
                curr_obs = format_msg_acctree_and_screenshot(it,"", fail_obs, ac_tree)


            if it == 1:
                cur_prompt = get_cur_prompt(curr_obs, "None (Currently, it is the first step with no previous step.)", init_msg, 1)
                messages_ = prepare_messages(cur_system_prompt, tutorial_message, cur_prompt, img_path)
            elif it == 2:
                cur_prompt = get_cur_prompt(curr_obs, last_step, init_msg, 2)
                last_prompt = get_last_prompt(last_obs, model_outputs[-1], 1)
                messages_ = prepare_messages(cur_system_prompt, tutorial_message, cur_prompt, img_path, last_turn_prompt=last_prompt, last_turn_image=last_screenshot)
            else:
                cur_prompt = get_cur_prompt(curr_obs, last_step, init_msg, it)
                last_prompt = get_last_prompt(last_obs, model_outputs[-1], it - 1)
                history_prompt = get_history_output(model_outputs[:-1])
                messages_ = prepare_messages(cur_system_prompt, tutorial_message, cur_prompt, img_path, last_turn_prompt=last_prompt, last_turn_image=last_screenshot, history_outputs=history_prompt)

            while True:
                openai_response = call_qwen_vl(model, processor, messages_)
                response_msg = openai_response
                cur_think = extract_patterntext(openai_response, 'think')
                cur_action = extract_patterntext(openai_response, 'action')
                cur_next = extract_patterntext(openai_response, 'action_desc')
                try:
                    action_key, info = extract_information(cur_action)
                except Exception as e:
                    continue
                if cur_action is not None and action_key is not None:
                    break

            model_outputs.append(openai_response)
            last_obs = curr_obs
            last_step = cur_next
            last_screenshot = img_path

            print(f'Thought: {cur_think}')
            chosen_action = cur_action
            print(f'Action: {chosen_action}')

            fail_obs = ""
            pdf_obs = ""
            warn_obs = ""
            # execute action

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
                    print('Answering...')
                    print(info['content'])
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

        last_action = extract_patterntext(model_outputs[-1], "action")

        results = {
            "task": task['ques'],
            "answer": last_action if 'ANSWER' in last_action else '',
            "step_num": len(model_outputs),
            "all_outputs": model_outputs,
        }
        print_message(results, task_dir, is_v=True)
        driver_task.quit()


if __name__ == '__main__':
    main()
    print('End of process')

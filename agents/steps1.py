import platform
import argparse
import time
import json
import re
import os
# import shutil
# import logging
# from time import sleep
#
# import requests
#
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.common.action_chains import ActionChains
#
# from WebVoyager.utils.utils import clip_message_and_obs_gemini, print_message_gemini
# from WebVoyager.utils.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_TEXT_ONLY, SYSTEM_PROMPT_ACTREE_IMAGE
# from openai import OpenAI
# from WebVoyager.utils.utils import get_web_element_rect, encode_image, extract_information, print_message,\
#     get_webarena_accessibility_tree, get_pdf_retrieval_ans_from_assistant, clip_message_and_obs, clip_message_and_obs_text_only
# import copy
# from WebVoyager.agents.inputs import system_prompt1, system_prompt2, extract_patterntext
from utils.llm import call_gemini_api
# from WebVoyager.utils.utils_localhost import retry_with_exponential_backoff, convert_msg_format_openai_to_localhost
import json


def only_actor(hisitory_message, cur_task):

    text_prompt = f'''Imagine you are a robot browsing the web, just like humans. Now you need to complete a provided instruction. In current step, you will receive an Observation that includes a screenshot and an accessibility tree of a webpage.
Carefully analyze the observation to identify the Numerical Label (in the Accessibility Tree) corresponding to the Web Element that requires interaction, then follow the guidelines and choose one of the following actions to complete the specific subtask:
1. Click a Web Element.
2. Delete existing content in a textbox and then type content. 
3. Scroll up or down. Multiple scrolls are allowed to browse the webpage. Pay attention!! The default scroll is the whole window. If the scroll widget is located in a certain area of the webpage, then you have to specify a Web Element in that area. I would hover the mouse there and then scroll.
4. Wait. Typically used to wait for unfinished webpage processes, with a duration of 5 seconds.
5. Go back, returning to the previous webpage (Deliberately).
6. Restart, directly jump to the beginning page. When you can't complete the task, try starting over from the beginning (Deliberately).
7. Answer. This action should only be chosen when the global task have been solved and should stop. (If the current subtask states that the global task has been completed, please summarize and answer to the global task as required.)

Correspondingly, Action should STRICTLY follow the format:
- Click [Numerical_Label]
- Type [Numerical_Label]; [Content]
- Scroll [Numerical_Label or WINDOW]; [up or down]
- Wait
- GoBack
- Restart
- ANSWER; [content]

Key Guidelines You MUST follow:
* Action guidelines *
1) To input text, Don't click on textbox first, directly select Type action. After typing, the system automatically hits `ENTER` key. Sometimes you should click the search button to apply search filters. Try to use simple language when searching.  
2) You must Distinguish between textbox and search button, don't type content into the button! If no textbox is found, you may need to click the search button first before the textbox is displayed. 
3) Execute only one action per iteration.
4) When a complex global Task involves multiple questions or steps, select "ANSWER" only at the very end, after addressing all of these questions (steps). Flexibly combine your own abilities with the information in the web page. Double check the formatting requirements in the global task when ANSWER. 
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
            "parts": [{
                "text": text_prompt
            }]
        }
    ]

    prompt_tokens, completion_tokens, call_error, openai_response = call_gemini_api(hisitory_message + message)
    if not call_error:
        raw_output = openai_response['candidates'][0]['content']['parts'][0]['text']

        return message, raw_output, call_error, prompt_tokens, completion_tokens
    else:
        return message, None, call_error, prompt_tokens, completion_tokens






def only_planning(tutorial_messages, all_plan, last_step, it, last_plan):

    text_prompt = f'''Please, First learn useful information (e.g., plan, actions, UI elements, etc.) from the provided guidebook above, while ignoring any details irrelevant to the current task. 
Then, based on the current (The No.{it} Page) and last (The No.{it - 1} Page) observations and their screenshots, Output the status and the next suggestions based on the below information. 

Software Usage Tips:
1) Check if the current UI state matches the main goal's success criteria, and detect meaningful changes between pre- and post-screenshots.
2) If the subtask is irrelevant/impossible (e.g., clicking a non-existent button) → "replace" + suggest correction.
3) Flag unexpected states (errors, timeouts, missing elements) as "failure".
4) STRICTLY Avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait is also NOT allowed.
5) If the goal was achieved via unplanned but valid methods → "success".
6) The action types in Subtask should be in [Click, Type, Scroll, Wait, GoBack, Restart, ANSWER], Avoid generate navigation or location actions, as they don't align with any specified step above; directly click on the desired link instead. 
7) IMPORTANT: Sometimes you need to do some reasoning and thinking. You MUST do it before output the current subtask. 
8) Don't redo subtasks in previous steps if they are completed.
9) Don't output any comments.
10) Please select the action "ANSWER;" when you believe you have completed the global task, and summarize and respond to this task in the format: <next>ANSWER; your answer</next>


Total Planning:
{all_plan}

Last SubTask:
{last_step}

Output:
1. Think: You are permitted to think step by step to briefly analyze the alignment between the observed state and the expected outcome of the Executed subtask. (in 50 words.)
2. Status (single-term output): [success/failure] Based on the provided screenshot, observation, total planning, and history info, determine if the Executed subtask was completed: 
    2.1 success: The executed subtask was completed as planned OR achieved through equivalent alternative one or more actions.
    2.2 failure: The executed subtask was not completed (e.g., wrong action taken, incorrect parameters, no detectable UI changes, timeout, etc.).
Note: Judge based on outcome equivalence, not procedural exactness.
3. Next: Based on the provided guidebook, screenshot, observations, total planning, and historical information regarding the completion status of the executed subtask: if the previous task was successfully completed, proceed to execute the next subtask according to the original global plan; if the previous task failed, continue working on the same subtask by implementing alternative approaches until its sub-goal is achieved. 
    3.1 If in the intermediate step, You are permitted to only output the current ONE STEP (in a sentence) to solve the task based on below information. **DO NOT Directly give the numerical label of the operated element**
    3.2 If you have completed the global task, please directly answer the task as required, or describe the results, or summarize your thoughts and actions to complete the task.


Output Format:
```
<think>your_think</think>
<status>your_judge</status> 
<next>your_output</next>
```
'''

    message = [
        {
            "role": "user",
            "parts": [{'text': text_prompt}]
        }
    ]

    messages_ = tutorial_messages + message

    prompt_tokens, completion_tokens, call_error, openai_response = call_gemini_api(messages_)
    if not call_error:
        raw_output = openai_response['candidates'][0]['content']['parts'][0]['text']

        return message, raw_output, call_error, prompt_tokens, completion_tokens
    else:
        return message, None, call_error, prompt_tokens, completion_tokens

    # raw_steps = call_llm(None, message)
    #
    # success_flag = self.extract_patterntext(raw_steps, 'status')
    # cur_step = self.extract_patterntext(raw_steps, 'next')
    # thinking = self.extract_patterntext(raw_steps, 'think')

    # return success_flag, cur_step, thinking, message, raw_steps


def first_planning(tutorial_message, all_plan, first_subtask):
    system_prompt1 = '''Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, you will receive an Observation that includes a screenshot and an accessibility tree of a webpage.
Carefully analyze the observation to identify the Numerical Label (in the Accessibility Tree) corresponding to the Web Element that requires interaction, then follow the guidelines and choose one of the following actions:
1. Click a Web Element.
2. Delete existing content in a textbox and then type content. 
3. Scroll up or down. Multiple scrolls are allowed to browse the webpage. Pay attention!! The default scroll is the whole window. If the scroll widget is located in a certain area of the webpage, then you have to specify a Web Element in that area. I would hover the mouse there and then scroll.
4. Wait. Typically used to wait for unfinished webpage processes, with a duration of 5 seconds.
5. Go back, returning to the previous webpage (Deliberately).
6. Restart, directly jump to the beginning page. When you can't complete the task, try starting over from the beginning (Deliberately).
7. Answer. This action should only be chosen when all questions in the task have been solved.

Correspondingly, Action should STRICTLY follow the format:
- Click [Numerical_Label]
- Type [Numerical_Label]; [Content]
- Scroll [Numerical_Label or WINDOW]; [up or down]
- Wait
- GoBack
- Restart
- ANSWER; [content]

Key Guidelines You MUST follow:
* Action guidelines *
1) To input text, Don't click on textbox first, directly select Type action. After typing, the system automatically hits `ENTER` key. Sometimes you should click the search button to apply search filters. Try to use simple language when searching.  
2) You must Distinguish between textbox and search button, don't type content into the button! If no textbox is found, you may need to click the search button first before the textbox is displayed. 
3) Execute only one action per iteration. 
4) STRICTLY Avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait is also NOT allowed.
5) When a complex Task involves multiple questions or steps, select "ANSWER" only at the very end, after addressing all of these questions (steps). Flexibly combine your own abilities with the information in the web page. Double check the formatting requirements in the task when ANSWER. 
* Web Browsing Guidelines *
1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages. Pay attention to Key Web Elements like search textbox and menu.
2) Vsit video websites like YouTube is allowed BUT you can't play videos. Clicking to download PDF is allowed and will be analyzed by the Assistant API.
3) Focus on the date in task, you must look for results that match the date. It may be necessary to find the correct year, month and day at calendar.
4) Pay attention to the filter and sort functions on the page, which, combined with scroll, can help you solve conditions like 'highest', 'cheapest', 'lowest', 'earliest', etc. Try your best to find the answer that best fits the task.

NOTE:
1. You should directly provide all the necessary content to solve the current subtask, without introducing unmentioned tools, unclear information, or rewriting it as a new problem.
3. MUST REMEMBER all information in the current subtask should be filled with the specific content, not the variable or other new subtask.
4. IMPORTANT: Sometimes you need to do some reasoning and thinking. You MUST do it before output the current subtask. 
6. Don't output any comments.

'''

    text_prompt = f'''

# Total Planning:
{all_plan}

# The First Subtask (The Current Subtask): {first_subtask}

Output:
1. Think: You are permitted to think step by step to output the current step. (in 50 words.)
2. DESC: Based on the provided screenshot, observations, total planning, and historical information and the current subtask: You are permitted to only output the current ACTION DESC in a sentence to solve the task based on below information (only one sentence).
3. ACTION: Generate the action and Only use actions predefined above, the [element_id] must correspond to an ID from the OBSERVATION

Output Format:
```
<think>your_think</think>
<desc>the_description_about_the_next action</desc>
<action>next_Action [element_id] [args if needed]</action>
```
'''

    message = []

    message.append({
        "role": "user",
        "parts": [
            {
                "text": system_prompt1 + text_prompt
            },

        ]
    })

    messages_ = tutorial_message + message

    prompt_tokens, completion_tokens, call_error, openai_response = call_gemini_api(messages_)
    if not call_error:
        raw_output = openai_response['candidates'][0]['content']['parts'][0]['text']

        return message, raw_output, call_error, prompt_tokens, completion_tokens
    else:
        return message, None, call_error, prompt_tokens, completion_tokens


#     example_text1 = f'''Please, Please, First learn useful information (e.g., plan, actions, UI elements, etc.) from the provided guidebook above, while ignoring any details irrelevant to the current task.
# Then, based on the screenshot and the parsed GUI elements from the provided screenshot below, Only output the Current Step to complete the OBJECTIVE (Main Goal).
#
# # OBSERVATION:
# [28] [A] [Pre-baked Gingerbread House Kit Value Pack, 17 oz., Pack of 2, Total 34 oz.]
# [] [StaticText] [$19.99]
# [30] [BUTTON] [Add to Cart]
# [34] [A] [V8 +Energy, Healthy Energy Drink, Steady Energy from Black and Green Tea, Pomegranate Blueberry, 8 Ounce Can, Pack of 24]
# [] [StaticText] [$14.47]
# [36] [BUTTON] [Add to Cart]
# [40] [A] [Elmwood Inn Fine Teas, Orange Vanilla Caffeine-free Fruit Infusion, 16-Ounce Pouch]
# [] [StaticText] [$19.36]
# [42] [BUTTON] [Add to Cart]
# [43] [A] [Add to Wish List]
#
# # The Current Page URL: http://onestopmarket.com
#
# # Information about Task:
# Can you take me to the product page for the first item added to the cart in the video?
#
#
# Output:
# 1. Think: You are permitted to think step by step to output the current step. (in 50 words.)
# 2. DESC: Based on the provided screenshot, observations, total planning, and historical information and the current subtask: You are permitted to only output the current ACTION DESC in a sentence to solve the task based on below information (only one sentence).
# 3. ACTION: Generate the action and Only use actions predefined in the system prompt, the [element_id] must correspond to an ID from the OBSERVATION
#
# Output Format:
# ```
# <think>your_think</think>
# <desc>the_description_about_the_next action</desc>
# <action>next_Action [element_id] [args if needed]</action>
# ```
# '''
#
#     example_text_ans1 = '''```
# <think>Let's think step-by-step. This page lists all the forums on the website. In the video, the post that the user left a comment was under the r/technology forum, so I should navigate to that. I can navigate to that forum by first clicking on the r/technology link. Therefore, I will issue the click action.</think>
# <desc>Clicking on the title of the listing will take me to the item page</desc>
# <action>click [40]</action>
# ```
# '''
#
#     example_text2 = f'''Please, based on the screenshot and the parsed GUI elements from the provided screenshot below, Only output the Current Step to complete the OBJECTIVE (Main Goal).
#
# # OBSERVATION:
# [] [StaticText] [Forums]
# [1] [A] [Forums]
# [9] [A] [Alphabetical]
# [] [StaticText] [allentown]
# [10] [A] [allentown]
# [] [StaticText] [baltimore]
# [16] [A] [baltimore]
# [] [StaticText] [books]
# [17] [A] [books]
# [] [StaticText] [boston]
# [18] [A] [boston]
# [] [StaticText] [MachineLearning]
# [52] [A] [MachineLearning]
# [] [StaticText] [pittsburgh]
# [78] [A] [pittsburgh]
# [] [StaticText] [technology]
# [90] [A] [technology]
# [] [StaticText] [television]
# [91] [A] [television]
# [] [StaticText] [Running Postmill]
# [105] [A] [Postmill]
#
# # The Current Page URL: http://reddit.com
#
# # Information about Task:
# Can you take me to the post in the video the user commented on?
#
#
# Output:
# 1. Think: You are permitted to think step by step to output the current step. (in 50 words.)
# 2. DESC: Based on the provided screenshot, observations, total planning, and historical information and the current subtask: You are permitted to only output the current ACTION DESC in a sentence to solve the task based on below information (only one sentence).
# 3. ACTION: Generate the action and Only use actions predefined in the system prompt, the [element_id] must correspond to an ID from the OBSERVATION
#
# Output Format:
# ```
# <think>your_think</think>
# <desc>the_description_about_the_next action</desc>
# <action>next_Action [element_id] [args if needed]</action>
# ```
# '''
#
#     example_text_ans2 = '''```
# <think>Let's think step-by-step. This page lists all the forums on the website. In the video, the post that the user left a comment was under the r/technology forum, so I should navigate to that. I can navigate to that forum by first clicking on the r/technology link. Therefore, I will issue the click action.</think>
# <desc>Navigate to that forum by first clicking on the r/technology link.</desc>
# <action>click [90]</action>
# ```
# '''
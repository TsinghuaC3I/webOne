import re
from utils.utils import get_web_element_rect, encode_image, extract_information, print_message,\
    get_webarena_accessibility_tree, get_pdf_retrieval_ans_from_assistant, clip_message_and_obs, clip_message_and_obs_text_only

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
5. stop the task when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket.
6. Don't output any comments.

Your reply should strictly follow the format:
```
<think>your_think</think>
<desc>the_description_about_the_next action</desc>
<action>next_Action [element_id] [args if needed]</action>
```
'''


system_prompt2 = '''You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
A list of frames sampled from a video tutorial about this task or a similar task and the audio transcription.
The current web page's accessibility tree: This is a simplified representation of the webpage, providing key information.
The current web page's URL: This is the page you're currently navigating.
The open tabs: These are the tabs you have open.
The previous action: This is the action you just performed. It may be helpful to track your progress.

The actions you can perform fall into several categories:

Page Operation Actions:
```click [id]```: This action clicks on an element with a specific id on the webpage.
```type [id] [content]```: Use this to type the content into the field with id. By default, the "Enter" key is pressed after typing unless press_enter_after is set to 0, i.e., ```type [id] [content] [0]```.

URL Navigation Actions:
```goto [url]```: Navigate to a specific URL.
```go_back```: Navigate to the previously viewed page.
```go_forward```: Navigate to the next page (if a previous 'go_back' action was performed).

Completion Action:
```stop [answer]```: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket.

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current observation
2. You should only issue one action at a time.
3. You should follow the examples to reason step by step and then issue the current action.
4. Generate the action in the correct format. Start with a "In summary, the current action I will perform is" phrase, followed by action inside ``````. For example, "In summary, the current action I will perform is ```click [1]```".
5. Issue stop action when you think you have achieved the objective. Don't generate anything after stop.'''



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


def get_guidebook_messages(interact_messages):

    title = interact_messages[1]['title']
    ques = interact_messages[0]['ques']

    parts = [{"text": f"You are provided with a potentially helpful guidebook, titled: {title}.",}]

    for iii in range(2, len(interact_messages)):
        message = interact_messages[iii]
        step_id = message['step_id']
        screenshot_path = message['screenshot_path'] if 'screenshot_path' in message else None
        # text = message['text']
        # alternatives = message['alternatives']
        if screenshot_path is not None:
            b64_img = encode_image(screenshot_path)

            text_input = f"""Step {iii - 1}: {message["text"]}
Alternative options potentially available for other similar tasks: {message["alternatives"]}.
The screenshot:
"""
            parts.append({"text": text_input})
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64_img,
                },
            })

        else:
            text_input = f"""Step {iii - 1}: {message["text"]}
Alternative options potentially available for other similar tasks: {message["alternatives"]}
"""
            parts.append({"text": text_input})

    guidebook_messages = [
        {
            "role": "user",
            "parts": parts
        }
    ]

    return guidebook_messages


def format_msg_acctree_and_screenshot_gemini(it, pdf_obs, warn_obs, ac_tree):
    if it == 1:
        init_msg = f"I've provided the Accessibility Tree and Screenshot of the No.1 Page:\n{ac_tree}\nScreenshot:\n"
        return init_msg
    else:
        if not pdf_obs:
            curr_msg = f"The No.{it} Page:\nObservation:{warn_obs} please analyze the attached Accessibility Tree & screenshot and complete the current subtask.\n{ac_tree}\nScreenshot:\n"

        else:
            curr_msg =  f"The No.{it} Page:\nObservation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The accessibility tree & screenshot of the current page (The No.{it} Page) is also attached, and complete the current subtask.\n{ac_tree}\nScreenshot:\n"

        return curr_msg


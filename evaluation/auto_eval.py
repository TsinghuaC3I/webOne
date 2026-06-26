import argparse
import os
import json
import time
import re
import base64
import pdb
from openai import OpenAI

SYSTEM_PROMPT = """As an evaluator, you will be presented with three primary components to assist you in your role:

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
USER_PROMPT = """TASK: <task>
Result Response: <answer>
<num> screenshots at the end: """


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def auto_eval_by_gpt4v(process_dir, openai_client, api_model, img_num):
    print(f'--------------------- {process_dir} ---------------------')
    res_files = sorted(os.listdir(process_dir))
    with open(os.path.join(process_dir, 'interact_messages.json')) as fr:
        it_messages = json.load(fr)
    cur_img_nums = 0
    for item in it_messages[3:]:
        if item['role'] == 'user' and len(item['content']) > 1 and  item['content'][1]['type'] == 'image':
            cur_img_nums += 1


    if len(it_messages) == 1:
        print('Not find answer for ' + process_dir + ' only system messages')
        print()
        return 0

    task_info = it_messages[3]["content"]
    if type(task_info) == list:
        task_info = task_info[0]["text"]
    assert 'Now given a task' in task_info
    pattern = r"Now given a task:(.+?)Please interact with"
    matches = re.search(pattern, task_info, re.DOTALL)
    task_content = matches.group(1).strip()

    ans_info = it_messages[-1]["content"]
    if isinstance(ans_info, str):
        ans_info = ans_info.replace("Action:ANSWER", "Action: ANSWER")

    # ans_info = ans_info.replace("'Action:ANSWER'", "'Action: ANSWER'")
    if 'Action: ANSWER' not in ans_info:
        print('Not find answer for ' + process_dir)
        print()
        return 0
    pattern_ans = r"ANSWER[;:\n ]+\[?(.[^\]]*)\]?"
    matches_ans = re.search(pattern_ans, ans_info)
    answer_content = matches_ans.group(1).strip()

    # max_screenshot_id = max([int(f[10:].split('.png')[0]) for f in os.listdir(process_dir) if '.png' in f])
    # final_screenshot = f'screenshot{max_screenshot_id}.png'
    # b64_img = encode_image(os.path.join(process_dir, final_screenshot))
    whole_content_img = []
    pattern_png = r'screenshot(\d+)\.png'
    matches = [(filename, int(re.search(pattern_png, filename).group(1))) for filename in res_files if re.search(pattern_png, filename)]
    matches.sort(key=lambda x: x[1])
    matches = matches[:cur_img_nums]
    end_files = matches[-img_num:]
    for png_file in end_files:
        b64_img = encode_image(os.path.join(process_dir, png_file[0]))
        whole_content_img.append(
            {
                'type': 'image_url',
                'image_url': {"url": f"data:image/png;base64,{b64_img}"}
            }
        )

    user_prompt_tmp = USER_PROMPT.replace('<task>', task_content)
    user_prompt_tmp = user_prompt_tmp.replace('<answer>', answer_content)
    user_prompt_tmp = user_prompt_tmp.replace('<num>', str(len(end_files)))
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': user_prompt_tmp}
            ]
            + whole_content_img
            + [{'type': 'text', 'text': "Your verdict:\n"}]
        }
    ]
    error_nums = 0
    while True:
        try:
            print('Calling gpt4v API to get the auto evaluation......')
            openai_response = openai_client.chat.completions.create(
                model=api_model, messages=messages, max_tokens=1000, seed=42, temperature=0
            )
            print('Prompt Tokens:', openai_response.usage.prompt_tokens, ';',
                  'Completion Tokens:', openai_response.usage.completion_tokens)
            print('Cost:', openai_response.usage.prompt_tokens/1000 * 0.01
                  + openai_response.usage.completion_tokens / 1000 * 0.03)

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
                return [{'content': 'content_filter', 'success':0}]
            else:
                time.sleep(10)
    gpt_4v_res = openai_response.choices[0].message.content
    print_message = messages[1]
    for idx in range(len(print_message['content'])):
        if print_message['content'][idx]['type'] == 'image_url':
            print_message['content'][idx]['image_url'] = {"url": "data:image/png;base64, b64_img"}

    
    print(gpt_4v_res)

    auto_eval_res = 0 if 'NOT SUCCESS' in gpt_4v_res else 1
    if 'SUCCESS' not in gpt_4v_res:
        auto_eval_res = 0# None
    print('Auto_eval_res:', auto_eval_res)
    print()
    messages.append({'role': 'assistant', 'content': gpt_4v_res, 'success': auto_eval_res})
    #exit()
    return messages

def read_json(ffile):
    with open(ffile) as f:
        return json.load(f)

website2domain = {
  "amazon": "Electronic Commerce",
  "amtrak": "Travel",
  "agoda": "Travel",
  "airbnb": "Travel",
  "apple": "Electronic Commerce",
  "allrecipes": "Other",
  "budget": "Travel",
  "boardgamegeek": "Social Media",
  "bbc news": "Other",
  "cvs": "Electronic Commerce",
  "carnival": "Travel",
  "cambridge dictionary": "Course & Learning",
  "coursera": "Course & Learning",
  "discogs": "Electronic Commerce",
  "eventbrite": "Other",
  "enterprise": "Travel",
  "ebay": "Electronic Commerce",
  "flightaware": "Travel",
  "foxsports": "Entertainment",
  "github": "Code Tool",
  "gitlab": "Code Tool",
  "google map": "Map",
  "ign": "Entertainment",
  "imdb": "Entertainment",
  "ikea": "Electronic Commerce",
  "koa": "Travel",
  "last.fm": "Entertainment",
  "mta.info": "Travel",
  "marriott": "Travel",
  "nba": "Entertainment",
  "nps.gov": "Travel",
  "nyc": "Other",
  "openstreetmap": "Map",
  "parking": "Travel",
  "ryanair": "Travel",
  "resy": "Other",
  "recreation.gov": "Travel",
  "reddit": "Social Media",
  "soundcloud": "Entertainment",
  "store.steampowered": "Entertainment",
  "stubhub": "Entertainment",
  "tvguide": "Entertainment",
  "travelzoo": "Travel",
  "thetrainline": "Travel",
  "target": "Electronic Commerce",
  "us.megabus": "Travel",
  "wolfram alpha": "Infoseek",
  "yelp": "Other",
  "arxiv": "Course & Learning",
  "booking": "Travel",
  "classified": "Other",
  "espn": "Entertainment",
  "google flights": "Travel",
  "huggingface": "Code Tool",
  "mbta": "Map",
  "rottentomatoes": "Entertainment",
  "sports.yahoo": "Entertainment",
  "shopping(onestop)": "Electronic Commerce",
  "shopping admin": "Electronic Commerce",
  "spothero": "Travel",
  "ticketcenter": "Entertainment",
  "uniqlo": "Electronic Commerce"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_dir', type=str, default='./results/guidebook/qwen3vl-8b-new') # 改这里
    parser.add_argument('--lesson_dir', type=str, default='results')
    parser.add_argument("--api_key", default="sk-vpj3a5KGeKl145pIDa1d3aB9Af994c7bAdAa53B0E2447807", type=str, help="YOUR_OPENAI_API_KEY")
    parser.add_argument("--api_model", default="gpt-4o", type=str, help="api model name")
    parser.add_argument("--max_attached_imgs", type=int, default=12)
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url="https://api3.apifans.com/v1")
    
    domain2res = {
        'Entertainment': [],
        'Shopping': [],
        'Travel': [],
    }
    web2domain = {}
    id2domain = {}
    id2web = {}
    id2datasource = {}
    all_test_data = read_json('./test_auto_eval/alltest.json')
    for item in all_test_data:
        cur_domain = website2domain[item['website'].lower()]

        if cur_domain not in domain2res:
            domain2res[cur_domain] = []

        if "datasource" not in item:
            datasource = "wv"
        else:
            ds = item["datasource"].lower()
            if "webvoyager" in ds:
                datasource = "wv"
            elif "mind2web" in ds:
                datasource = "m2w"
            else:
                datasource = "our"

        web2domain[item['website']] = cur_domain
        id2domain[item['new_id']] = cur_domain
        id2web[item['new_id']] = item['website'].lower()
        id2datasource[item['new_id']] = datasource
        item['ds'] = datasource

    web2res = {}
    datasource2res = {}
    for item in all_test_data:
        web2res[item['website'].lower()] = []
        datasource2res[item['ds']] = []

    # Please specify the coresponding task queries file
    # with open('./test_auto_eval/alltest.json') as fm:  # '../data_for_test/mind2web_test_cross_task.jsonl'
    #     for line in fm:
    #         item = json.loads(line)
    #         web2domain[item['website']] = item['domain']
    #         id2domain[item['id']] = item['domain']

    web_task_res = []
    for task_dir in sorted(os.listdir(args.process_dir)):
        file_dir = os.path.join(args.process_dir, task_dir)

        task_id = task_dir.replace('task', '')

        if task_id not in id2domain:
            continue

        if os.path.exists(os.path.join(file_dir, 'eval_res.json')):
            response = json.load(open(os.path.join(file_dir, 'eval_res.json')))
            web_task_res.append(response[-1]['success'])
            task_id = task_dir.replace('task', '')

            if task_id not in id2domain:
                continue

            web = id2web[task_id]
            web2res[web].append(response[-1]['success'])
            domain = id2domain[task_id]
            domain2res[domain].append(response[-1]['success'])
            ds = id2datasource[task_id]
            datasource2res[ds].append(response[-1]['success'])
        else:
            try:
                response = auto_eval_by_gpt4v(file_dir, client, args.api_model, args.max_attached_imgs)
            except Exception as e:
                print(e)
                continue
            if response:
                with open(os.path.join(file_dir, 'eval_res.json'), 'w') as fw:
                    json.dump(response, fw, indent=4, ensure_ascii=False)
                web_task_res.append(response[-1]['success'])
                task_id = task_dir.replace('task', '')
                domain = id2domain[task_id]
                web = id2web[task_id]
                web2res[web].append(response[-1]['success'])
                # web2domain[web] = domain
                domain2res[domain].append(response[-1]['success'])
                ds = id2datasource[task_id]
                datasource2res[ds].append(response[-1]['success'])
            else:
                web_task_res.append(0)
                task_id = task_dir.replace('task', '')
                domain = id2domain[task_id]
                web = id2web[task_id]
                web2res[web].append(0)
                domain2res[domain].append(0)
                ds = id2datasource[task_id]
                datasource2res[ds].append(0)

    print('overall:', sum(web_task_res)/len(web_task_res))

    numss = 0
    for domain, value in domain2res.items():
        if len(value) != 0:
            print(domain, sum(value)/len(value))
            numss += len(value)
    print(numss)
    # print(domain2res)
    # for domain, value in domain2res.items():
    #     print(domain, sum(value)/len(value))
    # print(web2res)
    for web, value in web2res.items():
        print( web, sum(value)/len(value) )

    nums = 0
    for ds, value in datasource2res.items():
        print(ds, sum(value)/len(value)  )
        nums += len(value)
    print(nums)

if __name__ == '__main__':
    main()

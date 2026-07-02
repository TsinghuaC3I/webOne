import platform
import time
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


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
    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or \
            (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
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
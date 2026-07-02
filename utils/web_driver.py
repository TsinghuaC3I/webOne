

from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.common.action_chains import ActionChains
#
# from WebVoyager.utils.utils import clip_message_and_obs_gemini, print_message_gemini
#
# import json


# def driver_config(args):
#     options = webdriver.ChromeOptions()
#
#     if args.save_accessibility_tree:
#         args.force_device_scale = True
#
#     if args.force_device_scale:
#         options.add_argument("--force-device-scale-factor=1")
#     # if args.headless:
#     #     options.add_argument("--headless")
#     #     options.add_argument(
#     #         "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
#     #     )
#     options.add_experimental_option(
#         "prefs", {
#             "download.default_directory": args.download_dir,
#             "plugins.always_open_pdf_externally": True
#         }
#     )
#
#     options.add_argument("--no-sandbox")
#
#     return options
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

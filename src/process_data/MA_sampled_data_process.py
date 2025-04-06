import os
import gzip
import io
import pickle
import json
from tqdm import tqdm
import math
import csv
import re
#from android_world.agents import m3a_utils
import random
import numpy as np

# for task a\b\c\d

PROMPT_PREFIX = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    " Based on user's goal/request, you may\n"
    '- Answer back if the request/goal is a question (or a chat message), like'
    ' user asks "What is my schedule for today?".\n'
    '- Complete some tasks described in the requests/goals by performing'
    ' actions (step by step) on the phone.\n\n'
    'When given a user request, you will try to complete it step by step. At'
    ' each step, a list of descriptions for most UI elements on the'
    ' current screen will be given to you (each element can be specified by an'
    ' index), together with a history of what you have done in previous steps.'
    ' Based on these pieces of information and the goal, you must choose to'
    ' perform one of the action in the following list (action description'
    ' followed by the JSON format) by outputting the action in the correct JSON'
    ' format.\n'
    '- If you think the task has been completed, finish the task by using the'
    ' status action with complete as goal_status:'
    ' `{{"action_type": "status", "goal_status": "complete"}}`\n'
    '- If you think the task is not'
    " feasible (including cases like you don't have enough information or can"
    ' not perform some necessary actions), finish by using the `status` action'
    ' with infeasible as goal_status:'
    ' `{{"action_type": "status", "goal_status": "infeasible"}}`\n'
    "- Answer user's question:"
    ' `{{"action_type": "answer", "text": "<answer_text>"}}`\n'
    '- Click/tap on a UI element (specified by its index) on the screen:'
    ' `{{"action_type": "click", "index": <target_index>}}`.\n'
    '- Long press on a UI element (specified by its index) on the screen can select that UI element:'
    ' `{{"action_type": "long_press", "index": <target_index>}}`.\n'
    '- Type text into an editable text field (specified by its index), this'
    ' action contains clicking the text field, typing in the text and pressing'
    ' the enter, so no need to click on the target field to start:'
    ' `{{"action_type": "input_text", "text": <text_input>, "index":'
    ' <target_index>}}`\n'
    '- Press the Enter key: `{{"action_type": "keyboard_enter"}}`\n'
    '- Navigate to the home screen: `{{"action_type": "navigate_home"}}`\n'
    '- Navigate back: `{{"action_type": "navigate_back"}}`\n'
    '- Scroll the screen or a scrollable UI element in one of the four'
    ' directions, use the same numeric index as above if you want to scroll a'
    ' specific UI element, leave it empty when scroll the whole screen:'
    ' `{{"action_type": "scroll", "direction": <up, down, left, right>,'
    ' "index": <optional_target_index>}}`\n'
    '- Open an app (nothing will happen if the app is not installed):'
    ' `{{"action_type": "open_app", "app_name": <name>}}`\n'
    '- Wait for the screen to update: `{{"action_type": "wait"}}`\n'
)

GUIDANCE = (
    'Here are some useful guidelines you need to follow:\n'
    'General\n'
    '- Usually there will be multiple ways to complete a task, pick the'
    ' easiest one. Also when something does not work as expected (due'
    ' to various reasons), sometimes a simple retry can solve the problem,'
    " but if it doesn't (you can see that from the history), try to"
    ' switch to other solutions.\n'
    '- Sometimes you may need to navigate the phone to gather information'
    ' needed to complete the task, for example if user asks'
    ' "what is my schedule tomorrow", then you may want to open the calendar'
    ' app (using the `open_app` action), look up information there, answer'
    " user's question (using the `answer` action) and finish (using"
    ' the `status` action with complete as goal_status).\n'
    '- For requests that are questions (or chat messages), remember to use'
    ' the `answer` action to reply to user explicitly before finish!'
    ' Merely displaying the answer on the screen is NOT sufficient (unless'
    ' the goal is something like "show me ...").\n'
    '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
    " it's already on), you can just complete the task.\n"
    'Action Related\n'
    '- Use the `open_app` action whenever you want to open an app'
    ' (nothing will happen if the app is not installed), do not use the'
    ' app drawer to open an app unless all other ways have failed.\n'
    '- Use the `input_text` action whenever you want to type'
    ' something (including password) instead of clicking characters on the'
    ' keyboard one by one. Sometimes there is some default text in the text'
    ' field you want to type in, you should `delete` default text before typing.'
    ' Care for the editable text field if there are default text\n'
    '- For `click`, `long_press` and `input_text`, the index parameter you'
    ' pick must be VISIBLE in the screenshot and also in the UI element'
    ' list given to you (some elements in the list may NOT be visible on'
    ' the screen so you can not interact with them).\n'
    '- Consider exploring the screen by using the `scroll`'
    ' action with different directions to reveal additional content.\n'
    '- The direction parameter for the `scroll` action can be confusing'
    " sometimes as it's opposite to swipe, for example, to view content at the"
    ' bottom, the `scroll` direction should be set to "down". It has been'
    ' observed that you have difficulties in choosing the correct direction, so'
    ' if one does not work, try the opposite as well.\n'
    'Text Related Operations\n'
    '- Normally to select some text on the screen: <i> Enter text selection'
    ' mode by long pressing the area where the text is, then some of the words'
    ' near the long press point will be selected (highlighted with two pointers'
    ' indicating the range) and usually a text selection bar will also appear'
    ' with options like `copy`, `paste`, `select all`, etc.'
    ' <ii> Select the exact text you need. Usually the text selected from the'
    ' previous step is NOT the one you want, you need to adjust the'
    ' range by dragging the two pointers. If you want to select all text in'
    ' the text field, simply click the `select all` button in the bar.\n'
    "- At this point, you don't have the ability to drag something around the"
    ' screen, so in general you can not select arbitrary text.\n'
    '- To delete some text: the most traditional way is to place the cursor'
    ' at the right place and use the backspace button in the keyboard to'
    ' delete the characters one by one (can long press the backspace to'
    ' accelerate if there are many to delete). Another approach is to first'
    ' select the text you want to delete, then click the backspace button'
    ' in the keyboard.\n'
    '- To copy some text: first select the exact text you want to copy, which'
    ' usually also brings up the text selection bar, then click the `copy`'
    ' button in bar.\n'
    '- To paste text into a text box, first long press the'
    ' text box, then usually the text selection bar will appear with a'
    ' `paste` button in it.\n'
    '- When typing into a text field, sometimes an auto-complete dropdown'
    ' list will appear. This usually indicating this is a enum field and you'
    ' should try to select the best match by clicking the corresponding one'
    ' in the list.\n'
)

GUIDANCE1 = (
    'Here are some useful guidelines you need to follow:\n'
    'Action Related\n'
    '- Use the `open_app` action whenever you want to open an app'
    ' (nothing will happen if the app is not installed), do not use the'
    ' app drawer to open an app unless all other ways have failed.\n'
    '- Use the `input_text` action whenever you want to type'
    ' something (including password) instead of clicking characters on the'
    ' keyboard one by one. Sometimes there is some default text in the text'
    ' field you want to type in, remember to delete them before typing.\n'
    '- For `click`, `long_press` and `input_text`, the index parameter you'
    ' pick must be VISIBLE in the screenshot and also in the UI element'
    ' list given to you (some elements in the list may NOT be visible on'
    ' the screen so you can not interact with them).\n'
    '- Consider exploring the screen by using the `scroll`'
    ' action with different directions to reveal additional content.\n'
    '- The direction parameter for the `scroll` action can be confusing'
    " sometimes as it's opposite to swipe, for example, to view content at the"
    ' bottom, the `scroll` direction should be set to "down". It has been'
    ' observed that you have difficulties in choosing the correct direction, so'
    ' if one does not work, try the opposite as well.\n'
)

Summary_UI = (
    'What is the purpose of the current UI?\n'
    'Summarize the current interface in one paragraph.\n'
    'What does the current UI aim to achieve?\n'
    )

GENERAL_PROMPT = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    'Please answer the following question based on the UI elements above.\n'
    'Questions: {question}'
    'Your Answer:\n'
    )

Query_UI_elements = (
    'What is the function of UI element X?\n'
    'What information does UI element X provide?\n'
    'What happens when click the UI element X?\n'
    'What action is associated with UI element X?\n'
)

QUERY_UI_PROMPT = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    'Please answer the following questions for all the UI elements above.\n'
    'Questions = ('
    'What is the function of UI element X?'
    'What information does UI element X provide?'
    'What happens when click the UI elementX?'
    'What action is associated with UI element X?'
    ')\n'
    'Please format your response as follows:\n'
    '{{"Question":"What is the function of UI element X?", "Answer":"........"}}\n'
    '{{"Question":"What information does UI element X provide?", "Answer":"........"}}\n'
    '{{"Question":"What happens when click the UI element X?", "Answer":"........"}}\n'
    '{{"Question":"What action is associated with UI element X?", "Answer":"........"}}\n'
    'Your response:\n'
    )

INSTRU_PROMPT = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}'
    + '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    + '\n\nNow output an action from the above list in the correct JSON format,'
    ' following the reason why you do that. Your answer should look like:\n'
    'Reason: ...\nAction: {{"action_type":...}}\n\n'
    'Your answer:\n'
    )

CRITIC_PREFIX = (
  "You are a super-intelligent agent who can expertly operate an Android phone on behalf of a user.\n"
  "Now, you need to act as a critic, evaluating the actions taken by other Android agents.\n "
  "These agents receive user tasks and current Android interface information and then take the next step. \n"
  "Your evaluation should be between [0,1]. A score close to 0 means the agent's action is useless or incorrect in "
  "achieving the user's task, a score close to 0.5 means you are uncertain whether the agent's decision is useful for "
  "achieving the user's task, and a score close to 1 means the agent's action is useful or correct in achieving the user's task."
)

ACTION_SELECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}'
    + '\n\nHere is a history of what you have done so far:\n{history}'
    + '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    + GUIDANCE
    + '{additional_guidelines}'
    + '\n\nNow output an action from the above list in the correct JSON format,'
    ' following the reason why you do that. Your answer should look like:\n'
    'Reason: ...\nAction: {{"action_type":...}}\n\n'
    'Your answer:\n'
    )

PLAUSIBLE_ACTION_SELECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}'
    + '\n\nHere is a history of what you have done so far:\n{history}'
    + '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    + GUIDANCE
    + '{additional_guidelines}'
    + '\n\nNow you need to role-play a very clumsy agent that'
    ' can only output incorrect answer (must absolutely wrong action and plausible reason, if you have no choice, you can invent a wrong action and reason) from the above list in the correct JSON format,'
    ' following the reason why you do that. Your answer should look like:\n'
    'Reason: ...\nAction: {{"action_type":...}}\n\n'
    'Your answer (Must absolutely wrong action and plausible reason):\n'
    )

ACTION_SELECTION_PROMPT_TEMPLATE1 = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}'
    + '\n\nHere is a history of what you have done so far:\n{history}'
    + '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    + GUIDANCE
    + '{additional_guidelines}'
    #+ 'The most recent action you have done is: {last_history}'
    + '\n\nNow output an action from the above list in the correct JSON format,'
    ' following the STEPS REFLECTION of your action history, the KEY STEPS NEXT for completing for user goal and the reason why you do next action. '
    #+ 'Please refer to history and don\'t get caught in a cycle of any specific action.'
    'Your answer should look like:\n'
    'Reason: [STEPS REFLECTION] The history of my action indicates...[KEY STEPS NEXT] The the next key steps for completing for user goal are ... \nAction: {{"action_type":...}}\n\n'
    + '\nHere are some demonstrations of answer:\n'
    + '1. Reason: [STEPS REFLECTION] The history of my actions indicates that I have gotten caught in a cycle of a specific action, so I need to take different and reasonable actions to break out of this loop...'
    '[KEY STEPS NEXT] The the next key steps for completing for user goal are ... \nAction: {{"action_type":...}}\n\n'
    + '2. Reason: [STEPS REFLECTION] The history of my actions indicates that I have previously performed a certain action on the same interface, it was not useful for completing the user\'s task, '
    'so I need to take different and reasonable actions...[KEY STEPS NEXT] The the next key steps for completing for user goal are ...  \nAction: {{"action_type":...}}\n\n'
    + '3. Reason: [STEPS REFLECTION] The history of my actions indicates that I have successfully opened the app, The current screen shows a dialog where I can rename the ..., but the editable text field contains default text.'
    '[KEY STEPS NEXT] The next key steps for completing for user goal are to click the delete button (UI element) to remove the default text firstly then input the desired name and save it.\nAction: {{"action_type":...}}\n\n' 
    '\nYour answer must in this format:\nReason: ...\nAction: {{"action_type":...}}\n\n Your Answer:\n'
)

PLAUSIBLE_ACTION_SELECTION_PROMPT_TEMPLATE1 = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}'
    + '\n\nHere is a history of what you have done so far:\n{history}'
    + '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    + GUIDANCE
    + '{additional_guidelines}'
    #+ 'The most recent action you have done is: {last_history}'
    + '\n\nNow output a \'plausible\' action (seemingly correct but wrong) from the above list in the correct JSON format,'
    ' following the STEPS REFLECTION of your action history, the KEY STEPS NEXT for completing for user goal and the reason why you do next action. '
    #+ 'Please refer to history and don\'t get caught in a cycle of any specific action.'
    'Your answer should look like:\n'
    'Reason: [STEPS REFLECTION] The history of my action indicates...[KEY STEPS NEXT] The the next key steps for completing for user goal are ... \nAction: {{"action_type":...}}\n\n'
    + '\nHere are some demonstrations of answer:\n'
    + '1. Reason: [STEPS REFLECTION] The history of my actions indicates that I have gotten caught in a cycle of a specific action, so I need to take different and reasonable actions to break out of this loop...'
    '[KEY STEPS NEXT] The the next key steps for completing for user goal are ... \nAction: {{"action_type":...}}\n\n'
    + '2. Reason: [STEPS REFLECTION] The history of my actions indicates that I have previously performed a certain action on the same interface, it was not useful for completing the user\'s task, '
    'so I need to take different and reasonable actions...[KEY STEPS NEXT] The the next key steps for completing for user goal are ...  \nAction: {{"action_type":...}}\n\n'
    + '3. Reason: [STEPS REFLECTION] The history of my actions indicates that I have successfully opened the app, The current screen shows a dialog where I can rename the ..., but the editable text field contains default text.'
    '[KEY STEPS NEXT] The next key steps for completing for user goal are to click the delete button (UI element) to remove the default text firstly then input the desired name and save it.\nAction: {{"action_type":...}}\n\n' 
    '\nYour answer must in this format:\nReason: ...\nAction: {{"action_type":...}}\n\n Your Answer:\n'
)

REFLECT_PROMPT = (
    'You are a super-intelligent agent who can expertly operate an Android phone on behalf of a user.'
    'You need to act as a reflective thinker, evaluating the history of action based on the user\'s goal.'
    + '\nThe current user goal/request is: {goal}'
    + '\n\nHere is a history of what you have done so far:\n{history}'
    'Your answer should look like:\n'
    'Reason: The history of my action indicates...'
    + '\nHere are some demonstrations of answer:\n'
    + '1. Reason: The history of my actions indicates that I have gotten caught in a cycle of a specific action, so I need to take different and reasonable actions to break out of this loop...\n'
    + '2. Reason: The history of my actions indicates that I have previously performed a certain action on the same interface, it was not useful for completing the user\'s task, '
    'so I need to take different and reasonable actions...\n'
    #+ '3. Reason: [STEPS REFLECTION] The history of my actions indicates that I have successfully opened the app, The current screen shows a dialog where I can rename the ..., but the editable text field contains default text.'
    #'[KEY STEPS NEXT] The next key steps for completing for user goal are to click the delete button (UI element) to remove the default text firstly then input the desired name and save it.\n' 
    'Your answer:\n'
)

CRITIC_PROMPT_TEMPLATE = (
    CRITIC_PREFIX
    + '\nThe current user goal/request is: {goal}'
    + '\n\nHere is a history of what have done so far:\n{history}'
    + '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    + GUIDANCE
    #+ '{additional_guidelines}'
    + '\nHere are the next actions different agents would like to take:\n{agents_actions}'
    + '\n\nPlease output each agent\'s score in the correct JSON format,'
    ' following the reason (also reflection) why you think the agents\' actions and reasons are correct or not, '
    'and ensuring that their actions are necessary and not redundant for achieving the user\'s goals when you give your scores. '
    'Your answer should look like:\n'
    'Reason: The goal is {{user goal}}...\nScore: {{"agent_id":score...}}\n'
    + '\nHere are some demonstrations of evaluations:\n'
    #+ '1. Reason: All agents have correctly identified that the first step to record an audio clip'
    #' using the Audio Recorder app is to open the app. This action is necessary and useful for achieving the user\'s task.\n'
    #'Score: {{"agent_0": 1, "agent_1": 1, "agent_2": 1}}\n'
    + '1. Reason: The goal is ... Agent 0 and Agent 2 attempt to scroll down to find additional options. This is a logical step given that '
    'no explicit save button is visible and the app might have additional options accessible through scrolling. Agent 1 decides '
    'to click the "Settings" button in hopes that it might lead to a menu with a save option. However, this seems less directly '
    'connected to saving the recording as the Settings menu is generally for configuration rather than saving recordings.\n'
    'Score: {{"agent_0": 0.9, "agent_1": 0.1, "agent_2": 0.9}}\n'
    + '2. Reason: The goal is ... All agents (Agent 0, 1, and 2) have chosen to input the desired name "xxx.m4a" into the text field, However, '
    'user did not specify a name. This is the incorrect next step, ...\n'
    'Score: {{"agent_0": 0.2, "agent_1": 0.2, "agent_2": 0.2}}\n'
    + '3. Reason: The goal is ... Historical information shows that the agent has taken the same action multiple times. I am unsure if taking the same action again is reasonable.\n'
    'Score: {{"agent_0": 0.5, "agent_1": 0.5, "agent_2": 0.5}}\n'
    + '\n\nNow output each agent\'s score. '
    'Your answer must in the format:\n'
    'Reason: The goal is {{user goal}}...\nScore: {{"agent_id": score, "agent_id": score, "agent_id": score}}\n\n'
    'Your Evaluation:\n'
)

SUMMARIZATION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe (overall) user goal/request is:{goal}\n'
    'Now I want you to summarize the latest step based on the action you'
    ' pick with the reason and descriptions for the before and after (the'
    ' action) screenshots.\n'
    'Here is the description for the before'
    ' screenshot:\n{before_elements}\n'
    'Here is the description for the after screenshot:\n{after_elements}\n'
    'This is the action you picked: {action}\n'
    'Based on the reason: {reason}\n\n'
    '\nBy comparing the descriptions for the two screenshots and the action'
    ' performed, give a brief summary of this step.'
    '\nHere is a history of what have done so far:\n{history}'
    ' This summary will be added to action history and used in future action'
    ' selection, so try to include essential information you think that will'
    ' be most useful for future action selection like'
    ' what you intended to do, why, if it worked as expected, if not'
    ' what might be the reason (be critical, the action/reason might not be'
    ' correct), what should/should not be done next and so on. '
    ' If you think the task has been completed, finish the task by using the'
    ' status action with complete as goal_status.'
    ' Some more rules/tips you must follow:\n'
    '- Keep it short and in one line.\n'
    "- Some actions (like `answer`, `wait`) don't involve screen change,"
    ' you can just assume they work as expected.\n'
    '- Given this summary will be added into action history, it can be used as'
    ' memory to include information that needs to be remembered, or shared'
    ' between different apps.\n\n'
    '- Keep the summary short and in one line.\n'
    'Summary of this step: '
)

SUMMARIZATION_PROMPT_TEMPLATE2 = (
    PROMPT_PREFIX
    + '\nThe (overall) user goal/request is:{goal}\n'
    'Now I want you to summarize the latest step based on the action you'
    ' pick with the reason and descriptions for the before and after (the'
    ' action) screenshots.\n'
    'Here is the description for the before'
    ' screenshot:\n{before_elements}\n'
    'Here is the description for the after screenshot:\n{after_elements}\n'
    '\nHere is a history of what have done so far:\n{history}'
    'This is the action you picked: {action}\n'
    'Based on the reason: {reason}\n\n'
    '\nBy comparing the descriptions for the two screenshots and the action'
    ' performed, give a brief summary of this step.'
    ' This summary will be added to action history and used in future action'
    ' selection, so try to include essential information you think that will'
    ' be most useful for future action selection like: '
    ' -[STEP SUMMARY]: a brief summary of current step. '
    #' -[STEPS REFLECTION]: the reflection of history actions, such as "The history of my actions indicates that I have previously performed a certain action on the same interface", "The history of my actions indicates that I have gotten caught in a cycle of a specific action, so I need to take different and reasonable actions".'
    #' -[KEY STEPS NEXT]:  what you intended to do based on your reflection, why, if it worked as expected, if not'
    #+ '\nHere are some demonstrations of STEPS REFLECTION:'
    #+ '1. [STEP SUMMARY] Clicked the "Get started" button to proceed with recording the audio clip.'
    #'\n1. [STEPS REFLECTION] The history of my actions indicates that this action does not repeat previous actions.'
    #'[KEY STEPS NEXT] The the next key steps for completing for user goal are clicked on the "Apply" button, Action: {{"action_type":...}}\n\n'
    #'1. [STEP SUMMARY] Scrolled down to find and adjust the bitrate options for the audio clip.'
    #'\n2. [STEPS REFLECTION] The history of my actions indicates that I have previously performed the ... action on the same interface, it was not useful for completing the user\'s task, so I need to take different and reasonable actions...'
    #'[KEY STEPS NEXT] The the next key steps for completing for user goal are ..., Action: {{"action_type":...}}\n\n'
    #' [] do you think the action are are necessary and and non-repetitive with historical actions,'
    #' - what you intended to do ({{"action_type":...}}), why, if it worked as expected, if not'
    #' - what might be the reason (be critical, the action/reason might not be'
    #' correct), what should/should not be done next and so on. '
    ' \nSome more rules/tips you must follow:\n'
    '- Keep it short and in one line.\n'
    "- Some actions (like `answer`, `wait`) don't involve screen change,"
    ' you can just assume they work as expected.\n'
    '- Given this summary will be added into action history, it can be used as'
    ' memory to include information that needs to be remembered, or shared'
    ' between different apps.\n\n'
    '- Keep the summary short and in one line.\n'
    'Summary of this step: '
)

SUMMARIZATION_PROMPT_TEMPLATE1 = (
    PROMPT_PREFIX
    + '\nThe (overall) user goal/request is:{goal}\n'
    '\nNow I want you to summerize the latest step based on the action you'
    ' pick with the reason and descriptions for the before and after (the'
    ' action) screenshots.\n'
    '\nHere is a history of what you have done so far:\n{history}.'
    'Here is the description for the before'
    ' screenshot:\n{before_elements}\n'
    'Here is the description for the after screenshot:\n{after_elements}\n'
    'This is the action you picked: {action}\n'
    'Based on the reason: {reason}\n\n'
    '\nBy comparing the descriptions for the two screenshots and the action'
    ' performed, give a brief summary of this step.'
    #' This summary will be added to action history and used in future action'
    #' selection, so try to include essential information you think that will'
    #' be most useful for future action selection'
    ' Your summary should include:'
    ' [STEP SUMMARY] (a brief summary of current step). '
    ' [STEPS REFLECTION] (the reflection of history actions like "The history of my actions indicates that I have previously performed a certain action on the same interface", "The history of my actions indicates that I have gotten caught in a cycle of a specific action, so I need to take different and reasonable actions"). '
    ' [KEY STEPS NEXT] (what you intended to do, what might be the reason, what should/should not be done next and so on.)'
    '- If you think the task has been completed, finish the task by using the'
    ' status action with "complete" as goal_status.'
    #' `{{"action_type": "status", "goal_status": "complete"}}`\n'
    #' what you intended to do, why, if it worked as expected, if not'
    #' what might be the reason (be critical, the action/reason might not be'
    #' correct), what should/should not be done next and so on. '
    ' Some more rules/tips you must follow:\n'
    '- Keep it short and in one line.\n'
    "- Some actions (like `answer`, `wait`) don't involve screen change,"
    ' you can just assume they work as expected.\n'
    '- Given this summary will be added into action history, it can be used as'
    ' memory to include information that needs to be remembered, or shared'
    ' between different apps.\n\n'
    '- Keep the summary short and in one line.\n'
    'Summary of this step: '
)

GUIDANCE2 = (
    'Here are some useful guidelines you need to follow:\n'
    'Action Related\n'
    '- Use the `open_app` action whenever you want to open an app'
    ' (nothing will happen if the app is not installed), do not use the'
    ' app drawer to open an app unless all other ways have failed.\n'
    '- Use the `input_text` action whenever you want to type'
    ' something (including password) instead of clicking characters on the'
    ' keyboard one by one. Sometimes there is some default text in the text'
    ' field you want to type in, remember to delete them before typing.\n'
    '- For `click`, `long_press` and `input_text`, the index parameter you'
    ' pick must be VISIBLE in the screenshot and also in the UI element'
    ' list given to you (some elements in the list may NOT be visible on'
    ' the screen so you can not interact with them).\n'
    '- Consider exploring the screen by using the `scroll`'
    ' action with different directions to reveal additional content.\n'
    '- The direction parameter for the `scroll` action can be confusing'
    " sometimes as it's opposite to swipe, for example, to view content at the"
    ' bottom, the `scroll` direction should be set to "down". It has been'
    ' observed that you have difficulties in choosing the correct direction, so'
    ' if one does not work, try the opposite as well.\n'
    '\n\nNow output an action from the above list in the correct JSON format,'
    ' following the reason why you do that. Your answer should look like:\n'
    'Reason: ...\nAction: {{"action_type":...}}\n\n'
    'Your Answer:\n'
)

ANSWER_FORMAT = (
    '\n\nNow output an action from the above list in the correct JSON format,'
    ' following the reason why you do that. Your answer should look like:\n'
    'Reason: ...\nAction: {{"action_type":...}}\n\n'
    'Your answer:\n'
)

def extract_and_find_max_dict(text):
    pattern = r'\{[^}]*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    dicts = [json.loads(match) for match in matches]
    #print('>>>>>dicts:', dicts)
    
    max_dict = max(dicts, key=lambda d: len(d.keys()))
    
    return max_dict

def extract_json_objects(text):
    lost_data = 0
    json_list = []
    pattern = r'\{.*?\}'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            json_list.append(json.loads(match))
        except:
            lost_data += 1
            continue
        

    return json_list, lost_data

def extract_instructions_responses(text):
    pattern1 = re.compile(r'(?<=})\s*(?={)')
    json_strings = pattern1.split(text)

    pattern2 = re.compile(r'{"Instruction":\s*"([^"]+)",\s*"Response":\s*"([^"]+Action:\s*{[^}]+})"}')
    
    instru_respon_list = []
    for string in json_strings:
        matches = pattern2.findall(string)
        for match in matches:
            instru_respon_list.append([match[0], match[1]])
            #print('\n')
    
    return instru_respon_list

def validate_ui_element(
    ui_element
) -> bool:
  """Used to filter out invalid UI element."""
  screen_width, screen_height = (1080, 2400)

  # Filters out invisible element.
  if not ui_element.is_visible:
    return False

  # Filters out element with invalid bounding box.
  if ui_element.bbox_pixels:
    x_min = ui_element.bbox_pixels.x_min
    x_max = ui_element.bbox_pixels.x_max
    y_min = ui_element.bbox_pixels.y_min
    y_max = ui_element.bbox_pixels.y_max

    if (
        x_min >= x_max
        or x_min >= screen_width
        or x_max <= 0
        or y_min >= y_max
        or y_min >= screen_height
        or y_max <= 0
    ):
      return False

  return True

def process_steps(text):
    steps = re.findall(r'(Step \d+: Action selected: JSONAction\(.*?\)\.\s.*?\.\sReason:.*?)(?=Step \d+: Action selected: JSONAction|\Z)', text, re.DOTALL)
    
    processed_steps = []
    for step in steps:
        reason_match = re.search(r'Reason:(.*?)(?=Step \d+: Action selected: JSONAction|\Z)', step, re.DOTALL)
        if reason_match:
            reason_text = reason_match.group(1).strip()
            first_sentence = reason_text.split('.')[0] + '.'
            new_step = re.sub(r'Reason:.*?(?=Step \d+: Action selected: JSONAction|\Z)', f'Reason: {first_sentence}', step, flags=re.DOTALL)
            processed_steps.append(new_step)
    
    steps_text = 'Here is a history of what you have done so far:\n' + '\n\n'.join(processed_steps) + '\n\nHere is a list of descriptions for some UI elements on the current screen:'
    new_text = re.sub(r'Here is a history of what you have done so far:.*?Here is a list of descriptions for some UI elements on the current screen:', steps_text, text, flags=re.DOTALL)
    
    return new_text

def _generate_ui_elements_description_list_full(
    ui_elements
) -> str:
  
  tree_info = ''
  for index, ui_element in enumerate(ui_elements):
    if validate_ui_element(ui_element):
      tree_info += f'UI element {index}: {str(ui_element)}\n'
  return tree_info

def parse_reason_action_output(raw_reason_action_output: str,):
  reason_result = re.search(
      r'Reason:(.*)Score:', raw_reason_action_output, flags=re.DOTALL
  )
  reason = reason_result.group(1).strip() if reason_result else None
  action_result = re.search(
      r'Score:(.*)', raw_reason_action_output, flags=re.DOTALL
  )
  action = action_result.group(1).strip() if action_result else None
  return reason, action

def remove_UI_property(text):
    fields_to_remove = ['bbox', 'bbox_pixels']

    pattern = r',?\s*(%s)=(?:BoundingBox\([^)]*\)|[^,)]*)' % '|'.join(fields_to_remove)

    cleaned_text = re.sub(pattern, '', text)

    return cleaned_text

def remove_guidelines(text):
    start_index = text.find("Here are some useful guidelines you need to follow:")

    if start_index != -1:
        text = text[:start_index]

    return text

def _unzip_and_read_pickle(file_path: str):
    with open(file_path, 'rb') as f:
        compressed = f.read()

    with gzip.open(io.BytesIO(compressed), 'rb') as f_in:
        return pickle.load(f_in)

def process_pkl_gz_files(folder_path):
    agent0_data = []
    agent1_data = []
    agent2_data = []
    agent3_data = []
    agent_total = []
    general_task_data = []
    instru_task_data = []
    traj_task_data = []
    traj_dpo_task_data = []
    task_completed = []
    task_all = []
    episode_num = 0
    score_extract = 0
    score_dict_keys = 0
    query_len = []
    s = 0
    f = 0
    lost_simple_data = 0
    scores_None = 0
    nan_num = 0

    tasks_data = []
    #i = 0
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            print('Now:', dir)
            current_task = 0
            dir_path = os.path.join(root, dir)
            i = 0
            for root1, dirs1, files1 in os.walk(dir_path):
                print('file  size:', len(files1))
                
                for file in tqdm(files1):
                    if file.endswith('.pkl.gz'):
                        print('file:', file)

                        current_task_dict = {}
                        file_path = os.path.join(root1, file)
                        unzip_info = _unzip_and_read_pickle(file_path)
                        current_file_name = file[:-7] + '.json'
                        current_file_path = os.path.join('Agents_android_world_sample/run_MA_0921', current_file_name)
                        is_successful = unzip_info[0]['is_successful']
                        if is_successful:
                            i += 1 
                        goal = unzip_info[0]['goal']
                        task_template = unzip_info[0]['task_template']
                        episode_length = unzip_info[0]['episode_length']
                        #episode_num += episode_length
                        episode_data = unzip_info[0]['episode_data']
                        initial_action_prompt = episode_data['initial_action_prompt']
                        agent_plauactions_rewards = episode_data['agent_plauactions_rewards']
                        agent_actions_rewards = episode_data['agent_actions_rewards']
                        action_adapted = episode_data['action_adapt'] 
                        action_adapted_frequency = episode_data['action_adapt_frequency']
                        step_summary = episode_data['summary']

                        current_task_dict['goal'] = goal
                        current_task_dict['task_template'] = task_template
                        current_task_dict['is_successful'] = is_successful if is_successful else 0
                        current_task_dict['episode_length'] = episode_length
                        current_task_dict['initial_action_prompt'] = initial_action_prompt
                        current_task_dict['agent_plauactions_rewards'] = agent_plauactions_rewards
                        current_task_dict['agent_actions_rewards'] = agent_actions_rewards
                        current_task_dict['action_adapted'] = action_adapted
                        current_task_dict['action_adapted_frequency'] = action_adapted_frequency
                        current_task_dict['step_summary'] = step_summary


                        with open(current_file_path, 'w', encoding='utf-8') as json_file:
                            json.dump(current_task_dict, json_file, ensure_ascii=False, indent=4)
                
                        print('success:', i)

                        

    #return task_completed, task_all
    print('Done!')
    return i


folder_path = ''
i = process_pkl_gz_files(folder_path)
print('task_success:', i)

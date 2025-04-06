# Copyright 2024 The android_world Authors.
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

# 每一步都记录下Critic给出的reward

"""T3A: Text-only Autonomous Agent for Android."""

import time
import asyncio
from collections import Counter
import random
from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer_zz
from android_world.agents import infer_qwen
from android_world.agents import m3a_utils
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils
from GPTSwarm.swarm.graph.simplified_graph import SimpleGraph

import re

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

# task a
Summary_UI = (
    'What is the purpose of the current UI?\n'
    'Summarize the current interface in one paragraph.\n'
    'What does the current UI aim to achieve?\n'
    )

SUMMARY_UI_PROMPT = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    'Please answer the following questions for all the UI elements above.\n'
    'Questions = ('
    'What is the purpose of the current UI?'
    'Summarize the current interface in one paragraph.'
    'What does the current UI aim to achieve?)\n'
    'Please format your response as follows:\n'
    '{{"Question":"What is the purpose of the current UI?", "Answer":"........"}}\n'
    '{{"Question":"Summarize the current interface in one paragraph.", "Answer":"........"}}\n'
    '{{"Question":"What does the current UI aim to achieve?", "Answer":"........"}}\n'
    'Your response:\n'
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

# task b
INSTRU_ACTION_PROMPT = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    'The action space of the agent:\n'
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
    '- Wait for the screen to update: `{{"action_type": "wait"}}`\n\n'
    + GUIDANCE
    + '{additional_guidelines}'
    'Please propose diverse simple instructions (one-step tasks) as many as possible based on the agent\'s action space and the current UI elements above '
    'in the following format: (contains at least one but no more than two \'complete\' actions and no more than one \'answer\' action)\n'
    '{{"Instruction": "......", "Response": "Reason: ... Action: {{"action_type":...}}"}}\n'
    'For example: \n'
    '{{"Instruction": "I need to start recording audio", "Response": "Reason: The recording settings are all configured, I need to click \'Apply\' to apply the current settings and start recording. Action: {{"action_type": "click", "index": 3}}"}}\n'
    '{{"Instruction": "I want to select the M4a format for recording.", "Response": "Reason: The recording format has been set correctly. Action: {{"action_type": "status", "goal_status": "complete"}}"}}\n'
    'Your answer should look like:\n'
    '{{"Instruction": "......", "Response": "Reason: ... Action: {{"action_type":...}}"}}\n'
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
    ' Some more rules/tips you must follow:\n'
    '- Keep it short and in one line.\n'
    '- Keep it short and in one line.\n'
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

def remove_UI_property(text):
    fields_to_remove = ['bbox', 'bbox_pixels']

    pattern = r',?\s*(%s)=(?:BoundingBox\([^)]*\)|[^,)]*)' % '|'.join(fields_to_remove)

    cleaned_text = re.sub(pattern, '', text)

    return cleaned_text

def _generate_ui_elements_description_list_full(
    ui_elements: list[representation_utils.UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
  """Generate description for a list of UIElement using full information.

  Args:
    ui_elements: UI elements for the current screen.
    screen_width_height_px: Logical screen size.

  Returns:
    Information for each UIElement.
  """
  tree_info = ''
  for index, ui_element in enumerate(ui_elements):
    if m3a_utils.validate_ui_element(ui_element, screen_width_height_px):
      tree_info += f'UI element {index}: {str(ui_element)}\n'
  return tree_info


def _action_selection_prompt(
    goal: str,
    history: list[str],
    last_history,
    ui_elements_description: str,
    additional_guidelines: list[str] | None = None,
) -> str:
  
  if history:
    history = '\n'.join(history)
  else:
    history = 'You just started, no action has been performed yet.'

  extra_guidelines = ''
  if additional_guidelines:
    extra_guidelines = 'For The Current Task:\n'
    for guideline in additional_guidelines:
      extra_guidelines += f'- {guideline}\n'

  return ACTION_SELECTION_PROMPT_TEMPLATE.format(
      history=history,
      goal=goal,
      last_history=last_history,
      ui_elements_description=ui_elements_description
      if ui_elements_description
      else 'Not available',
      additional_guidelines=extra_guidelines,
  )

def plausible_action_selection_prompt(
    goal: str,
    history: list[str],
    last_history,
    ui_elements_description: str,
    additional_guidelines: list[str] | None = None,
) -> str:
  
  if history:
    history = '\n'.join(history)
  else:
    history = 'You just started, no action has been performed yet.'

  extra_guidelines = ''
  if additional_guidelines:
    extra_guidelines = 'For The Current Task:\n'
    for guideline in additional_guidelines:
      extra_guidelines += f'- {guideline}\n'

  return PLAUSIBLE_ACTION_SELECTION_PROMPT_TEMPLATE.format(
      history=history,
      goal=goal,
      last_history=last_history,
      ui_elements_description=ui_elements_description
      if ui_elements_description
      else 'Not available',
      additional_guidelines=extra_guidelines,
  )

def get_most_common_key(key, dict_list):
  keys = [d.key for d in dict_list]
  keys_counter = Counter(keys)
  max_frequency = max(keys_counter.values())
  most_common_keys = [key for key, freq in keys_counter.items() if freq == max_frequency]
  selected_key = random.choice(most_common_keys)
  most_common_dicts_with_indices = [(index, d) for index, d in enumerate(dict_list) if d.key == selected_key]
  return max_frequency, most_common_dicts_with_indices 

def get_most_common_action_type_with_indices(dict_list):
  action_types = [d.action_type for d in dict_list]
  counter = Counter(action_types)
  max_frequency = max(counter.values())
  most_common_action_types = [action for action, freq in counter.items() if freq == max_frequency]
  selected_action_type = random.choice(most_common_action_types)
  most_common_dicts_with_indices = [(index, d) for index, d in enumerate(dict_list) if d.action_type == selected_action_type]

  if selected_action_type in ['click', 'long_press', 'input_text']:
      indices = [d.index for idx, d in most_common_dicts_with_indices]
      index_counter = Counter(indices)
      max_frequency = max(index_counter.values())
      most_common_indices = [index for index, freq in index_counter.items() if freq == max_frequency]
      selected_index = random.choice(most_common_indices)
      most_common_dicts = [(i, d) for i, d in most_common_dicts_with_indices if d.index == selected_index]

      if selected_action_type == 'input_text':
        texts = [d.text for idx, d in most_common_dicts]
        text_counter = Counter(texts)
        max_frequency = max(text_counter.values())
        most_common_texts = [text for text, freq in text_counter.items() if freq == max_frequency]
        selected_text = random.choice(most_common_texts)
        most_common_dicts_ = [(i, d) for i, d in most_common_dicts if d.text == selected_text]
      else:
        most_common_dicts_ = most_common_dicts
  else:
    most_common_dicts_ = most_common_dicts_with_indices

  return max_frequency, most_common_dicts_[0]


def _summarize_prompt(
    goal: str,
    history,
    action: str,
    reason: str,
    before_elements: str,
    after_elements: str,
) -> str:
  """Generate the prompt for the summarization step.

  Args:
    goal: The overall goal.
    action: The action picked for the step.
    reason: The reason why pick the action.
    before_elements: Information for UI elements on the before screenshot.
    after_elements: Information for UI elements on the after screenshot.

  Returns:
    The text prompt for summarization that will be sent to gpt4v.
  """
  return SUMMARIZATION_PROMPT_TEMPLATE.format(
      goal=goal,
      history=history,
      action=action,
      reason=reason,
      before_elements=before_elements if before_elements else 'Not available',
      after_elements=after_elements if after_elements else 'Not available',
  )


class T3A(base_agent.EnvironmentInteractingAgent):
  """Text only autonomous agent for Android."""

  # Wait a few seconds for the screen to stablize after executing an action.
  WAIT_AFTER_ACTION_SECONDS = 2.0

  def __init__(
      self,
      env: interface.AsyncEnv,
      #llm: infer.LlmWrapper,
      graph_agent,
      name: str = 'Agent_group',  
  ):
    """Initializes a RandomAgent.

    Args:
      env: The environment.
      llm: The text only LLM.
      name: The agent name.
    """
    super().__init__(env, name)
    self.graph_agent = graph_agent
    self.llm = infer_zz.Gpt4Wrapper('gpt-4o')
    #self.llm = infer_qwen.Gpt4Wrapper('qwen')
    self.history = []
    self.additional_guidelines = None

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self.history = []

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  async def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_data = {
        'before_screenshot': None,
        'after_screenshot': None,
        'before_element_list': None,
        'after_element_list': None,
        'initial_action_prompt': None, 
        'action_output_list': None,
        'action_list': None,
        'reason_list': None,
        'action_raw_response_list': None,
        'action_adapt': None,
        'action_adapt_frequency': None,
        'summary_prompt': None,
        'summary': None,
        'summary_raw_response': None,
        'agent_step_scores': None,
        'step_scores_prompt': None,
        'summary_UI': None,
        'query_UI': None,
        'instru_action': None,
        'reflect': None,
    }
    print('\n----------step ' + str(len(self.history) + 1))
    time.sleep(2)
    state = self.get_post_transition_state()
    #print('\n================state:', state)
    logical_screen_size = self.env.logical_screen_size
    #print('\n================logical_screen_size:', logical_screen_size)

    ui_elements = state.ui_elements
    before_element_list = _generate_ui_elements_description_list_full(
        ui_elements,
        logical_screen_size,
    )
    # 去除UI多余属性
    before_element_list = remove_UI_property(before_element_list)
    step_data['before_screenshot'] = state.pixels.copy()
    step_data['before_element_list'] = ui_elements

    history_str = [
            '\nStep ' + str(i + 1) + ': ' + step_info['summary']
            for i, step_info in enumerate(self.history)
        ]    
    #print('====history:', history_str)
    # 设计节点之间的交互
    action_prompt = _action_selection_prompt(
        goal,
        history_str,
        history_str[-1] if history_str else 'You just started, no action has been performed yet.',
        before_element_list,
        self.additional_guidelines,
    )

    plausible_action_prompt = plausible_action_selection_prompt(
        goal,
        history_str,
        history_str[-1] if history_str else 'You just started, no action has been performed yet.',
        before_element_list,
        self.additional_guidelines,
    )

    step_data['initial_action_prompt'] = action_prompt

    answers = await self.graph_agent.evaluate([action_prompt, action_prompt]) 
    # action_output, raw_response = "output", "raw_response"
    action_output_list = [answer["output"] for answer in answers]
    raw_response_list = [answer["raw_response"] for answer in answers]
    
    """ action_output, raw_response = self.llm.predict(
        action_prompt,
    )
    if not raw_response:
      raise RuntimeError('Error calling LLM in action selection phase.') """

    step_data['action_output_list'] = action_output_list
    step_data['action_raw_response'] = raw_response_list

    converted_action_list = []
    action_list = []
    reason_list = []
    for idx, action_output in enumerate(action_output_list):
      reason, action = m3a_utils.parse_reason_action_output(action_output)
      
      print('\n=====Action{}:{} '.format(idx, action))
      print('=====Reason{}:{} '.format(idx, reason))
      print('==========================')
      reason_list.append(reason)
      action_list.append(action)
      try:
        converted_action = json_action.JSONAction(
            **agent_utils.extract_json(action),
        )
        converted_action_list.append(converted_action)
      
      except Exception as e:  # pylint: disable=broad-exception-caught
        print('Failed to convert the agent{}\'s output to a valid action.'.format(idx))
        print('======action_output{}:{} '.format(idx, action_output))
        print(str(e))
        """ step_data['summary'] = (
            'Can not parse the output to a valid action. Please make sure to pick'
            ' the action from the list with the correct json format!'
        ) """
    step_data['action_list'] = action_list
    step_data['reason_list'] = reason_list

    agents_actions = [
            'agent_' + str(i) + ': ' + agent_step_info
            #for i, agent_step_info in enumerate(action_list)
            for i, agent_step_info in enumerate(action_output_list)
        ]
    agents_actions = '\n'.join(agents_actions)
    step_history= [
            'Step ' + str(i + 1) + ': ' + step_info['summary']
            for i, step_info in enumerate(self.history)
        ]
    step_history = '\n'.join(step_history)

    score_prompt = CRITIC_PROMPT_TEMPLATE.format(
      history=step_history,
      goal=goal,
      ui_elements_description=before_element_list
      if before_element_list
      else 'Not available',
      agents_actions=agents_actions,
      )

    print('\nEvaluating Agent step Scores----')
    scores, raw_score_response = self.llm.predict(
        score_prompt,
    )

    step_data['agent_step_scores'] = scores
    step_data['step_scores_prompt'] = score_prompt
    print('Agent step Scores:', scores)

    converted_action_adapt = converted_action_list[0]
    converted_action_reason = reason_list[0]
    step_data['action_adapt'] = converted_action_adapt.action_type
    step_data['action_adapt_frequency'] = 1
    print('------action_adapt:{}, {}'.format(converted_action_adapt, 1) )

    if converted_action_adapt.action_type in ['click', 'long-press', 'input-text']:
      if converted_action_adapt.index is not None and converted_action_adapt.index >= len(
          ui_elements
      ):
        print('Index out of range.')
        step_data['summary'] = (
            'The parameter index is out of range. Remember the index must be in'
            ' the UI element list!'
        )
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, 
                                                step_data, 
                                                #self.history
                                                )
      else:
        # Add mark for the target ui element, just used for visualization.
        m3a_utils.add_ui_element_mark(
            step_data['before_screenshot'],
            ui_elements[converted_action_adapt.index],
            converted_action_adapt.index,
            logical_screen_size,
            adb_utils.get_physical_frame_boundary(self.env.base_env),
            adb_utils.get_orientation(self.env.base_env),
        )

    if converted_action_adapt.action_type == 'status':
      if converted_action_adapt.goal_status == 'infeasible':
        print('Agent stopped since it thinks mission impossible.')
      step_data['summary'] = 'Agent thinks the request has been completed.'
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(
          True,
          step_data,
          #self.history
      )

    if converted_action_adapt.action_type == 'answer':
      print('Agent answered with: ' + converted_action_adapt.text)

    try:
      self.env.execute_action(converted_action_adapt)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          'Some error happened executing the action ',
          converted_action_adapt.action_type,
      )
      print(str(e))
      step_data['summary'] = (
          'Some error happened executing the action '
          + converted_action_adapt.action_type
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
          #self.history
      )

    time.sleep(self.WAIT_AFTER_ACTION_SECONDS)

    state = self.env.get_state()
    ui_elements = state.ui_elements
    after_element_list = _generate_ui_elements_description_list_full(
        ui_elements,
        self.env.logical_screen_size,
    )

    # Save screenshot only for result visualization.
    step_data['after_screenshot'] = state.pixels.copy()
    step_data['after_element_list'] = ui_elements

    # task e
    Reflect_prompt = REFLECT_PROMPT.format(goal=goal, history=history_str)
    Reflect, raw_response = self.llm.predict(
        Reflect_prompt,
    )
    step_data['reflect'] = Reflect
    print('>>>>>>Reflect: Done')


    # step summary

    summary_prompt = _summarize_prompt(
        goal,
        [
            '\nStep ' + str(i + 1) + ': ' + step_info['summary']
            for i, step_info in enumerate(self.history)
        ],
        converted_action_adapt,
        converted_action_reason,
        before_element_list,
        after_element_list,
    )

    summary, raw_response = self.llm.predict(
        summary_prompt,
    )

    print('----Step summary:', f'Action selected: {converted_action_adapt}.\n Summary: {summary}.\n Reflect: {Reflect}')
    step_data['summary_prompt'] = summary_prompt
    step_data['summary'] = (
        f'Action selected: {converted_action_adapt}. {summary}. {Reflect}'
        #f'Action selected: {converted_action_adapt}'
        if raw_response
        else 'Error calling LLM in summerization phase.'
    )
    step_data['summary_raw_response'] = raw_response

    self.history.append(step_data)
    
    return base_agent.AgentInteractionResult(
        False,
        step_data,
        #self.history
    )
